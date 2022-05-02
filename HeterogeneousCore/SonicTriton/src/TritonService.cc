#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "grpc_client.h"
#include "grpc_service.pb.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <utility>
#include <tuple>
#include <unistd.h>

namespace tc = triton::client;

const std::string TritonService::Server::fallbackName{"fallback"};
const std::string TritonService::Server::fallbackAddress{"0.0.0.0"};

namespace {
  std::pair<std::string, int> execSys(const std::string& cmd) {
    //redirect stderr to stdout
    auto pipe = popen((cmd + " 2>&1").c_str(), "r");
    int thisErrno = errno;
    if (!pipe)
      throw cms::Exception("SystemError")
          << "TritonService: popen() failed with errno " << thisErrno << " for command: " << cmd;

    //extract output
    constexpr static unsigned buffSize = 128;
    std::array<char, buffSize> buffer;
    std::string result;
    while (!feof(pipe)) {
      if (fgets(buffer.data(), buffSize, pipe))
        result += buffer.data();
      else {
        thisErrno = ferror(pipe);
        if (thisErrno)
          throw cms::Exception("SystemError")
              << "TritonService: failed reading command output with errno " << thisErrno;
      }
    }

    int rv = pclose(pipe);
    return std::make_pair(result, rv);
  }
}  // namespace

TritonService::TritonService(const edm::ParameterSet& pset, edm::ActivityRegistry& areg)
    : verbose_(pset.getUntrackedParameter<bool>("verbose")),
      fallbackOpts_(pset.getParameterSet("fallback")),
      currentModuleId_(0),
      allowAddModel_(false),
      startedFallback_(false),
      pid_(std::to_string(::getpid())) {
  //module construction is assumed to be serial (correct at the time this code was written)
  areg.watchPreModuleConstruction(this, &TritonService::preModuleConstruction);
  areg.watchPostModuleConstruction(this, &TritonService::postModuleConstruction);
  areg.watchPreModuleDestruction(this, &TritonService::preModuleDestruction);
  //fallback server will be launched (if needed) before beginJob
  areg.watchPreBeginJob(this, &TritonService::preBeginJob);

  //include fallback server in set if enabled
  if (fallbackOpts_.enable) {
    auto serverType = TritonServerType::Remote;
    if (!fallbackOpts_.useGPU)
      serverType = TritonServerType::LocalCPU;
#ifdef TRITON_ENABLE_GPU
    else
      serverType = TritonServerType::LocalGPU;
#endif

    servers_.emplace(std::piecewise_construct,
                     std::forward_as_tuple(Server::fallbackName),
                     std::forward_as_tuple(Server::fallbackName, Server::fallbackAddress, serverType));
  }

  //loop over input servers: check which models they have
  std::string msg;
  if (verbose_)
    msg = "List of models for each server:\n";
  for (const auto& serverPset : pset.getUntrackedParameterSetVector("servers")) {
    const std::string& serverName(serverPset.getUntrackedParameter<std::string>("name"));
    //ensure uniqueness
    auto [sit, unique] = servers_.emplace(serverName, serverPset);
    if (!unique)
      throw cms::Exception("DuplicateServer")
          << "TritonService: Not allowed to specify more than one server with same name (" << serverName << ")";
    auto& server(sit->second);

    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    TRITON_THROW_IF_ERROR(
        tc::InferenceServerGrpcClient::Create(&client, server.url, false, server.useSsl, server.sslOptions),
        "TritonService(): unable to create inference context for " + serverName + " (" + server.url + ")");

    if (verbose_) {
      inference::ServerMetadataResponse serverMetaResponse;
      TRITON_THROW_IF_ERROR(client->ServerMetadata(&serverMetaResponse),
                            "TritonService(): unable to get metadata for " + serverName + " (" + server.url + ")");
      edm::LogInfo("TritonService") << "Server " << serverName << ": url = " << server.url
                                    << ", version = " << serverMetaResponse.version();
    }

    inference::RepositoryIndexResponse repoIndexResponse;
    TRITON_THROW_IF_ERROR(
        client->ModelRepositoryIndex(&repoIndexResponse),
        "TritonService(): unable to get repository index for " + serverName + " (" + server.url + ")");

    //servers keep track of models and vice versa
    if (verbose_)
      msg += serverName + ": ";
    for (const auto& modelIndex : repoIndexResponse.models()) {
      const auto& modelName = modelIndex.name();
      auto mit = models_.find(modelName);
      if (mit == models_.end())
        mit = models_.emplace(modelName, "").first;
      auto& modelInfo(mit->second);
      modelInfo.servers.insert(serverName);
      server.models.insert(modelName);
      if (verbose_)
        msg += modelName + ", ";
    }
    if (verbose_)
      msg += "\n";
  }
  if (verbose_)
    edm::LogInfo("TritonService") << msg;
}

void TritonService::preModuleConstruction(edm::ModuleDescription const& desc) {
  currentModuleId_ = desc.id();
  allowAddModel_ = true;
}

void TritonService::addModel(const std::string& modelName, const std::string& path) {
  //should only be called in module constructors
  if (!allowAddModel_)
    throw cms::Exception("DisallowedAddModel")
        << "TritonService: Attempt to call addModel() outside of module constructors";
  //if model is not in the list, then no specified server provides it
  auto mit = models_.find(modelName);
  if (mit == models_.end()) {
    auto& modelInfo(unservedModels_.emplace(modelName, path).first->second);
    modelInfo.modules.insert(currentModuleId_);
    //only keep track of modules that need unserved models
    modules_.emplace(currentModuleId_, modelName);
  }
}

void TritonService::postModuleConstruction(edm::ModuleDescription const& desc) { allowAddModel_ = false; }

void TritonService::preModuleDestruction(edm::ModuleDescription const& desc) {
  //remove destructed modules from unserved list
  if (unservedModels_.empty())
    return;
  auto id = desc.id();
  auto oit = modules_.find(id);
  if (oit != modules_.end()) {
    const auto& moduleInfo(oit->second);
    auto mit = unservedModels_.find(moduleInfo.model);
    if (mit != unservedModels_.end()) {
      auto& modelInfo(mit->second);
      modelInfo.modules.erase(id);
      //remove a model if it is no longer needed by any modules
      if (modelInfo.modules.empty())
        unservedModels_.erase(mit);
    }
    modules_.erase(oit);
  }
}

//second return value is only true if fallback CPU server is being used
TritonService::Server TritonService::serverInfo(const std::string& model, const std::string& preferred) const {
  auto mit = models_.find(model);
  if (mit == models_.end())
    throw cms::Exception("MissingModel") << "TritonService: There are no servers that provide model " << model;
  const auto& modelInfo(mit->second);
  const auto& modelServers = modelInfo.servers;

  auto msit = modelServers.end();
  if (!preferred.empty()) {
    msit = modelServers.find(preferred);
    //todo: add a "strict" parameter to stop execution if preferred server isn't found?
    if (msit == modelServers.end())
      edm::LogWarning("PreferredServer") << "Preferred server " << preferred << " for model " << model
                                         << " not available, will choose another server";
  }
  const auto& serverName(msit == modelServers.end() ? *modelServers.begin() : preferred);

  //todo: use some algorithm to select server rather than just picking arbitrarily
  const auto& server(servers_.find(serverName)->second);
  return server;
}

void TritonService::preBeginJob(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const&) {
  //only need fallback if there are unserved models
  if (!fallbackOpts_.enable or unservedModels_.empty())
    return;

  std::string msg;
  if (verbose_)
    msg = "List of models for fallback server: ";
  //all unserved models are provided by fallback server
  auto& server(servers_.find(Server::fallbackName)->second);
  for (const auto& [modelName, model] : unservedModels_) {
    auto& modelInfo(models_.emplace(modelName, model).first->second);
    modelInfo.servers.insert(Server::fallbackName);
    server.models.insert(modelName);
    if (verbose_)
      msg += modelName + ", ";
  }
  if (verbose_)
    edm::LogInfo("TritonService") << msg;

  //assemble server start command
  std::string command("cmsTriton -P -1 -p " + pid_);
  if (fallbackOpts_.debug)
    command += " -c";
  if (fallbackOpts_.verbose)
    command += " -v";
  if (fallbackOpts_.useDocker)
    command += " -d";
  if (fallbackOpts_.useGPU)
    command += " -g";
  if (!fallbackOpts_.instanceName.empty())
    command += " -n " + fallbackOpts_.instanceName;
  if (fallbackOpts_.retries >= 0)
    command += " -r " + std::to_string(fallbackOpts_.retries);
  if (fallbackOpts_.wait >= 0)
    command += " -w " + std::to_string(fallbackOpts_.wait);
  for (const auto& [modelName, model] : unservedModels_) {
    command += " -m " + model.path;
  }
  if (!fallbackOpts_.imageName.empty())
    command += " -i " + fallbackOpts_.imageName;
  if (!fallbackOpts_.sandboxName.empty())
    command += " -s " + fallbackOpts_.sandboxName;
  //don't need this anymore
  unservedModels_.clear();

  //get a random temporary directory if none specified
  if (fallbackOpts_.tempDir.empty()) {
    auto tmp_dir_path{std::filesystem::temp_directory_path() /= edm::createGlobalIdentifier()};
    fallbackOpts_.tempDir = tmp_dir_path.string();
  }
  //special case ".": use script default (temp dir = .$instanceName)
  if (fallbackOpts_.tempDir != ".")
    command += " -t " + fallbackOpts_.tempDir;

  command += " start";

  if (fallbackOpts_.debug)
    edm::LogInfo("TritonService") << "Fallback server temporary directory: " << fallbackOpts_.tempDir;
  if (verbose_)
    edm::LogInfo("TritonService") << command;

  //mark as started before executing in case of ctrl+c while command is running
  startedFallback_ = true;
  const auto& [output, rv] = execSys(command);
  if (verbose_)
    edm::LogInfo("TritonService") << output;
  if (rv != 0) {
    edm::LogError("TritonService") << output;
    throw cms::Exception("FallbackFailed")
        << "TritonService: Starting the fallback server failed with exit code " << rv;
  }
  //get the port
  const std::string& portIndicator("CMS_TRITON_GRPC_PORT: ");
  //find last instance in log in case multiple ports were tried
  auto pos = output.rfind(portIndicator);
  if (pos != std::string::npos) {
    auto pos2 = pos + portIndicator.size();
    auto pos3 = output.find('\n', pos2);
    const auto& portNum = output.substr(pos2, pos3 - pos2);
    server.url += ":" + portNum;
  } else
    throw cms::Exception("FallbackFailed") << "TritonService: Unknown port for fallback server, log follows:\n"
                                           << output;
}

void TritonService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("verbose", false);

  edm::ParameterSetDescription validator;
  validator.addUntracked<std::string>("name");
  validator.addUntracked<std::string>("address");
  validator.addUntracked<unsigned>("port");
  validator.addUntracked<bool>("useSsl", false);
  validator.addUntracked<std::string>("rootCertificates", "");
  validator.addUntracked<std::string>("privateKey", "");
  validator.addUntracked<std::string>("certificateChain", "");

  desc.addVPSetUntracked("servers", validator, {});

  edm::ParameterSetDescription fallbackDesc;
  fallbackDesc.addUntracked<bool>("enable", false);
  fallbackDesc.addUntracked<bool>("debug", false);
  fallbackDesc.addUntracked<bool>("verbose", false);
  fallbackDesc.addUntracked<bool>("useDocker", false);
  fallbackDesc.addUntracked<bool>("useGPU", false);
  fallbackDesc.addUntracked<int>("retries", -1);
  fallbackDesc.addUntracked<int>("wait", -1);
  fallbackDesc.addUntracked<std::string>("instanceBaseName", "triton_server_instance");
  fallbackDesc.addUntracked<std::string>("instanceName", "");
  fallbackDesc.addUntracked<std::string>("tempDir", "");
  fallbackDesc.addUntracked<std::string>("imageName", "");
  fallbackDesc.addUntracked<std::string>("sandboxName", "");
  desc.add<edm::ParameterSetDescription>("fallback", fallbackDesc);

  descriptions.addWithDefaultLabel(desc);
}
