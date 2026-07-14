#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/GetEnvironmentVariable.h"

#include "grpc_client.h"
#include "grpc_service.pb.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <utility>
#include <tuple>
#include <unistd.h>
#include <format>

namespace tc = triton::client;

const std::string TritonService::Server::fallbackName{"fallback"};
const std::string TritonService::Server::fallbackAddress{"0.0.0.0"};
const std::string TritonService::Server::siteconfName{"SONIC_LOCAL_BALANCER"};

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

  //extract specific info from log
  std::string extractFromLog(const std::string& output, const std::string& indicator) {
    //find last instance in log (in case of multiple)
    auto pos = output.rfind(indicator);
    if (pos != std::string::npos) {
      auto pos2 = pos + indicator.size();
      auto pos3 = output.find('\n', pos2);
      return output.substr(pos2, pos3 - pos2);
    } else
      return "";
  }
}  // namespace

TritonService::TritonService(const edm::ParameterSet& pset, edm::ActivityRegistry& areg)
    : verbose_(pset.getUntrackedParameter<bool>("verbose")),
      fallbackOpts_(pset.getParameterSet("fallback")),
      currentModuleId_(0),
      allowAddModel_(false),
      startedFallback_(false),
      callFails_(0),
      pid_(std::to_string(::getpid())) {
  //module construction is assumed to be serial (correct at the time this code was written)

  areg.watchPreallocate(this, &TritonService::preallocate);

  areg.watchPreModuleConstruction(this, &TritonService::preModuleConstruction);
  areg.watchPostModuleConstruction(this, &TritonService::postModuleConstruction);
  areg.watchPreModuleDestruction(this, &TritonService::preModuleDestruction);
  //fallback server will be launched (if needed) before beginJob
  areg.watchPreBeginJob(this, &TritonService::preBeginJob);
  areg.watchPostEndJob(this, &TritonService::postEndJob);

  //check for server specified in SITECONF
  //(temporary solution, to be replaced with entry in site-local-config.xml or similar)
  std::string siteconf_address(edm::getEnvironmentVariable(Server::siteconfName + "_HOST"));
  std::string siteconf_port(edm::getEnvironmentVariable(Server::siteconfName + "_PORT"));
  if (!siteconf_address.empty() and !siteconf_port.empty()) {
    servers_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(Server::siteconfName),
        std::forward_as_tuple(Server::siteconfName, siteconf_address + ":" + siteconf_port, TritonServerType::Remote));
    if (verbose_)
      edm::LogInfo("TritonDiscovery") << "Obtained server from SITECONF: "
                                      << servers_.find(Server::siteconfName)->second.url;
  } else if (siteconf_address.empty() != siteconf_port.empty()) {  //xor
    edm::LogWarning("TritonDiscovery") << "Incomplete server information from SITECONF: HOST = " << siteconf_address
                                       << ", PORT = " << siteconf_port;
  } else
    edm::LogWarning("TritonDiscovery") << "No server information from SITECONF";

  //finally, populate list of servers from config input
  for (const auto& serverPset : pset.getUntrackedParameterSetVector("servers")) {
    const std::string& serverName(serverPset.getUntrackedParameter<std::string>("name"));
    //ensure uniqueness
    auto [sit, unique] = servers_.emplace(serverName, serverPset);
    if (!unique)
      throw cms::Exception("DuplicateServer")
          << "TritonService: Not allowed to specify more than one server with same name (" << serverName << ")";
  }

  //loop over all servers: check which models they have, populate serverHealth
  std::string msg;
  if (verbose_)
    msg = "List of models for each server:\n";
  for (auto& [serverName, server] : servers_) {
    //populate serverHealth
    serversHealth_.emplace(serverName, ServerHealth{});

    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    TRITON_THROW_IF_ERROR(
        tc::InferenceServerGrpcClient::Create(&client, server.url, false, server.useSsl, server.sslOptions),
        "TritonService(): unable to create inference context for " + serverName + " (" + server.url + ")",
        nullptr);

    if (verbose_) {
      inference::ServerMetadataResponse serverMetaResponse;
      auto err = client->ServerMetadata(&serverMetaResponse);
      if (err.IsOk())
        edm::LogInfo("TritonService") << "Server " << serverName << ": url = " << server.url
                                      << ", version = " << serverMetaResponse.version();
      else
        edm::LogInfo("TritonService") << "unable to get metadata for " + serverName + " (" + server.url + ")"
                                      << err.Message();
    }

    //if this query fails, it indicates that the server is nonresponsive or saturated
    //in which case it should just be skipped
    inference::RepositoryIndexResponse repoIndexResponse;
    auto err = client->ModelRepositoryIndex(&repoIndexResponse);

    //servers keep track of models and vice versa
    if (verbose_)
      msg += serverName + ": ";
    if (err.IsOk()) {
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
    } else {
      const std::string& baseMsg = "unable to get repository index";
      const std::string& extraMsg = err.Message().empty() ? "" : ": " + err.Message();
      if (verbose_)
        msg += baseMsg + extraMsg;
      else
        edm::LogWarning("TritonFailure") << "TritonService(): " << baseMsg << " for " << serverName << " ("
                                         << server.url << ")" << extraMsg;
    }
    if (verbose_)
      msg += "\n";
  }
  if (verbose_)
    edm::LogInfo("TritonDiscovery") << msg;
}

void TritonService::preallocate(edm::service::SystemBounds const& bounds) {
  numberOfThreads_ = bounds.maxNumberOfThreads();
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

  auto& modelInfo(models_.emplace(modelName, path).first->second);
  // Update path if model was previously added (e.g., by server scanning) with empty path
  if (modelInfo.path.empty() && !path.empty())
    modelInfo.path = path;

  modelInfo.modules.insert(currentModuleId_);
  modules_.emplace(currentModuleId_, modelName);
}

void TritonService::postModuleConstruction(edm::ModuleDescription const& desc) { allowAddModel_ = false; }

void TritonService::preModuleDestruction(edm::ModuleDescription const& desc) {
  auto id = desc.id();
  auto oit = modules_.find(id);
  if (oit != modules_.end()) {
    const auto& moduleInfo(oit->second);
    auto mit = models_.find(moduleInfo.model);
    if (mit != models_.end()) {
      auto& modelInfo(mit->second);
      modelInfo.modules.erase(id);
    }
    modules_.erase(oit);
  }
}

// Returns the name of the server assigned to serve the given model, or nullptr if no server is currently assigned.
// If a preferred server(current server) is specified but unavailable, falls back to any assigned server.
// Callers are responsible for handling the nullptr case.
const std::string* TritonService::resolveServerName(const std::string& model, const std::string& preferred) const {
  auto mit = models_.find(model);
  if (mit == models_.end() || mit->second.servers.empty())
    return nullptr;  // no server assigned - caller decides what to do

  const auto& modelServers = mit->second.servers;

  if (!preferred.empty()) {
    auto msit = modelServers.find(preferred);
    if (msit != modelServers.end())
      return &(*msit);
    edm::LogWarning("PreferredServer") << "Preferred server " << preferred << " for model " << model
                                       << " not available, will choose another server";
  }
  //Prefer remote servers over fallback if available
  if (modelServers.size() > 1) {
    auto rit = std::find_if(modelServers.begin(), modelServers.end(), [this](const std::string& name) {
      auto sit = servers_.find(name);
      return sit != servers_.end() && !sit->second.isFallback;
    });
    if (rit != modelServers.end())
      return &(*rit);
  }
  return &(*modelServers.begin());
}

// Returns the full server info for the server assigned to serve the given model.
// Throws MissingModel if no server is currently assigned.
// Wraps resolveServerName; use that directly if nullptr should be handled by the caller
const std::pair<const std::string, TritonService::Server>& TritonService::resolveServer(
    const std::string& model, const std::string& preferred) const {
  const auto* name = resolveServerName(model, preferred);
  if (!name)
    throw cms::Exception("MissingModel") << "TritonService: There are no servers that provide model " << model;
  return *servers_.find(*name);
}

// Returns the list of model names that are not currently assigned to any server.
std::vector<std::string> TritonService::unassignedModels() const {
  std::vector<std::string> result;
  for (const auto& [name, info] : models_) {
    if (info.servers.empty())
      result.push_back(name);
  }
  return result;
}

void TritonService::updateServerHealth(const std::string& modelName) const {
  for (auto& [serverName, server] : servers_) {
    edm::LogInfo("TritonService") << "Updating server health for server = " << serverName;
    if (server.isFallback) {
      edm::LogInfo("TritonService") << serverName << " is skipped because it is a fallback server";
      continue;  // fallback is a last resort, not a candidate for getBestServer
    }
    try {
      std::unique_ptr<tc::InferenceServerGrpcClient> client;
      TRITON_THROW_IF_ERROR(
          tc::InferenceServerGrpcClient::Create(&client, server.url, false, server.useSsl, server.sslOptions),
          "TritonService(): unable to create inference context for " + serverName + " (" + server.url + ")");

      bool live = false, ready = false;
      TRITON_THROW_IF_ERROR(client->IsServerLive(&live),
                            "TritonService(): unable to query IsServerLive " + serverName + " (" + server.url + ")");
      TRITON_THROW_IF_ERROR(client->IsServerReady(&ready),
                            "TritonService(): unable to query IsServerReady " + serverName + " (" + server.url + ")");

      edm::LogInfo("TritonService") << serverName << " : live = " << live << " ready = " << ready;

      inference::ModelStatisticsResponse stats;
      if (!modelName.empty()) {
        client->ModelInferenceStatistics(&stats, modelName);
      } else {
        for (const auto& m : server.models) {
          client->ModelInferenceStatistics(&stats, m);
        }
      }

      uint64_t infer_count = 0, queue_count = 0, failures = 0;
      double avgQueueTimeMs = 0.0;
      double avgInferTimeMs = 0.0;

      for (const auto& mstat : stats.model_stats()) {
        if (modelName.empty() || mstat.name() == modelName) {
          const auto& infer = mstat.inference_stats();

          infer_count += infer.compute_infer().count();
          avgInferTimeMs += infer.compute_infer().ns() / 1e3;
          queue_count += infer.queue().count();
          avgQueueTimeMs += infer.queue().ns() / 1e3;
          failures += infer.fail().count();
        }
      }
      // Update health map safely with accessor
      tbb::concurrent_hash_map<std::string, ServerHealth>::accessor acc;
      serversHealth_.find(acc, serverName);

      ServerHealth& health = acc->second;
      health.live = live;
      health.ready = ready;
      health.failureCount = failures;
      health.avgQueueTimeMs = (queue_count > 0) ? avgQueueTimeMs / queue_count : 0.0;
      health.avgInferTimeMs = (infer_count > 0) ? avgInferTimeMs / infer_count : 0.0;

    } catch (const std::exception& e) {
      // mark existing entry unhealthy if present
      tbb::concurrent_hash_map<std::string, ServerHealth>::accessor acc;
      if (serversHealth_.find(acc, serverName)) {
        ServerHealth& health = acc->second;
        health.live = false;
        health.ready = false;
      }
    }
  }
}

std::optional<std::string> TritonService::getBestServer(const std::string& modelName,
                                                        const std::string& ignoreServer) const {
  std::optional<std::string> bestServerName;
  ServerHealth bestHealth;

  // get fresh ServerHealth statistics
  updateServerHealth(modelName);
  edm::LogInfo("TritonService") << "Getting best server";

  for (auto& [serverName, server] : servers_) {
    if (serverName == ignoreServer) {
      edm::LogInfo("TritonService") << serverName << " is ignored";
      continue;  // skip ignored server
    }
    if (server.isFallback) {
      edm::LogInfo("TritonService") << serverName << " is skipped because it is a fallback server";
      continue;  // fallback is a last resort, not a candidate for getBestServer
    }
    if (server.models.find(modelName) == server.models.end()) {
      edm::LogInfo("TritonService") << serverName << " is skipped because it does not have " << modelName;
      continue;  // server doesn't have model
    }

    tbb::concurrent_hash_map<std::string, ServerHealth>::const_accessor acc;
    if (!serversHealth_.find(acc, serverName)) {
      edm::LogInfo("TritonService") << serverName << " is skipped because it does not have health info";
      continue;  // no health info
    }

    const ServerHealth& health = acc->second;

    if (!health.live || !health.ready) {
      edm::LogInfo("TritonService") << serverName << " is skipped because is not live or ready";
      continue;  // skip unhealthy
    }

    // Select server according to rules:
    // 1) lowest failureCount
    // 2) tie-breaker: lowest avgQueueTimeMs
    if (!bestServerName || health.failureCount < bestHealth.failureCount ||
        (health.failureCount == bestHealth.failureCount && health.avgQueueTimeMs < bestHealth.avgQueueTimeMs)) {
      bestServerName = serverName;
      bestHealth = health;
    }
  }
  if (verbose_ && bestServerName) {
    edm::LogInfo("TritonDiscovery") << "Chosen server for model '" << modelName << "': " << *bestServerName
                                    << " (failures=" << bestHealth.failureCount
                                    << ", avgQueueTime=" << bestHealth.avgQueueTimeMs << " ms)";
  }
  return bestServerName;
}

void TritonService::startFallbackServer() {
  // Idempotent: do nothing if already running or disabled
  if (!fallbackOpts_.enable || startedFallback_)
    return;

  //include fallback server in set
  auto serverType = TritonServerType::LocalCPU;
  if (fallbackOpts_.device == "gpu")
    serverType = TritonServerType::LocalGPU;
  servers_.emplace(std::piecewise_construct,
                   std::forward_as_tuple(Server::fallbackName),
                   std::forward_as_tuple(Server::fallbackName, Server::fallbackAddress, serverType));

  std::string msg;
  if (verbose_)
    msg = "List of models for fallback server: ";
  // Provide all declared models with known paths via the fallback server
  auto& server(servers_.find(Server::fallbackName)->second);
  for (const auto& [modelName, model] : models_) {
    // Only seed models for which we have a repository path
    if (model.path.empty())
      continue;
    auto& modelInfo(models_.find(modelName)->second);
    modelInfo.servers.insert(Server::fallbackName);
    server.models.insert(modelName);
    if (verbose_)
      msg += modelName + ", ";
  }
  if (verbose_)
    edm::LogInfo("TritonDiscovery") << msg;

  //assemble server start command
  fallbackOpts_.command = "cmsTriton -P -1 -p " + pid_;
  fallbackOpts_.command += " -g " + fallbackOpts_.device;
  fallbackOpts_.command += " -d " + fallbackOpts_.container;
  if (fallbackOpts_.debug)
    fallbackOpts_.command += " -c";
  if (fallbackOpts_.verbose)
    fallbackOpts_.command += " -v";
  if (!fallbackOpts_.instanceName.empty())
    fallbackOpts_.command += " -n " + fallbackOpts_.instanceName;
  if (fallbackOpts_.retries >= 0)
    fallbackOpts_.command += " -r " + std::to_string(fallbackOpts_.retries);
  if (fallbackOpts_.wait >= 0)
    fallbackOpts_.command += " -w " + std::to_string(fallbackOpts_.wait);
  for (const auto& [modelName, model] : models_) {
    if (model.path.empty())
      continue;
    fallbackOpts_.command += " -m " + model.path;
  }
  std::string thread_string = " -I " + std::to_string(numberOfThreads_);
  fallbackOpts_.command += thread_string;
  if (!fallbackOpts_.imageName.empty())
    fallbackOpts_.command += " -i " + fallbackOpts_.imageName;
  if (!fallbackOpts_.sandboxDir.empty())
    fallbackOpts_.command += " -s " + fallbackOpts_.sandboxDir;
  // models_ remains for runtime queries; nothing to clear here

  //get a random temporary directory if none specified
  if (fallbackOpts_.tempDir.empty()) {
    auto tmp_dir_path{std::filesystem::temp_directory_path() /= edm::createGlobalIdentifier()};
    fallbackOpts_.tempDir = tmp_dir_path.string();
  }
  //special case ".": use script default (temp dir = .$instanceName)
  if (fallbackOpts_.tempDir != ".")
    fallbackOpts_.command += " -t " + fallbackOpts_.tempDir;

  std::string command = fallbackOpts_.command + " start";

  if (fallbackOpts_.debug)
    edm::LogInfo("TritonService") << "Fallback server temporary directory: " << fallbackOpts_.tempDir;
  if (verbose_)
    edm::LogInfo("TritonService") << command;

  //mark as started before executing in case of ctrl+c while command is running
  startedFallback_ = true;
  const auto& [output, rv] = execSys(command);
  if (rv != 0) {
    edm::LogError("TritonService") << output;
    printFallbackServerLog<edm::LogError>();
    throw edm::Exception(edm::errors::ExternalFailure)
        << "TritonService: Starting the fallback server failed with exit code " << rv;
  } else if (verbose_)
    edm::LogInfo("TritonService") << output;

  //get the chosen device
  std::string chosenDevice(fallbackOpts_.device);
  if (chosenDevice == "auto") {
    chosenDevice = extractFromLog(output, "CMS_TRITON_CHOSEN_DEVICE: ");
    if (!chosenDevice.empty()) {
      if (chosenDevice == "cpu")
        server.type = TritonServerType::LocalCPU;
      else if (chosenDevice == "gpu")
        server.type = TritonServerType::LocalGPU;
      else
        throw edm::Exception(edm::errors::ExternalFailure)
            << "TritonService: unsupported device choice " << chosenDevice << " for fallback server, log follows:\n"
            << output;
    } else
      throw edm::Exception(edm::errors::ExternalFailure)
          << "TritonService: unknown device choice for fallback server, log follows:\n"
          << output;
  }
  //print server info
  std::transform(chosenDevice.begin(), chosenDevice.end(), chosenDevice.begin(), toupper);
  if (verbose_)
    edm::LogInfo("TritonDiscovery") << "Fallback server started: " << chosenDevice;

  //get the port
  const auto& portNum = extractFromLog(output, "CMS_TRITON_GRPC_PORT: ");
  if (!portNum.empty())
    server.url += ":" + portNum;
  else
    throw edm::Exception(edm::errors::ExternalFailure)
        << "TritonService: Unknown port for fallback server, log follows:\n"
        << output;
}

void TritonService::preBeginJob(edm::ProcessContext const&) {
  // Capture unassigned models *before* startFallbackServer() is called.
  // startFallbackServer() seeds all known-path models into the fallback server
  // set, which would make unassignedModels() return empty afterward.
  const auto& unassigned = unassignedModels();

  // Always start the fallback server so it is ready for on-demand model
  // loading during retries, even when every model has a primary server.
  startFallbackServer();

  if (!unassigned.empty() && startedFallback_) {
    auto& server(servers_.find(Server::fallbackName)->second);
    for (const auto& modelName : unassigned) {
      server.models.insert(modelName);
      loadModel(modelName);
    }
  }
}

void TritonService::notifyCallStatus(bool status) const {
  if (status)
    --callFails_;
  else
    ++callFails_;
}

void TritonService::postEndJob() {
  if (!startedFallback_)
    return;

  std::string command = fallbackOpts_.command;
  //prevent log cleanup during server stop
  if (callFails_ > 0)
    command += " -c";
  command += " stop";
  if (verbose_)
    edm::LogInfo("TritonService") << command;

  const auto& [output, rv] = execSys(command);
  if (rv != 0 or callFails_ > 0) {
    //print logs if cmsRun is currently exiting because of a TritonException
    edm::LogError("TritonService") << output;
    printFallbackServerLog<edm::LogError>();
    if (rv != 0) {
      std::string stopCat("FallbackFailed");
      std::string stopMsg = std::format("TritonService: Stopping the fallback server failed with exit code {}", rv);
      //avoid throwing if the stack is already unwinding
      if (callFails_ > 0)
        edm::LogWarning(stopCat) << stopMsg;
      else
        throw cms::Exception(stopCat) << stopMsg;
    }
  } else if (verbose_) {
    edm::LogInfo("TritonService") << output;
    printFallbackServerLog<edm::LogInfo>();
  }
}

template <typename LOG>
void TritonService::printFallbackServerLog() const {
  std::vector<std::string> logNames{"log_" + fallbackOpts_.instanceName + ".log"};
  //cmsTriton script moves log from temp to current dir in verbose mode or in some cases when auto_stop is called
  // -> check both places
  logNames.push_back(fallbackOpts_.tempDir + "/" + logNames[0]);
  bool foundLog = false;
  for (const auto& logName : logNames) {
    std::ifstream infile(logName);
    if (infile.is_open()) {
      LOG("TritonService") << "TritonService: server log " << logName << "\n" << infile.rdbuf();
      foundLog = true;
      break;
    }
  }
  if (!foundLog)
    LOG("TritonService") << "TritonService: could not find server log " << logNames[0] << " in current directory or "
                         << fallbackOpts_.tempDir;
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
  fallbackDesc.ifValue(edm::ParameterDescription<std::string>("container", "apptainer", false),
                       edm::allowedValues<std::string>("apptainer", "docker", "podman"));
  fallbackDesc.ifValue(edm::ParameterDescription<std::string>("device", "auto", false),
                       edm::allowedValues<std::string>("auto", "cpu", "gpu"));
  fallbackDesc.addUntracked<int>("retries", -1);
  fallbackDesc.addUntracked<int>("wait", -1);
  fallbackDesc.addUntracked<std::string>("instanceBaseName", "triton_server_instance");
  fallbackDesc.addUntracked<std::string>("instanceName", "");
  fallbackDesc.addUntracked<std::string>("tempDir", "");
  fallbackDesc.addUntracked<std::string>("imageName", "");
  fallbackDesc.addUntracked<std::string>("sandboxDir", "");
  desc.add<edm::ParameterSetDescription>("fallback", fallbackDesc);

  descriptions.addWithDefaultLabel(desc);
}

bool TritonService::loadModel(const std::string& modelName) {
  std::lock_guard<std::mutex> lock(modelLoadMutex_);

  // Get model from models_ map (should exist from addModel during module construction)
  auto mit = models_.find(modelName);
  if (mit == models_.end()) {
    edm::LogWarning("TritonService") << "loadModel: Model " << modelName << " not found in models_ map";
    return false;
  }

  return loadModel(modelName, mit->second);
}

bool TritonService::loadModel(const std::string& modelName, Model& model) {
  // if already loaded, bump refcount
  if (model.refCount > 0) {
    ++model.refCount;
    if (verbose_)
      edm::LogInfo("TritonService") << "Model " << modelName << " already loaded, ref count: " << model.refCount;
    return true;
  }

  if (!startedFallback_) {
    throw cms::Exception("TritonService")
        << "loadModel: fallback server not started; cannot load model '" << modelName << "'";
  }

  auto sit = servers_.find(Server::fallbackName);
  if (sit == servers_.end()) {
    throw cms::Exception("TritonService") << "loadModel: fallback server not found";
  }

  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  TRITON_THROW_IF_ERROR(tc::InferenceServerGrpcClient::Create(
                            &client, sit->second.url, false, sit->second.useSsl, sit->second.sslOptions),
                        "loadModel: unable to create client for fallback server");

  TRITON_THROW_IF_ERROR(client->LoadModel(modelName),
                        "loadModel: failed to load model " + modelName + " on fallback server");

  // Update state and tracking
  model.refCount = 1;
  model.servers.insert(Server::fallbackName);
  sit->second.models.insert(modelName);
  fallbackLoadedModels_.insert(modelName);

  if (verbose_)
    edm::LogInfo("TritonService") << "Successfully loaded model " << modelName << " on fallback server";
  return true;
}

bool TritonService::unloadModel(const std::string& modelName) {
  std::lock_guard<std::mutex> lock(modelLoadMutex_);

  // Get model from models_ map
  auto mit = models_.find(modelName);
  if (mit == models_.end()) {
    edm::LogWarning("TritonService") << "unloadModel: Model " << modelName << " not found in models_ map";
    return false;
  }

  return unloadModel(modelName, mit->second);
}

bool TritonService::unloadModel(const std::string& modelName, Model& model) {
  if (model.refCount == 0) {
    edm::LogWarning("TritonService") << "unloadModel: Model " << modelName << " is not loaded";
    return false;
  }

  if (model.refCount > 1) {
    --model.refCount;
    if (verbose_)
      edm::LogInfo("TritonService") << "Model " << modelName << " still in use, ref count: " << model.refCount;
    return true;
  }

  auto sit = servers_.find(Server::fallbackName);
  if (sit == servers_.end()) {
    edm::LogWarning("TritonService") << "unloadModel: Fallback server not found";
    return false;
  }

  if (verbose_)
    edm::LogInfo("TritonService") << "Model " << modelName << " ref count is 1, unloading from fallback server";

  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  TRITON_THROW_IF_ERROR(tc::InferenceServerGrpcClient::Create(
                            &client, sit->second.url, false, sit->second.useSsl, sit->second.sslOptions),
                        "unloadModel: unable to create client for fallback server");

  TRITON_THROW_IF_ERROR(client->UnloadModel(modelName),
                        "unloadModel: failed to unload model " + modelName + " from fallback server");

  model.refCount = 0;
  model.servers.erase(Server::fallbackName);
  sit->second.models.erase(modelName);
  fallbackLoadedModels_.erase(modelName);

  if (verbose_)
    edm::LogInfo("TritonService") << "Successfully unloaded model " << modelName << " from fallback server";
  return true;
}
