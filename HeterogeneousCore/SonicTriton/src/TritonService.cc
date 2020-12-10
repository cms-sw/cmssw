#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"

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

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

const std::string TritonService::Server::fallbackName{"fallback"};
const std::string TritonService::Server::fallbackUrl{"0.0.0.0:8001"};

TritonService::TritonService(const edm::ParameterSet& pset, edm::ActivityRegistry& areg) : verbose_(pset.getUntrackedParameter<bool>("verbose")), fallbackOpts_(pset.getParameterSet("fallback")), startedFallback_(false) {
	//fallback server will be launched (if needed) before beginJob
	areg.watchPreBeginJob(this, &TritonService::preBeginJob);

	//include fallback server in set if enabled
	if(fallbackOpts_.enable)
		servers_.emplace(Server::fallbackName,Server::fallbackUrl);

	//loop over input servers: check which models they have
	std::string msg;
	if(verbose_)
		msg = "List of models for each server:\n";
	for(const auto& serverPset : pset.getUntrackedParameterSetVector("servers")){
		Server tmp(serverPset);
		//ensure uniqueness
		auto sit = servers_.find(tmp);
		if (sit!=servers_.end())
			throw cms::Exception("DuplicateServer") << "Not allowed to specify more than one server with same name (" << tmp.name << ")";

		std::unique_ptr<nic::InferenceServerGrpcClient> client;
		triton_utils::throwIfError(nic::InferenceServerGrpcClient::Create(&client, tmp.url, false), "TritonService(): unable to create inference context for "+tmp.name+" ("+tmp.url+")");

		inference::RepositoryIndexResponse repoIndexResponse;
		triton_utils::throwIfError(client->ModelRepositoryIndex(&repoIndexResponse), "TritonService(): unable to get repository index for "+tmp.name+" ("+tmp.url+")");

		//servers keep track of models and vice versa
		if(verbose_)
			msg += tmp.name + ": ";
		for(const auto& modelIndex : repoIndexResponse.models()){
			const auto& modelName = modelIndex.name();
			auto mit = findModel(modelName);
			if(mit==models_.end())
				mit = models_.emplace(modelName).first;
			mit->servers.insert(tmp.name);
			tmp.models.insert(modelName);
			if(verbose_)
				msg += modelName + ", ";
		}
		if(verbose_)
			msg += "\n";
		servers_.insert(tmp);
	}
	if(verbose_)
		edm::LogInfo("TritonService") << msg;
}

void TritonService::addModel(const std::string& modelName, const std::string& path) {
	//if model is not in the list, then no specified server provides it
	auto mit = findModel(modelName);
	if(mit==models_.end())
		unservedModels_.emplace(modelName,path);
}

//second return value is only true if fallback CPU server is being used
std::pair<std::string,bool> TritonService::serverAddress(const std::string& model, const std::string& preferred) const {
	auto mit = findModel(model);
	if (mit==models_.end())
		throw cms::Exception("MissingModel") << "There are no servers that provide model " << model;

	const auto& modelServers = mit->servers;

	if (!preferred.empty()){
		auto sit = modelServers.find(preferred);
		//todo: add a "strict" parameter to stop execution if preferred server isn't found?
		if(sit==modelServers.end())
			edm::LogWarning("PreferredServer") << "Preferred server " << preferred << " for model " << model << " not available, will choose another server";
		else
			return std::make_pair(findServer(preferred)->url,false);
	}

	//todo: use some algorithm to select server rather than just picking arbitrarily
	auto sit = findServer(*modelServers.begin());
	bool isFallbackCPU = sit->isFallback and !fallbackOpts_.useGPU;
	return std::make_pair(sit->url,isFallbackCPU);
}

void TritonService::preBeginJob(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const&) {
	//only need fallback if there are unserved models
	if (!fallbackOpts_.enable or unservedModels_.empty()) return;

	std::string msg;
	if(verbose_)
		msg = "List of models for fallback server: ";
	//all unserved models are provided by fallback server
	auto sit = findServer(Server::fallbackName);
	for(const auto& model : unservedModels_){
		auto mit = models_.insert(model).first;
		mit->servers.insert(Server::fallbackName);
		sit->models.insert(mit->name);
		if(verbose_)
			msg += mit->name + ", ";
	}
	if(verbose_)
		edm::LogInfo("TritonService") << msg;

	//assemble server start command
	std::string command("triton");
	if (fallbackOpts_.verbose)
		command += " -v";
	if (fallbackOpts_.useDocker)
		command += " -d";
	if (fallbackOpts_.useGPU)
		command += " -g";
	if (!fallbackOpts_.instanceName.empty())
		command += " -n "+fallbackOpts_.instanceName;
	if (fallbackOpts_.retries >= 0)
		command += " -r "+std::to_string(fallbackOpts_.retries);
	if (fallbackOpts_.wait >=0)
		command += " -w "+std::to_string(fallbackOpts_.wait);
	for (const auto& model : unservedModels_) {
		command += " -m "+model.path;
	}
	//don't need this anymore
	unservedModels_.clear();

	//get a random temporary directory if none specified
	if (fallbackOpts_.tempDir.empty()) {
		auto tmp_dir_path{std::filesystem::temp_directory_path() /= std::tmpnam(nullptr)};
		fallbackOpts_.tempDir = tmp_dir_path.string();
	}
	//special case ".": use script default (temp dir = .$instanceName)
	if (fallbackOpts_.tempDir != ".")
		command += " -t "+fallbackOpts_.tempDir;

	command += " start";

	if(verbose_)
		edm::LogInfo("TritonService") << command;

	//mark as started before executing in case of ctrl+c while command is running
	startedFallback_ = true;
	int rv = std::system(command.c_str());
	if (rv != 0)
		throw cms::Exception("FallbackFailed") << "Starting the fallback server failed with exit code " << rv;
}

TritonService::~TritonService() {
	if (!startedFallback_) return;

	//assemble server stop command
	std::string command("triton");

	if (fallbackOpts_.verbose)
		command += " -v";
	if (fallbackOpts_.useDocker)
		command += " -d";
	if (!fallbackOpts_.instanceName.empty())
		command += " -n "+fallbackOpts_.instanceName;
	if (fallbackOpts_.tempDir != ".")
		command += " -t "+fallbackOpts_.tempDir;

	command += " stop";

	if(verbose_)
		edm::LogInfo("TritonService") << command;
	int rv = std::system(command.c_str());
	if (rv != 0)
		edm::LogError("FallbackFailed") << "Stopping the fallback server failed with exit code " << rv;
}

void TritonService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	edm::ParameterSetDescription desc;
	desc.addUntracked<bool>("verbose",false);

	edm::ParameterSetDescription validator;
	validator.addUntracked<std::string>("name");
	validator.addUntracked<std::string>("address");
	validator.addUntracked<unsigned>("port");

	desc.addVPSetUntracked("servers", validator, {});

	edm::ParameterSetDescription fallbackDesc;
	fallbackDesc.addUntracked<bool>("enable",false);
	fallbackDesc.addUntracked<bool>("verbose",false);
	fallbackDesc.addUntracked<bool>("useDocker",false);
	fallbackDesc.addUntracked<bool>("useGPU",false);
	fallbackDesc.addUntracked<int>("retries",-1);
	fallbackDesc.addUntracked<int>("wait",-1);
	fallbackDesc.addUntracked<std::string>("instanceName","");
	fallbackDesc.addUntracked<std::string>("tempDir","");
	desc.add<edm::ParameterSetDescription>("fallback",fallbackDesc);

	descriptions.addWithDefaultLabel(desc);
}
