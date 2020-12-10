#ifndef HeterogeneousCore_SonicTriton_TritonService
#define HeterogeneousCore_SonicTriton_TritonService

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "tbb/concurrent_unordered_set.h"

#include <vector>
#include <unordered_set>
#include <string>
#include <functional>
#include <utility>

//forward declarations
namespace edm {
	class ActivityRegistry;
	class ConfigurationDescriptions;
	class PathsAndConsumesOfModulesBase;
	class ProcessContext;
}

class TritonService {
public:
	//classes and defs
	struct FallbackOpts {
		FallbackOpts(const edm::ParameterSet& pset) : enable(pset.getUntrackedParameter<bool>("enable")), verbose(pset.getUntrackedParameter<bool>("verbose")), useDocker(pset.getUntrackedParameter<bool>("useDocker")), useGPU(pset.getUntrackedParameter<bool>("useGPU")), retries(pset.getUntrackedParameter<int>("retries")), wait(pset.getUntrackedParameter<int>("wait")), instanceName(pset.getUntrackedParameter<std::string>("instanceName")), tempDir(pset.getUntrackedParameter<std::string>("tempDir")) {}

		bool enable;
		bool verbose;
		bool useDocker;
		bool useGPU;
		int retries;
		int wait;
		std::string instanceName;
		std::string tempDir;
	};
	struct Server {
		Server(const edm::ParameterSet& pset) : name(pset.getUntrackedParameter<std::string>("name")), url(pset.getUntrackedParameter<std::string>("address") + ":" + std::to_string(pset.getUntrackedParameter<unsigned>("port"))), isFallback(name==fallbackName) {}
		Server(const std::string& name_, const std::string& url_="") : name(name_), url(url_), isFallback(name==fallbackName) {}

		struct Hash {
			size_t operator()(const Server& obj) const {
				return hashObj(obj.name);
			}
			std::hash<std::string> hashObj;
		};

		struct Equal {
			bool operator()(const Server& lhs, const Server& rhs) const {
				return lhs.name == rhs.name;
			}
		};

		//members
		std::string name;
		std::string url;
		bool isFallback;
		mutable std::unordered_set<std::string> models;
		static const std::string fallbackName;
		static const std::string fallbackUrl;
	};
	struct Model {
		Model(const std::string& name_, const std::string& path_="") : name(name_), path(path_) {}

		struct Hash {
			size_t operator()(const Model& obj) const {
				return hashObj(obj.name);
			}
			std::hash<std::string> hashObj;
		};

		struct Equal {
			bool operator()(const Model& lhs, const Model& rhs) const {
				return lhs.name == rhs.name;
			}
		};

		//members
		std::string name;
		std::string path;
		mutable std::unordered_set<std::string> servers;
	};

	TritonService(const edm::ParameterSet& pset, edm::ActivityRegistry& areg);
	~TritonService();

	//accessors
	void addModel(const std::string& modelName, const std::string& path);
	std::pair<std::string,bool> serverAddress(const std::string& model, const std::string& preferred="") const;

	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
	void preBeginJob(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const&);

	//to search without full object
	auto findServer(const std::string& name) const { return servers_.find(Server(name)); }
	auto findModel(const std::string& name) const { return models_.find(Model(name)); }

	bool verbose_;
	FallbackOpts fallbackOpts_;
	bool startedFallback_;
	//concurrent data type is used because addModel() might be called by multiple threads
	tbb::concurrent_unordered_set<Model,Model::Hash,Model::Equal> unservedModels_;
	//this is a lazy and inefficient many:many map
	std::unordered_set<Server,Server::Hash,Server::Equal> servers_;
	std::unordered_set<Model,Model::Hash,Model::Equal> models_;
};

#endif
