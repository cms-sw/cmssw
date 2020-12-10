#ifndef HeterogeneousCore_SonicTriton_TritonService
#define HeterogeneousCore_SonicTriton_TritonService

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <unordered_set>
#include <string>
#include <functional>

//forward declarations
namespace edm {
	class ActivityRegistry;
	class ConfigurationDescriptions;
}

class TritonService {
public:
	//classes and defs
	struct Server {
		Server(const edm::ParameterSet& pset) : name(pset.getUntrackedParameter<std::string>("name")), url(pset.getUntrackedParameter<std::string>("address") + ":" + std::to_string(pset.getUntrackedParameter<unsigned>("port"))) {}
		Server(const std::string& name_) : name(name_), url("") {}

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
		mutable std::unordered_set<std::string> models;
	};
	struct Model {
		Model(const std::string& name_) : name(name_) {}

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
		mutable std::unordered_set<std::string> servers;
	};

	TritonService(const edm::ParameterSet& pset, edm::ActivityRegistry& areg);

	//accessors
	std::string serverAddress(const std::string& model, const std::string& preferred="") const;

	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
	//to search without full object
	auto findServer(const std::string& name) const { return servers_.find(Server(name)); }
	auto findModel(const std::string& name) const { return models_.find(Model(name)); }

	//this is a lazy and inefficient many:many map
	std::unordered_set<Server,Server::Hash,Server::Equal> servers_;
	std::unordered_set<Model,Model::Hash,Model::Equal> models_;
};

#endif
