#ifndef HeterogeneousCore_SonicTriton_TritonService
#define HeterogeneousCore_SonicTriton_TritonService

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <functional>
#include <utility>

//forward declarations
namespace edm {
  class ActivityRegistry;
  class ConfigurationDescriptions;
  class PathsAndConsumesOfModulesBase;
  class ProcessContext;
  class ModuleDescription;
}  // namespace edm

enum class TritonServerType { Remote = 0, LocalCPU = 1, LocalGPU = 2 };

class TritonService {
public:
  //classes and defs
  struct FallbackOpts {
    FallbackOpts(const edm::ParameterSet& pset)
        : enable(pset.getUntrackedParameter<bool>("enable")),
          debug(pset.getUntrackedParameter<bool>("debug")),
          verbose(pset.getUntrackedParameter<bool>("verbose")),
          useDocker(pset.getUntrackedParameter<bool>("useDocker")),
          useGPU(pset.getUntrackedParameter<bool>("useGPU")),
          retries(pset.getUntrackedParameter<int>("retries")),
          wait(pset.getUntrackedParameter<int>("wait")),
          instanceName(pset.getUntrackedParameter<std::string>("instanceName")),
          tempDir(pset.getUntrackedParameter<std::string>("tempDir")) {}

    bool enable;
    bool debug;
    bool verbose;
    bool useDocker;
    bool useGPU;
    int retries;
    int wait;
    std::string instanceName;
    std::string tempDir;
  };
  struct Server {
    Server(const edm::ParameterSet& pset)
        : url(pset.getUntrackedParameter<std::string>("address") + ":" +
              std::to_string(pset.getUntrackedParameter<unsigned>("port"))),
          isFallback(pset.getUntrackedParameter<std::string>("name") == fallbackName) {}
    Server(const std::string& name_, const std::string& url_) : url(url_), isFallback(name_ == fallbackName) {}

    //members
    std::string url;
    bool isFallback;
    std::unordered_set<std::string> models;
    static const std::string fallbackName;
    static const std::string fallbackAddress;
  };
  struct Model {
    Model(const std::string& path_ = "") : path(path_) {}

    //members
    std::string path;
    std::unordered_set<std::string> servers;
    std::unordered_set<unsigned> modules;
  };
  struct Module {
    //currently assumes that a module can only have one associated model
    Module(const std::string& model_) : model(model_) {}

    //members
    std::string model;
  };

  TritonService(const edm::ParameterSet& pset, edm::ActivityRegistry& areg);
  ~TritonService() = default;

  //accessors
  void addModel(const std::string& modelName, const std::string& path);
  std::pair<std::string, TritonServerType> serverAddress(const std::string& model,
                                                         const std::string& preferred = "") const;
  const std::string& pid() const { return pid_; }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void preModuleConstruction(edm::ModuleDescription const&);
  void postModuleConstruction(edm::ModuleDescription const&);
  void preModuleDestruction(edm::ModuleDescription const&);
  void preBeginJob(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const&);

  bool verbose_;
  FallbackOpts fallbackOpts_;
  unsigned currentModuleId_;
  bool allowAddModel_;
  bool startedFallback_;
  std::string pid_;
  std::unordered_map<std::string, Model> unservedModels_;
  //this represents a many:many:many map
  std::unordered_map<std::string, Server> servers_;
  std::unordered_map<std::string, Model> models_;
  std::unordered_map<unsigned, Module> modules_;
};

#endif
