#ifndef HeterogeneousCore_SonicTriton_TritonService
#define HeterogeneousCore_SonicTriton_TritonService

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "tbb/concurrent_unordered_map.h"

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
}  // namespace edm

class TritonService {
public:
  //classes and defs
  struct FallbackOpts {
    FallbackOpts(const edm::ParameterSet& pset)
        : enable(pset.getUntrackedParameter<bool>("enable")),
          verbose(pset.getUntrackedParameter<bool>("verbose")),
          useDocker(pset.getUntrackedParameter<bool>("useDocker")),
          useGPU(pset.getUntrackedParameter<bool>("useGPU")),
          retries(pset.getUntrackedParameter<int>("retries")),
          wait(pset.getUntrackedParameter<int>("wait")),
          instanceName(pset.getUntrackedParameter<std::string>("instanceName")),
          tempDir(pset.getUntrackedParameter<std::string>("tempDir")) {}

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
    static const std::string fallbackUrl;
  };
  struct Model {
    Model(const std::string& path_ = "") : path(path_) {}

    //members
    std::string path;
    std::unordered_set<std::string> servers;
  };

  TritonService(const edm::ParameterSet& pset, edm::ActivityRegistry& areg);
  ~TritonService();

  //accessors
  void addModel(const std::string& modelName, const std::string& path);
  std::pair<std::string, bool> serverAddress(const std::string& model, const std::string& preferred = "") const;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void preBeginJob(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const&);

  bool verbose_;
  FallbackOpts fallbackOpts_;
  bool startedFallback_;
  //concurrent data type is used because addModel() might be called by multiple threads
  tbb::concurrent_unordered_map<std::string, Model> unservedModels_;
  //this is a lazy and inefficient many:many map
  std::unordered_map<std::string, Server> servers_;
  std::unordered_map<std::string, Model> models_;
};

#endif
