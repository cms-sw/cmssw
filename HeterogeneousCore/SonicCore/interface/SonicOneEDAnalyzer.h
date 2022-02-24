#ifndef HeterogeneousCore_SonicCore_SonicOneEDAnalyzer
#define HeterogeneousCore_SonicCore_SonicOneEDAnalyzer

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientBase.h"
#include "HeterogeneousCore/SonicCore/interface/sonic_utils.h"

#include <string>
#include <chrono>

//this is a one analyzer to enable writing trees/histograms for analysis users with results from inference as a service
template <typename Client, typename... Capabilities>
class SonicOneEDAnalyzer : public edm::one::EDAnalyzer<Capabilities...> {
public:
  //typedefs to simplify usage
  typedef typename Client::Input Input;
  typedef typename Client::Output Output;
  //constructor
  SonicOneEDAnalyzer(edm::ParameterSet const& cfg, bool verbose = true)
      : clientPset_(cfg.getParameterSet("Client")),
        debugName_(cfg.getParameter<std::string>("@module_label")),
        verbose_(clientPset_.getUntrackedParameter<bool>("verbose")) {
    //ExternalWork is not compatible with one modules, so Sync mode is enforced
    if (clientPset_.getParameter<std::string>("mode") != "Sync") {
      edm::LogWarning("ResetClientMode") << "Resetting client mode to Sync for SonicOneEDAnalyzer";
      clientPset_.addParameter<std::string>("mode", "Sync");
    }
  }
  //destructor
  ~SonicOneEDAnalyzer() override = default;

  //construct client at beginning of job
  //in case client constructor depends on operations happening in derived module constructors
  void beginJob() override { makeClient(); }

  //derived classes still use a dedicated acquire() interface that incorporates client_->input() for consistency
  virtual void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) = 0;
  //derived classes use a dedicated analyze() interface that incorporates client_->output()
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) final {
    auto t0 = std::chrono::high_resolution_clock::now();
    acquire(iEvent, iSetup, client_->input());
    if (verbose_)
      sonic_utils::printDebugTime(debugName_, "acquire() time: ", t0);

    //pattern similar to ExternalWork, but blocking
    auto t1 = std::chrono::high_resolution_clock::now();
    client_->dispatch();

    //measure time between acquire and produce
    if (verbose_)
      sonic_utils::printDebugTime(debugName_, "dispatch() time: ", t1);

    auto t2 = std::chrono::high_resolution_clock::now();
    analyze(iEvent, iSetup, client_->output());
    if (verbose_)
      sonic_utils::printDebugTime(debugName_, "analyze() time: ", t2);

    //reset client data
    client_->reset();
  }
  virtual void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) = 0;

protected:
  //helper
  void makeClient() { client_ = std::make_unique<Client>(clientPset_, debugName_); }

  //members
  edm::ParameterSet clientPset_;
  std::unique_ptr<Client> client_;
  std::string debugName_;
  bool verbose_;
};

#endif
