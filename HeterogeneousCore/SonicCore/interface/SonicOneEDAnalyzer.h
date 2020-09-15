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
  SonicOneEDAnalyzer(edm::ParameterSet const& cfg) : client_(cfg.getParameter<edm::ParameterSet>("Client")) {
    //ExternalWork is not compatible with one modules, so Sync mode is enforced
    if (client_.mode() != SonicMode::Sync)
      throw cms::Exception("UnsupportedMode") << "SonicOneEDAnalyzer can only use Sync mode for clients";
  }
  //destructor
  ~SonicOneEDAnalyzer() override = default;

  //derived classes still use a dedicated acquire() interface that incorporates client_.input() for consistency
  virtual void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) = 0;
  //derived classes use a dedicated analyze() interface that incorporates client_.output()
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) final {
    auto t0 = std::chrono::high_resolution_clock::now();
    acquire(iEvent, iSetup, client_.input());
    sonic_utils::printDebugTime(client_.debugName(), "acquire() time: ", t0);

    //pattern similar to ExternalWork, but blocking
    auto t1 = std::chrono::high_resolution_clock::now();
    client_.dispatch();

    //measure time between acquire and produce
    sonic_utils::printDebugTime(client_.debugName(), "dispatch() time: ", t1);

    auto t2 = std::chrono::high_resolution_clock::now();
    analyze(iEvent, iSetup, client_.output());
    sonic_utils::printDebugTime(client_.debugName(), "analyze() time: ", t2);

    //reset client data
    client_.reset();
  }
  virtual void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) = 0;

protected:
  //for debugging
  void setDebugName(const std::string& debugName) { client_.setDebugName(debugName); }
  //members
  Client client_;
};

#endif
