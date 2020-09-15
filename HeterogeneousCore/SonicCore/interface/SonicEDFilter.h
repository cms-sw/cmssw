#ifndef HeterogeneousCore_SonicCore_SonicEDFilter
#define HeterogeneousCore_SonicCore_SonicEDFilter

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/SonicCore/interface/sonic_utils.h"

#include <string>
#include <chrono>

//this is a stream filter because client operations are not multithread-safe in general
//it is designed such that the user never has to interact with the client or the acquire() callback directly
template <typename Client, typename... Capabilities>
class SonicEDFilter : public edm::stream::EDFilter<edm::ExternalWork, Capabilities...> {
public:
  //typedefs to simplify usage
  typedef typename Client::Input Input;
  typedef typename Client::Output Output;
  //constructor
  SonicEDFilter(edm::ParameterSet const& cfg) : client_(cfg.getParameter<edm::ParameterSet>("Client")) {}
  //destructor
  ~SonicEDFilter() override = default;

  //derived classes use a dedicated acquire() interface that incorporates client_.input()
  //(no need to interact with callback holder)
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, edm::WaitingTaskWithArenaHolder holder) final {
    auto t0 = std::chrono::high_resolution_clock::now();
    acquire(iEvent, iSetup, client_.input());
    sonic_utils::printDebugTime(client_.debugName(), "acquire() time: ", t0);
    t_dispatch_ = std::chrono::high_resolution_clock::now();
    client_.dispatch(holder);
  }
  virtual void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) = 0;
  //derived classes use a dedicated produce() interface that incorporates client_.output()
  bool filter(edm::Event& iEvent, edm::EventSetup const& iSetup) final {
    //measure time between acquire and produce
    sonic_utils::printDebugTime(client_.debugName(), "dispatch() time: ", t_dispatch_);

    auto t0 = std::chrono::high_resolution_clock::now();
    bool result = produce(iEvent, iSetup, client_.output());
    sonic_utils::printDebugTime(client_.debugName(), "produce() time: ", t0);

    //reset client data
    client_.reset();

    return result;
  }
  virtual bool filter(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) = 0;

protected:
  //for debugging
  void setDebugName(const std::string& debugName) { client_.setDebugName(debugName); }
  //members
  Client client_;
  std::chrono::time_point<std::chrono::high_resolution_clock> t_dispatch_;
};

#endif
