#ifndef HeterogeneousCore_SonicCore_SonicEDProducer
#define HeterogeneousCore_SonicCore_SonicEDProducer

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <chrono>

//this is a stream producer because client operations are not multithread-safe in general
//it is designed such that the user never has to interact with the client or the acquire() callback directly
template <typename Client, typename... Capabilities>
class SonicEDProducer : public edm::stream::EDProducer<edm::ExternalWork, Capabilities...> {
public:
  //typedefs to simplify usage
  typedef typename Client::Input Input;
  typedef typename Client::Output Output;
  //constructor
  SonicEDProducer(edm::ParameterSet const& cfg) : client_(cfg.getParameter<edm::ParameterSet>("Client")) {}
  //destructor
  ~SonicEDProducer() override = default;

  //derived classes use a dedicated acquire() interface that incorporates client_.input()
  //(no need to interact with callback holder)
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, edm::WaitingTaskWithArenaHolder holder) final {
    auto t0 = std::chrono::high_resolution_clock::now();
    acquire(iEvent, iSetup, client_.input());
    auto t1 = std::chrono::high_resolution_clock::now();
    if (!client_.debugName().empty())
      edm::LogInfo(client_.debugName()) << "acquire() time: "
                                        << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    t_dispatch_ = std::chrono::high_resolution_clock::now();
    client_.dispatch(holder);
  }
  virtual void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) = 0;
  //derived classes use a dedicated produce() interface that incorporates client_.output()
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) final {
    //measure time between acquire and produce
    auto t_finish = std::chrono::high_resolution_clock::now();
    if (!client_.debugName().empty())
      edm::LogInfo(client_.debugName())
          << "dispatch() time: "
          << std::chrono::duration_cast<std::chrono::microseconds>(t_finish - t_dispatch_).count();
    auto t0 = std::chrono::high_resolution_clock::now();
    produce(iEvent, iSetup, client_.output());
    auto t1 = std::chrono::high_resolution_clock::now();
    if (!client_.debugName().empty())
      edm::LogInfo(client_.debugName()) << "produce() time: "
                                        << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    //reset client data
    client_.reset();
  }
  virtual void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) = 0;

protected:
  //for debugging
  void setDebugName(const std::string& debugName) { client_.setDebugName(debugName); }
  //members
  Client client_;
  std::chrono::time_point<std::chrono::high_resolution_clock> t_dispatch_;
};

#endif
