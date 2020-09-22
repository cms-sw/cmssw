#ifndef HeterogeneousCore_SonicCore_SonicAcquirer
#define HeterogeneousCore_SonicCore_SonicAcquirer

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "HeterogeneousCore/SonicCore/interface/sonic_utils.h"

#include <chrono>

template <typename Client, typename Module>
class SonicAcquirer : public Module {
public:
  //typedef to simplify usage
  typedef typename Client::Input Input;
  //constructor
  SonicAcquirer(edm::ParameterSet const& cfg) : client_(cfg.getParameter<edm::ParameterSet>("Client")) {}
  //destructor
  ~SonicAcquirer() override = default;

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

protected:
  //for debugging
  void setDebugName(const std::string& debugName) { client_.setDebugName(debugName); }
  //members
  Client client_;
  std::chrono::time_point<std::chrono::high_resolution_clock> t_dispatch_;
};

#endif
