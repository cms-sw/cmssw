#ifndef HeterogeneousCore_SonicCore_SonicAcquirer
#define HeterogeneousCore_SonicCore_SonicAcquirer

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "HeterogeneousCore/SonicCore/interface/sonic_utils.h"

#include <chrono>
#include <memory>
#include <string>

template <typename Client, typename Module>
class SonicAcquirer : public Module {
public:
  //typedef to simplify usage
  typedef typename Client::Input Input;
  //constructor
  SonicAcquirer(edm::ParameterSet const& cfg)
      : clientPset_(cfg.getParameterSet("Client")),
        debugName_(cfg.getParameter<std::string>("@module_label")),
        verbose_(clientPset_.getUntrackedParameter<bool>("verbose")) {}
  //destructor
  ~SonicAcquirer() override = default;

  //construct client at beginning of job
  //in case client constructor depends on operations happening in derived module constructors
  void beginStream(edm::StreamID) override { makeClient(); }

  //derived classes use a dedicated acquire() interface that incorporates client_->input()
  //(no need to interact with callback holder)
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, edm::WaitingTaskWithArenaHolder holder) final {
    auto t0 = std::chrono::high_resolution_clock::now();
    acquire(iEvent, iSetup, client_->input());
    if (verbose_)
      sonic_utils::printDebugTime(debugName_, "acquire() time: ", t0);
    t_dispatch_ = std::chrono::high_resolution_clock::now();
    client_->dispatch(holder);
  }
  virtual void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) = 0;

protected:
  //helper
  void makeClient() { client_ = std::make_unique<Client>(clientPset_, debugName_); }

  //members
  edm::ParameterSet clientPset_;
  std::unique_ptr<Client> client_;
  std::string debugName_;
  bool verbose_;
  std::chrono::time_point<std::chrono::high_resolution_clock> t_dispatch_;
};

#endif
