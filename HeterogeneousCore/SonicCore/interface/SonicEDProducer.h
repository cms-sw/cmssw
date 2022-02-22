#ifndef HeterogeneousCore_SonicCore_SonicEDProducer
#define HeterogeneousCore_SonicCore_SonicEDProducer

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "HeterogeneousCore/SonicCore/interface/sonic_utils.h"
#include "HeterogeneousCore/SonicCore/interface/SonicAcquirer.h"

#include <string>
#include <chrono>

//this is a stream producer because client operations are not multithread-safe in general
//it is designed such that the user never has to interact with the client or the acquire() callback directly
template <typename Client, typename... Capabilities>
class SonicEDProducer : public SonicAcquirer<Client, edm::stream::EDProducer<edm::ExternalWork, Capabilities...>> {
public:
  //typedef to simplify usage
  typedef typename Client::Output Output;
  //constructor
  SonicEDProducer(edm::ParameterSet const& cfg)
      : SonicAcquirer<Client, edm::stream::EDProducer<edm::ExternalWork, Capabilities...>>(cfg) {}
  //destructor
  ~SonicEDProducer() override = default;

  //derived classes use a dedicated produce() interface that incorporates client_->output()
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) final {
    //measure time between acquire and produce
    if (this->verbose_)
      sonic_utils::printDebugTime(this->debugName_, "dispatch() time: ", this->t_dispatch_);

    auto t0 = std::chrono::high_resolution_clock::now();
    produce(iEvent, iSetup, this->client_->output());
    if (this->verbose_)
      sonic_utils::printDebugTime(this->debugName_, "produce() time: ", t0);

    //reset client data
    this->client_->reset();
  }
  virtual void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) = 0;
};

#endif
