#ifndef HeterogeneousCore_SonicCore_SonicEDFilter
#define HeterogeneousCore_SonicCore_SonicEDFilter

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "HeterogeneousCore/SonicCore/interface/sonic_utils.h"
#include "HeterogeneousCore/SonicCore/interface/SonicAcquirer.h"

#include <string>
#include <chrono>

//this is a stream filter because client operations are not multithread-safe in general
//it is designed such that the user never has to interact with the client or the acquire() callback directly
template <typename Client, typename... Capabilities>
class SonicEDFilter : public SonicAcquirer<Client, edm::stream::EDFilter<edm::ExternalWork, Capabilities...>> {
public:
  //typedef to simplify usage
  typedef typename Client::Output Output;
  //constructor
  SonicEDFilter(edm::ParameterSet const& cfg, const std::string& debugName)
      : SonicAcquirer<Client, edm::stream::EDFilter<edm::ExternalWork, Capabilities...>>(cfg, debugName) {}
  //destructor
  ~SonicEDFilter() override = default;

  //derived classes use a dedicated produce() interface that incorporates client_->output()
  bool filter(edm::Event& iEvent, edm::EventSetup const& iSetup) final {
    //measure time between acquire and produce
    sonic_utils::printDebugTime(this->debugName_, "dispatch() time: ", this->t_dispatch_);

    auto t0 = std::chrono::high_resolution_clock::now();
    bool result = filter(iEvent, iSetup, this->client_->output());
    sonic_utils::printDebugTime(this->debugName_, "filter() time: ", t0);

    //reset client data
    this->client_->reset();

    return result;
  }
  virtual bool filter(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) = 0;
};

#endif
