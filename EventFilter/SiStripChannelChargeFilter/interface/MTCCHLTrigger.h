#ifndef MTCCHLTrigger_H
#define MTCCHLTrigger_H 

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
// #include "SimDataFormats/TrackingHit/interface/PSimHit.h"
// #include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

namespace cms
{
 class MTCCHLTrigger : public edm::EDFilter {
  public:
    MTCCHLTrigger(const edm::ParameterSet& ps);
    virtual ~MTCCHLTrigger() {}

    virtual bool filter(edm::Event & e, edm::EventSetup const& c);

  private:
   //   bool selOnClusterCharge;
   bool selOnDigiCharge;
   unsigned int ChargeThreshold;
   //   unsigned int digiChargeThreshold;
   //   std::string rawtodigiProducer;
   //   std::string zsdigiProducer;
   std::string clusterProducer;
  };
}
#endif
