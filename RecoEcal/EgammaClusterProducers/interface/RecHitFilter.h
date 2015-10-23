#ifndef RecoEcal_EgammaClusterProducers_RecHitFilter_h_
#define RecoEcal_EgammaClusterProducers_RecHitFilter_h_
/** \class RecHitFilter
 **   simple filter of EcalRecHits
 **
 **  \author Shahram Rahatlou, University of Rome & INFN, May 2006
 **
 ***/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


class RecHitFilter : public edm::global::EDProducer<> {

  public:

      RecHitFilter(const edm::ParameterSet& ps);

      ~RecHitFilter();

      virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const;

   private:

      const double        noiseEnergyThreshold_;
      const double        noiseChi2Threshold_;
      const std::string   reducedHitCollection_;
      const edm::EDGetTokenT<EcalRecHitCollection> hitCollection_;


};
#endif
