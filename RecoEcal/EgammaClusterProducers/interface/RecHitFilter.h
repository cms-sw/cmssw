#ifndef RecoEcal_EgammaClusterProducers_RecHitFilter_h_
#define RecoEcal_EgammaClusterProducers_RecHitFilter_h_
/** \class RecHitFilter
 **   simple filter of EcalRecHits
 **
 **  $Id: RecHitFilter.h,v 1.3 2013/04/19 07:24:06 argiro Exp $
 **  $Date: 2013/04/19 07:24:06 $
 **  $Revision: 1.3 $
 **  \author Shahram Rahatlou, University of Rome & INFN, May 2006
 **
 ***/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


class RecHitFilter : public edm::EDProducer {

  public:

      RecHitFilter(const edm::ParameterSet& ps);

      ~RecHitFilter();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

      double        noiseEnergyThreshold_;
      double        noiseChi2Threshold_;
      std::string   reducedHitCollection_;
      edm::InputTag hitCollection_;


};
#endif
