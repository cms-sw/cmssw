#ifndef RecoEcal_EgammaClusterProducers_RecHitFilter_h_
#define RecoEcal_EgammaClusterProducers_RecHitFilter_h_
/** \class RecHitFilter
 **   simple filter of EcalRecHits
 **
 **  $Id: RecHitFilter.h,v 1.1 2006/05/04 18:05:44 rahatlou Exp $
 **  $Date: 2006/05/04 18:05:44 $
 **  $Revision: 1.1 $
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

      double      noiseThreshold_;
      std::string reducedHitCollection_;
      std::string hitProducer_;
      std::string hitCollection_;

};
#endif
