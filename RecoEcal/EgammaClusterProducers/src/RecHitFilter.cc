/** \class RecHitFilter
 **   simple filter of EcalRecHits
 **
 **  $Id: RecHitFilter.cc,v 1.2 2007/03/08 19:11:10 futyand Exp $
 **  $Date: 2007/03/08 19:11:10 $
 **  $Revision: 1.2 $
 **  \author Shahram Rahatlou, University of Rome & INFN, May 2006
 **
 ***/
// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"


// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/RecHitFilter.h"


RecHitFilter::RecHitFilter(const edm::ParameterSet& ps)
{

  noiseEnergyThreshold_       = ps.getParameter<double>("noiseEnergyThreshold");
  //noiseEtThreshold_       = ps.getParameter<double>("noiseEtThreshold");
  noiseChi2Threshold_       = ps.getParameter<double>("noiseChi2Threshold");
  hitProducer_          = ps.getParameter<std::string>("hitProducer");
  hitCollection_        = ps.getParameter<std::string>("hitCollection");
  reducedHitCollection_ = ps.getParameter<std::string>("reducedHitCollection");

  produces< EcalRecHitCollection >(reducedHitCollection_);
}


RecHitFilter::~RecHitFilter()
{
}


void RecHitFilter::produce(edm::Event& evt, const edm::EventSetup& es)
{
  // get the hit collection from the event:
  edm::Handle<EcalRecHitCollection> rhcHandle;
  evt.getByLabel(hitProducer_, hitCollection_, rhcHandle);
  if (!(rhcHandle.isValid())) 
    {
      std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
      return;
    }
  const EcalRecHitCollection* hit_collection = rhcHandle.product();

  int nTot = hit_collection->size();
  int nRed = 0;

  // create an auto_ptr to a BasicClusterCollection, copy the clusters into it and put in the Event:
  std::auto_ptr< EcalRecHitCollection > redCollection(new EcalRecHitCollection);
  //clusters_p->assign(clusters.begin(), clusters.end());

  for(EcalRecHitCollection::const_iterator it = hit_collection->begin(); it != hit_collection->end(); ++it) {
    //std::cout << *it << std::endl;
    if(it->energy() > noiseEnergyThreshold_ && it->chi2() < noiseChi2Threshold_) { 
        nRed++;
        redCollection->push_back( EcalRecHit(*it) );
    }
   
  }
  std::cout << "total # hits: " << nTot << "  #hits with E > " << noiseEnergyThreshold_ << " GeV  and  chi2 < " <<  noiseChi2Threshold_ << " : " << nRed << std::endl;

  evt.put(redCollection, reducedHitCollection_);

  //std::cout << "BasicClusterCollection added to the Event! :-)" << std::endl;

}
