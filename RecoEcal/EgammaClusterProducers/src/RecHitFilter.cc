/** \class RecHitFilter
 **   simple filter of EcalRecHits
 **
 **  $Id: RecHitFilter.cc,v 1.5 2013/04/19 07:24:05 argiro Exp $
 **  $Date: 2013/04/19 07:24:05 $
 **  $Revision: 1.5 $
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/RecHitFilter.h"


RecHitFilter::RecHitFilter(const edm::ParameterSet& ps)
{

  noiseEnergyThreshold_       = ps.getParameter<double>("noiseEnergyThreshold");
  noiseChi2Threshold_       = ps.getParameter<double>("noiseChi2Threshold");
  hitCollection_        = ps.getParameter<edm::InputTag>("hitCollection");
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
  evt.getByLabel(hitCollection_, rhcHandle);
  if (!(rhcHandle.isValid())) 
    {
      edm::LogError("ProductNotFound")<<" Could not retrieve collection "<< hitCollection_ << std::endl; 

      return;
    }
  const EcalRecHitCollection* hit_collection = rhcHandle.product();

  int nTot = hit_collection->size();
  int nRed = 0;

  // create an auto_ptr to a BasicClusterCollection, copy the clusters into it and put in the Event:
  std::auto_ptr< EcalRecHitCollection > redCollection(new EcalRecHitCollection);

  for(EcalRecHitCollection::const_iterator it = hit_collection->begin(); it != hit_collection->end(); ++it) {
    //std::cout << *it << std::endl;
    if(it->energy() > noiseEnergyThreshold_ && it->chi2() < noiseChi2Threshold_) { 
        nRed++;
        redCollection->push_back( EcalRecHit(*it) );
    }
   
  }

  edm::LogInfo("")<< "total # hits: " << nTot << "  #hits with E > " << noiseEnergyThreshold_ << " GeV  and  chi2 < " <<  noiseChi2Threshold_ << " : " << nRed << std::endl;

  evt.put(redCollection, reducedHitCollection_);

}
