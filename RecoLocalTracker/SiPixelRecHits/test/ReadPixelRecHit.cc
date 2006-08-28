//--------------------------------------------
// File: ReadPixelRecHit.cc
// Description:  see ReadPixelRecHit.h
// Author:  J.Sheav (JHU)
//          11/8/06: New loop over rechits and InputTag, V.Chiochia
// Creation Date:  OGU Aug. 1 2005 Initial version.
//
//--------------------------------------------
#include <memory>
#include <string>
#include <iostream>

#include "RecoLocalTracker/SiPixelRecHits/test/ReadPixelRecHit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"



ReadPixelRecHit::ReadPixelRecHit(edm::ParameterSet const& conf) : 
  conf_(conf),
  src_( conf.getParameter<edm::InputTag>( "src" ) )
{
}

// Virtual destructor needed.
ReadPixelRecHit::~ReadPixelRecHit() { }  

// Functions that gets called by framework every event
void ReadPixelRecHit::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  using namespace edm;

  edm::Handle<SiPixelRecHitCollection> recHitColl;
  e.getByLabel( src_ , recHitColl);
 
  std::cout <<" FOUND "<<(recHitColl.product())->size()<<" Pixel Hits"<<std::endl;

  SiPixelRecHitCollection::id_iterator recHitIdIterator;
  SiPixelRecHitCollection::id_iterator recHitIdIteratorBegin = (recHitColl.product())->id_begin();
  SiPixelRecHitCollection::id_iterator recHitIdIteratorEnd   = (recHitColl.product())->id_end();

  // Loop over Detector IDs
  for ( recHitIdIterator = recHitIdIteratorBegin; recHitIdIterator != recHitIdIteratorEnd; recHitIdIterator++) {

    SiPixelRecHitCollection::range pixelrechitRange = (recHitColl.product())->get(*recHitIdIterator);

    std::cout <<"     Det ID " << (*recHitIdIterator).rawId() << std::endl;
    
    SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.first;
    SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorEnd = pixelrechitRange.second;
    SiPixelRecHitCollection::const_iterator pixeliter = pixelrechitRangeIteratorBegin;
    
    //----Loop over rechits for this detId
    for ( ; pixeliter != pixelrechitRangeIteratorEnd; ++pixeliter) {
      std::cout <<"     Position " << pixeliter->localPosition() << std::endl;
	}
  }
  
}

