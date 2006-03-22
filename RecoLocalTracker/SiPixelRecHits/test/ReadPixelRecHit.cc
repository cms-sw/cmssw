// File: ReadPixelRecHit.cc
// Description:  see ReadPixelRecHit.h
// Author:  O. Gutsche
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
  conf_(conf)
{
}

// Virtual destructor needed.
ReadPixelRecHit::~ReadPixelRecHit() { }  

// Functions that gets called by framework every event
void ReadPixelRecHit::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  using namespace edm;
  std::string rechitProducer = conf_.getParameter<std::string>("RecHitProducer");
  
  // Step A: Get Inputs 
  edm::Handle<SiPixelRecHitCollection> coll;
  e.getByType(coll);
  
  std::cout <<" FOUND "<<(coll.product())->size()<<" Pixel Hits"<<std::endl;
  
  
}

