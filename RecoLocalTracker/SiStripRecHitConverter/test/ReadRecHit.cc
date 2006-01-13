// File: ReadRecHit.cc
// Description:  see ReadRecHit.h
// Author:  O. Gutsche
// Creation Date:  OGU Aug. 1 2005 Initial version.
//
//--------------------------------------------
#include <memory>
#include <string>
#include <iostream>

#include "RecoLocalTracker/SiStripRecHitConverter/test/ReadRecHit.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"


namespace cms
{

  ReadRecHit::ReadRecHit(edm::ParameterSet const& conf) : 
    readRecHitAlgorithm_(conf) ,
    conf_(conf)
  {
    //    produces<SiStripRecHit2DLocalPosCollection>();
  }


  // Virtual destructor needed.
  ReadRecHit::~ReadRecHit() { }  

  // Functions that gets called by framework every event
  void ReadRecHit::analyze(const edm::Event& e, const edm::EventSetup& es)
  {
    using namespace edm;
    std::string rechitProducer = conf_.getParameter<std::string>("RecHitProducer");

    // Step A: Get Inputs 
    edm::Handle<SiStripRecHit2DLocalPosCollection> rechitsmatched;
    edm::Handle<SiStripRecHit2DLocalPosCollection> rechitsrphi;
    edm::Handle<SiStripRecHit2DLocalPosCollection> rechitsstereo;
    e.getByLabel(rechitProducer,"matchedRecHit", rechitsmatched);
    e.getByLabel(rechitProducer,"rphiRecHit", rechitsrphi);
    e.getByLabel(rechitProducer,"stereoRecHit", rechitsstereo);

    std::cout<<"Matched hits:"<<std::endl;
    readRecHitAlgorithm_.run(rechitsmatched.product());
    std::cout<<"Rphi hits:"<<std::endl;
    readRecHitAlgorithm_.run(rechitsrphi.product());
  std::cout<<"Stereo hits:"<<std::endl;
    readRecHitAlgorithm_.run(rechitsstereo.product());
  }

}
