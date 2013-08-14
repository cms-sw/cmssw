// -*- C++ -*-
//
// Package:    SiPixelCalibDigiFilter
// Class:      SiPixelCalibDigiFilter
// 
/**\class SiPixelCalibDigiFilter SiPixelCalibDigiFilter.cc CalibTracker/SiPixelTools/src/SiPixelCalibDigiFilter.cc

 Description: Filters events that contain no information after the digis are collected into patterns by SiPixelCalibDigiProducer

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Evan Klose Friis
//         Created:  Tue Nov  6 16:59:50 CET 2007
// $Id: SiPixelCalibDigiFilter.cc,v 1.4 2010/08/10 09:06:13 ursl Exp $
//
//

#include "SiPixelCalibDigiFilter.h"

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigi.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelCalibDigiFilter::SiPixelCalibDigiFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


SiPixelCalibDigiFilter::~SiPixelCalibDigiFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
SiPixelCalibDigiFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   Handle<DetSetVector<SiPixelCalibDigi> > listOfDetIds;
   iEvent.getByLabel("SiPixelCalibDigiProducer", listOfDetIds);

   if (listOfDetIds->size() == 0)
      return false;
   else
      return true;
}

// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelCalibDigiFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelCalibDigiFilter::endJob() {
}

// -- define this as a plug-in
DEFINE_FWK_MODULE(SiPixelCalibDigiFilter);
