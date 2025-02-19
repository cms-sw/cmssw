// -*- C++ -*-
//
// Package:    SimpleTestPrintOutPixelCalibAnalyzer
// Class:      SimpleTestPrintOutPixelCalibAnalyzer
// 
/**\class SimpleTestPrintOutPixelCalibAnalyzer CalibTracker/SiPixelGainCalibration/test/SimpleTestPrintOutPixelCalibAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Mon Nov  5 16:56:35 CET 2007
// $Id: SimpleTestPrintOutPixelCalibAnalyzer.cc,v 1.7 2009/12/17 21:30:29 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigi.h"

#include "CalibTracker/SiPixelGainCalibration/test/SimpleTestPrintOutPixelCalibAnalyzer.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SimpleTestPrintOutPixelCalibAnalyzer::SimpleTestPrintOutPixelCalibAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


SimpleTestPrintOutPixelCalibAnalyzer::~SimpleTestPrintOutPixelCalibAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
void
SimpleTestPrintOutPixelCalibAnalyzer::printInfo(const edm::Event& iEvent, const edm::EventSetup& iSetup){

  using namespace edm;
  
  Handle<DetSetVector<SiPixelCalibDigi> > pIn;
  iEvent.getByLabel("siPixelCalibDigis",pIn);

  DetSetVector<SiPixelCalibDigi>::const_iterator digiIter;
  for(digiIter=pIn->begin(); digiIter!=pIn->end(); ++digiIter){
    uint32_t detid = digiIter->id;
    DetSet<SiPixelCalibDigi>::const_iterator ipix;
    for(ipix= digiIter->data.begin(); ipix!=digiIter->end(); ++ipix){
      std::cout  << std::endl;
      for(uint32_t ipoint=0; ipoint<ipix->getnpoints(); ++ipoint)
	std::cout << "\t Det ID " << detid << " row:" << ipix->row() << " col:" << ipix->col() << " point " << ipoint << " has " << ipix->getnentries(ipoint) << " entries, adc: " << ipix->getsum(ipoint) << ", adcsq: " << ipix->getsumsquares(ipoint) << std::endl;
    }
  }

}
// ------------ method called to for each event  ------------
void
SimpleTestPrintOutPixelCalibAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   printInfo(iEvent,iSetup);

}


// ------------ method called once each job just before starting event loop  ------------
void 
SimpleTestPrintOutPixelCalibAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SimpleTestPrintOutPixelCalibAnalyzer::endJob() {
}

