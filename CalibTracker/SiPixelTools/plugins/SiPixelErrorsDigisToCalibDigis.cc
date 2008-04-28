// -*- C++ -*-
//
// Package:    SiPixelErrorsDigisToCalibDigis
// Class:      SiPixelErrorsDigisToCalibDigis
// 
/**\class SiPixelErrorsDigisToCalibDigis SiPixelErrorsDigisToCalibDigis.cc SiPixelErrors/SiPixelErrorsDigisToCalibDigis/src/SiPixelErrorsDigisToCalibDigis.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ricardo Vasquez Sierra
//         Created:  Wed Apr  9 12:43:02 CEST 2008
// $Id: SiPixelErrorsDigisToCalibDigis.cc,v 1.1 2008/04/21 19:00:15 vasquez Exp $
//
//


// system include files
#include <memory>

#include "CalibTracker/SiPixelTools/interface/SiPixelErrorsDigisToCalibDigis.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelErrorsDigisToCalibDigis::SiPixelErrorsDigisToCalibDigis(const edm::ParameterSet& iConfig)

{

  siPixelProducerLabel_ = iConfig.getParameter<edm::InputTag>("SiPixelProducerLabelTag");
  createOutputFile_ = iConfig.getUntrackedParameter<bool>("saveFile",false);
  outputFilename_ = iConfig.getParameter<std::string>("outputFilename");
  //  daqBE_ = &*edm::Service<DQMStore>();

  std::cout<<"siPixelProducerLabel_ = "<<siPixelProducerLabel_<<std::endl;
  std::cout<<"createOutputFile_= "<< createOutputFile_<<std::endl;
  std::cout<<"outpuFilename_= "<< outputFilename_<< std::endl;
}


SiPixelErrorsDigisToCalibDigis::~SiPixelErrorsDigisToCalibDigis()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
SiPixelErrorsDigisToCalibDigis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;


   Handle<DetSetVector<SiPixelCalibDigiError> > thePlaquettes;
   //   iEvent.getByLabel(siPixelProducerLabel_, thePlaquettes);
   iEvent.getByLabel("siPixelCalibDigis", thePlaquettes);


   DetSetVector<SiPixelCalibDigiError>::const_iterator digiIter;



   for (digiIter=thePlaquettes->begin(); digiIter!=thePlaquettes->end(); digiIter++)
     {
       uint32_t detId = digiIter->id;
       
       DetSet<SiPixelCalibDigiError>::const_iterator ipix;
       //loop over pixel errors pulsed in the current plaquette
       for(ipix=digiIter->data.begin(); ipix!=digiIter->end(); ++ipix)
	 {
	   std::cout << ipix->getRow() << " " << ipix->getCol() << std::endl;
	   
	 }
       
     }	 


//    Handle<DetSetVector<SiPixelCalibDigiError> > thePlaquettes;
//    //   iEvent.getByLabel(siPixelProducerLabel_, thePlaquettes);
//    iEvent.getByLabel("siPixelCalibDigis", thePlaquettes);


//    DetSetVector<SiPixelCalibDigiError>::const_iterator digiIter;



//    for (digiIter=thePlaquettes->begin(); digiIter!=thePlaquettes->end(); digiIter++)
//      {
//        uint32_t detId = digiIter->id;
       
//        DetSet<SiPixelCalibDigiError>::const_iterator ipix;
//        //loop over pixel errors pulsed in the current plaquette
//        for(ipix=digiIter->data.begin(); ipix!=digiIter->end(); ++ipix)
// 	 {
// 	   std::cout << ipix->getRow() << " " << ipix->getCol() << std::endl;
	   
// 	 }
       
//      }	 


}


// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelErrorsDigisToCalibDigis::beginJob(const edm::EventSetup&)
{



}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelErrorsDigisToCalibDigis::endJob() {
}

