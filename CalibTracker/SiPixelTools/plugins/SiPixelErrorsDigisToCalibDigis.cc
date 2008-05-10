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
// $Id: SiPixelErrorsDigisToCalibDigis.cc,v 1.1 2008/04/28 21:53:01 vasquez Exp $
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
  daqBE_ = &*edm::Service<DQMStore>();

  folderMaker_ = new SiPixelFolderOrganizer();

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
  iEvent.getByLabel(siPixelProducerLabel_, thePlaquettes);
  // iEvent.getByLabel("siPixelCalibDigis", thePlaquettes);
  
  
  DetSetVector<SiPixelCalibDigiError>::const_iterator digiIter;
  
  
  for (digiIter=thePlaquettes->begin(); digiIter!=thePlaquettes->end(); digiIter++)
    {
      uint32_t detId = digiIter->id;
      
      DetSet<SiPixelCalibDigiError>::const_iterator ipix;
      //loop over pixel errors pulsed in the current plaquette

      MonitorElement* temp_;

      std::map<uint32_t, MonitorElement*>::iterator mapIterator =  SiPixelErrorsDigisToCalibDigis_2DErrorInformation_.find(detId);
      
      if (digiIter->begin() != digiIter->end()) {
	if ( mapIterator == SiPixelErrorsDigisToCalibDigis_2DErrorInformation_.end() )
	  {
	    std::cout << "This is the beginning of an error 2d histo booking: "<<std::endl;
	    temp_ = bookDQMHistoPlaquetteSummary2D(detId, "SiPixelErrorsCalibDigis", "SiPixelErrorsDigisToCalibDigis");
	    SiPixelErrorsDigisToCalibDigis_2DErrorInformation_.insert( std::make_pair(detId,temp_));
	  }
	else
	  {
	    std::cout << "This one was already booked."<<std::endl;
	    temp_ = (*mapIterator).second;
	  }
	
	for(ipix=digiIter->begin(); ipix!=digiIter->end(); ++ipix)
	  {
	    temp_->Fill(ipix->getRow(), ipix->getCol());
	    std::cout << "detId: " << detId << " " << ipix->getRow() << " " << ipix->getCol() << std::endl;	  
	  }
	
      } // end of the if statement asking if the plaquette in question has any errors in it    
      
    }// end of the for loop that goes through all plaquettes

}


// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelErrorsDigisToCalibDigis::beginJob(const edm::EventSetup& iSetup)
{
  iSetup.get<TrackerDigiGeometryRecord>().get( geom_ );
  theHistogramIdWorker_ = new SiPixelHistogramId(siPixelProducerLabel_.label());

}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelErrorsDigisToCalibDigis::endJob() { 

  if (!outputFilename_.empty() && createOutputFile_)
    {
      edm::LogInfo("SiPixelErrorCalibDigis") << "Writing ROOT file to: " << outputFilename_ << std::endl;
      if ( &*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (outputFilename_);
    }
}

// ------------ helper functions ---------------------------------------------------------

MonitorElement* SiPixelErrorsDigisToCalibDigis::bookDQMHistogram2D(uint32_t detid, std::string name, std::string title, int nchX, double lowX, double highX, int nchY, double lowY, double highY)
{
  std::string hid = theHistogramIdWorker_->setHistoId(name,detid);
  return daqBE_->book2D(hid, title, nchX, lowX, highX, nchY, lowY, highY);
}

MonitorElement* SiPixelErrorsDigisToCalibDigis::bookDQMHistoPlaquetteSummary2D(uint32_t detid, std::string name,std::string title){

  //  std::cerr<< "Are we ever in this function0???"<< std::endl;
  DetId detId(detid);

  //  std::cerr<< "Are we ever in this function1???"<< std::endl;
  const TrackerGeometry &theTracker(*geom_);

  //  std::cerr<< "Are we ever in this function2???"<< std::endl;
  const PixelGeomDetUnit *theGeomDet = dynamic_cast<const PixelGeomDetUnit*> ( theTracker.idToDet(detId) ); 

  //  std::cerr<< "Are we ever in this function3???"<< std::endl;
  int maxcol = theGeomDet->specificTopology().ncolumns();

  //  std::cerr<< "Are we ever in this function4???"<< std::endl;
  int maxrow = theGeomDet->specificTopology().nrows();  

  std::string hid = theHistogramIdWorker_->setHistoId(name,detid);
  return daqBE_->book2D(hid,title,maxcol,0,maxcol,maxrow,0,maxrow);
}
