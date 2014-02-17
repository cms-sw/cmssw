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
// $Id: SiPixelErrorsDigisToCalibDigis.cc,v 1.10 2010/08/10 09:06:13 ursl Exp $
//
//


// system include files
#include <memory>

#include "SiPixelErrorsDigisToCalibDigis.h"

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

//  std::cout<<"siPixelProducerLabel_ = "<<siPixelProducerLabel_<<std::endl;
//  std::cout<<"createOutputFile_= "<< createOutputFile_<<std::endl;
//  std::cout<<"outpuFilename_= "<< outputFilename_<< std::endl;
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

  static int first(1); 
  if (1 == first) {
    first = 0; 
    iSetup.get<TrackerDigiGeometryRecord>().get( geom_ );
    theHistogramIdWorker_ = new SiPixelHistogramId(siPixelProducerLabel_.label());
  }
  
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
//	    std::cout << "This is the beginning of an error 2d histo booking: "<<std::endl;
	    setDQMDirectory(detId);
	    temp_ = bookDQMHistoPlaquetteSummary2D(detId, "SiPixelErrorsCalibDigis", "SiPixelErrorsDigisToCalibDigis");
	    SiPixelErrorsDigisToCalibDigis_2DErrorInformation_.insert( std::make_pair(detId,temp_));
	  }
	else
	  {
//	    std::cout << "This one was already booked."<<std::endl;
	    temp_ = (*mapIterator).second;
	  }
	
	for(ipix=digiIter->begin(); ipix!=digiIter->end(); ++ipix)
	  {
	    temp_->Fill(ipix->getCol(), ipix->getRow());
//	    std::cout << "detId: " << detId << " " << ipix->getRow() << " " << ipix->getCol() << std::endl;	  
	  }
	
      } // end of the if statement asking if the plaquette in question has any errors in it    
      
    }// end of the for loop that goes through all plaquettes

}


// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelErrorsDigisToCalibDigis::beginJob()
{

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

  DetId detId(detid);
  const TrackerGeometry &theTracker(*geom_);
  const PixelGeomDetUnit *theGeomDet = dynamic_cast<const PixelGeomDetUnit*> ( theTracker.idToDet(detId) ); 
  int maxcol = theGeomDet->specificTopology().ncolumns();
  int maxrow = theGeomDet->specificTopology().nrows();  

  std::string hid = theHistogramIdWorker_->setHistoId(name,detid);
  return daqBE_->book2D(hid,title,maxcol,0,maxcol,maxrow,0,maxrow);
}

bool SiPixelErrorsDigisToCalibDigis::setDQMDirectory(std::string dirName)
{
   daqBE_->setCurrentFolder(dirName);
   return daqBE_->dirExists(dirName);
}

bool SiPixelErrorsDigisToCalibDigis::setDQMDirectory(uint32_t detID)
{
  return folderMaker_->setModuleFolder(detID,0);
}

// -- define this as a plug-in
DEFINE_FWK_MODULE(SiPixelErrorsDigisToCalibDigis);
