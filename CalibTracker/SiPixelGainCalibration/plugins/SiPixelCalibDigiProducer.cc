// -*- C++ -*-
//
// Package:    SiPixelCalibDigiProducer
// Class:      SiPixelCalibDigiProducer
// 
/**\class SiPixelCalibDigiProducer SiPixelCalibDigiProducer.cc CalibTracker/SiPixelCalibDigiProducer/src/SiPixelCalibDigiProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Wed Oct 31 15:28:52 CET 2007
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/SiPixelCalibDigi/interface/SiPixelCalibDigifwd.h"
#include "DataFormats/SiPixelCalibDigi/interface/SiPixelCalibDigi.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include "CalibFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CondFormats/DataRecord/interface/SiPixelCalibConfigurationRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 

#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "SiPixelCalibDigiProducer.h"

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelCalibDigiProducer::SiPixelCalibDigiProducer(const edm::ParameterSet& iConfig):
  src_(iConfig.getParameter<edm::InputTag>("src")),
  iEventCounter_(0),
  conf_(iConfig)
{
   //register your products
  produces< edm::DetSetVector<SiPixelCalibDigi> >();

}


SiPixelCalibDigiProducer::~SiPixelCalibDigiProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

/////////////////////////////////////////////
// function description:
// this function checks where/when the pattern changes so that one can fill the data from the temporary container into the event 
bool 
SiPixelCalibDigiProducer::store()
{
  if(iEventCounter_%pattern_repeat_==0){
    edm::LogInfo("INFO") << "now at event " << iEventCounter_ <<" where we save the calibration information into the CMSSW digi";
    return 1;
  }
  else
    return 0;
  return 1;
}
////////////////////////////////////////////////////////////////////
// function description:
// fill function, uses maps to keep track of which pixel is where in the local storage container. Called every event to fill and compress the digi data into calibdig format
void
SiPixelCalibDigiProducer::fill(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  // figure out which calibration point we're on now..
  short icalibpoint = calib_->vcalIndexForEvent(iEventCounter_);
  edm::Handle< edm::DetSetVector<PixelDigi> > pixelDigis;
  iEvent.getByLabel( src_, pixelDigis );

  // loop over the data and store things
  edm::DetSetVector<PixelDigi>::const_iterator digiIter;
  for(digiIter=pixelDigis->begin(); digiIter!=pixelDigis->end(); ++digiIter){// ITERATOR OVER DET IDs
    uint32_t detid = digiIter->id;
    edm::DetSet<PixelDigi>::const_iterator ipix; // ITERATOR OVER DIGI DATA  

    for(ipix = digiIter->data.begin(); ipix!=digiIter->end(); ++ipix){
      edm::LogInfo("DEBUG") << "now looking at pixel (row,col):" << ipix->row() << "," << ipix->column() << " with adc counts " << ipix->adc() ;

      // fill in the appropriate location of the temporary data container
      fillPixel(detid,ipix->row(),ipix->column(),icalibpoint,ipix->adc());
    }
  }
}
////////////////////////////////////////////////////
// function description:
// this is the function where we look in the maps to find the correct calibration digi container, after which the data is filled.
void SiPixelCalibDigiProducer::fillPixel(uint32_t detid, short row, short col, short ipoint, short adc){
  bool createnewdetid=false;
  bool createnewpixel=false;
  if(detPixelMap_.size()==0){ 			  
    createnewdetid=true;
    createnewpixel=true;
  }
  else{
    std::map<uint32_t,std::vector<std::pair<short,short> > >::const_iterator imap = detPixelMap_.find(detid);
    if(imap->first!=detid){
      createnewdetid=true;
      createnewpixel=true;
    }
    else{
      std::pair<short, short> pix(row,col);
      bool foundpixel=0;
      for(uint32_t ii=0; ii<detPixelMap_[detid].size();++ii){
	if(detPixelMap_[detid][ii]==pix)
	  foundpixel=true;
      }
      if(!foundpixel)
	createnewpixel=true;
    }
  }
  if(createnewdetid){
    std::vector<std::pair<short,short> > vec(0);
    detPixelMap_[detid]=vec; 
    std::vector<SiPixelCalibDigi> digivec(0,SiPixelCalibDigi(calib_->nVCal()));
    intermediate_data_[detid]=digivec; 	
  }
  if(createnewpixel){
    std::pair<short, short> pix(row,col);
    detPixelMap_[detid].push_back(pix);
    if(intermediate_data_[detid].size()<detPixelMap_[detid].size()){
      SiPixelCalibDigi digi(calib_->nVCal());
      intermediate_data_[detid].push_back(digi);
      intermediate_data_[detid][intermediate_data_[detid].size()-1].setrowcol((short) row, (short) col);
    }
    else{
      intermediate_data_[detid][detPixelMap_[detid].size()-1].reset();
      intermediate_data_[detid][detPixelMap_[detid].size()-1].setrowcol((short) row, (short) col);
    }
  }
  
  // now actually fill the object,this unfortunately involves looping, 
  // I don't know how to deal with the pixel indices except yet another map... :(
  std::pair<short, short> inputpix(row,col);
  for(uint32_t ii=0; ii<detPixelMap_[detid].size(); ++ii){
    if(detPixelMap_[detid][ii]==inputpix)
      intermediate_data_[detid][ii].fill(ipoint,adc);
  }

  
}
//////////////////////////////////////////////////////////////
// function description:
// this function cleans up after everything in the pattern is filled. This involves setting the content of the intermediate_data_ containers to zero and completely emptying the map 
void
SiPixelCalibDigiProducer::clear(){

  // this is where we empty the containers so they can be re-filled
  // the idea: the detPixelMap_ container shrinks/expands as a function 
  // of the number of pixels looked at at one particular time... 
  // unlike the intermediate_data_ container which only expands when 
  // detPixelMap_ becomes bigger than intermedate_data_
  
  // shrink the detPixelMap_

  for(std::map<uint32_t,std::vector<std::pair<short, short> > >::const_iterator idet = detPixelMap_.begin(); idet!=detPixelMap_.end(); ++idet){
    while(detPixelMap_[idet->first].size()!=0){
      detPixelMap_[idet->first].erase(detPixelMap_[idet->first].end());
    }
  }

  // and clear the SiPixelCalibDigi objects
  for(std::map<uint32_t,std::vector<SiPixelCalibDigi> >::const_iterator idet = intermediate_data_.begin(); idet!=intermediate_data_.end(); ++idet){
    for(uint32_t ii=0; ii<intermediate_data_[idet->first].size();++ii){
      intermediate_data_[idet->first][ii].reset();
    }
  }
}

////////////////////////////////////////////////////
// function description:
// This method gets the pattern from the calib_ (SiPixelCalibConfiguration) object and fills a vector of pairs that is easier to check 
void 
SiPixelCalibDigiProducer::setPattern(){

  uint32_t patternnumber = (iEventCounter_-1)/pattern_repeat_;
  uint32_t rowpatternnumber = patternnumber/calib_->nRowPatterns();
  uint32_t colpatternnumber = patternnumber%calib_->nColumnPatterns();
  // update currentpattern_
  std::vector<short> calibcols = calib_->getColumnPattern();
  std::vector<short> calibrows = calib_->getRowPattern();
  std::vector<short> temprowvals(0);
  std::vector<short> tempcolvals(0);
  uint32_t nminuscol=0;
  uint32_t nminusrow=0;
  uint32_t npatterns=0;
  for(uint32_t icol=0; icol<calibcols.size(); icol++){
    if(calibcols[icol]==-1){
      nminuscol++;
    }
    else if(nminuscol==colpatternnumber){
      short val=calibcols[icol];
      tempcolvals.push_back(val);
    }
    else if(nminuscol>colpatternnumber)
      break;
  }
  for(uint32_t irow=0; irow<calibrows.size(); irow++){
    if(calibrows[irow]==-1)
      nminusrow++;
    else if(nminusrow==rowpatternnumber){
      short val=calibrows[irow];
      temprowvals.push_back(val);
    }
    else if(nminusrow>rowpatternnumber)
      break;
  }
  //now clean up the currentpattern_;
  while(currentpattern_.size()>temprowvals.size()*tempcolvals.size()){
    currentpattern_.erase(currentpattern_.end());
  }
  for(uint32_t irow=0; irow<temprowvals.size(); irow++){
    for(uint32_t icol=0; icol<tempcolvals.size(); icol++){
      std::pair<short,short> pattern(temprowvals[irow],tempcolvals[icol]);
      npatterns++;
      if(npatterns>currentpattern_.size())
	currentpattern_.push_back(pattern);
      else
	currentpattern_[npatterns-1]=pattern;
    }
  }
  std::cout << "summary of created patterns: " ;
  for(uint32_t i=0; i<currentpattern_.size(); ++i){
    if(i!=0)
      std::cout << " - ";
    std::cout << currentpattern_[i].first << ","<< currentpattern_[i].second ;
   
  }
  std::cout << std::endl;
}

////////////////////////////////////////////
// function description:
// produce method. This is the main loop method
void
SiPixelCalibDigiProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  iEventCounter_++;
  fill(iEvent,iSetup); // fill method where the actual looping over the digis is done.
  std::auto_ptr<edm::DetSetVector<SiPixelCalibDigi> > pOut(new edm::DetSetVector<SiPixelCalibDigi>);
  // copy things over if necessary (this is only once per 
  if(store()){
    setPattern();
    for(std::map<uint32_t,std::vector<SiPixelCalibDigi> >::const_iterator idet=intermediate_data_.begin(); idet!=intermediate_data_.end(); ++idet){
      uint32_t detid = idet->first;
      edm::DetSet<SiPixelCalibDigi> & detSet = pOut->find_or_insert(detid);
      std::vector<SiPixelCalibDigi> tempdata(0);
      for(std::vector<SiPixelCalibDigi>::const_iterator idigi=idet->second.begin(); idigi!= idet->second.end(); ++idigi){
	//	std::cout << "pixel " << detid << " "  << idigi->row() << " " << idigi->col() << " has " << idigi->getnpoints() << " entries." << std::endl;
	if(checkPixel(detid,idigi->row(),idigi->col())){
	  // now fill
	  tempdata.push_back(*idigi);
	}
      }
      detSet.data = tempdata;
    }
    edm::LogInfo("INFO") << "now filling event " << iEventCounter_ << " as pixel pattern changes every " <<  pattern_repeat_ << " events..." << std::endl;
    clear();
  }
  iEvent.put(pOut);
}
//-----------------------------------------------
//  method to check that the pixels are actually valid...
bool SiPixelCalibDigiProducer::checkPixel(uint32_t detid, short row, short col){
  
  const TrackerGeometry& theTracker(*geom_);
  DetId detId(detid);
  uint32_t detsubId = detId.subdetId();
  uint32_t maxcol,maxrow;
  if(detsubId==1){//BPIX
    PXBDetId pdetId = PXBDetId(detid);
    const PixelGeomDetUnit *theGeomDet = dynamic_cast<const PixelGeomDetUnit*> ( theTracker.idToDet(detId) );   
    maxcol = theGeomDet->specificTopology().ncolumns();
    maxrow = theGeomDet->specificTopology().nrows();
   
  }
  else if(detsubId==2){ // FPIX
    PXFDetId pdetId = PXFDetId(detid);
    const PixelGeomDetUnit *theGeomDet = dynamic_cast<const PixelGeomDetUnit*> ( theTracker.idToDet(detId) );   
    maxcol = theGeomDet->specificTopology().ncolumns();
    maxrow = theGeomDet->specificTopology().nrows();
  }
  else{//not a pixel detid!!
    return false;
  }
  PixelIndices indexer(maxcol,maxrow);

  int roccol, rocrow, iroc;
  indexer.transformToROC(col,row,iroc,roccol,rocrow);
  // now get the pattern from the calib object...
 
 
  //  edm::LogInfo("INFO")
  
  // now loop over the pattern for this event and check:
  currentpair_.first=rocrow;
  currentpair_.second=roccol;
  for(uint32_t i=0; i<currentpattern_.size(); ++i){
    if(currentpair_==currentpattern_[i])
      return true;
  }
  edm::LogError("ERROR") << "DETID " << detid << ", row, col (offline)="<< row << ","<< col <<  "row, col (ROC) ="<<rocrow<<","<< roccol << ". This is a " << maxrow/80. << "x" << maxcol/52. << " device. " << std::endl;
  return false;
}


// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelCalibDigiProducer::beginJob(const edm::EventSetup& iSetup)
{
  iSetup.get<SiPixelCalibConfigurationRcd>().get(calib_);
  pattern_repeat_=calib_->getNTriggers()*calib_->nVCal();
  //iSetup.get<SiPixelFedCablingMapRcd>().get( cablingmap_ );
  iSetup.get<TrackerDigiGeometryRecord>().get( geom_ );
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelCalibDigiProducer::endJob() {
}

