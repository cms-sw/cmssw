// -*- C++ -*-
//
// Package:    SiPixelGainCalibrationAnalysis
// Class:      SiPixelGainCalibrationAnalysis
// 
/**\class SiPixelGainCalibrationAnalysis SiPixelGainCalibrationAnalysis.cc CalibTracker/SiPixelGainCalibrationAnalysis/src/SiPixelGainCalibrationAnalysis.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Mon May  7 14:22:37 CEST 2007
// $Id: SiPixelGainCalibrationAnalysis.cc,v 1.1 2007/05/20 18:08:09 fblekman Exp $
//
//


// system include files
#include <memory>
#include <map>

// user include files
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "PhysicsTools/UtilAlgos/interface/TFileDirectory.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
// other classes
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "CalibTracker/SiPixelGainCalibration/interface/SiPixelGainCalibrationAnalysis.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
// root classes
#include "TF1.h"
//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelGainCalibrationAnalysis::SiPixelGainCalibrationAnalysis(const edm::ParameterSet& iConfig):
  conf_(iConfig),
  appendMode_(conf_.getUntrackedParameter<bool>("appendDatabaseMode",false)),
  SiPixelGainCalibration_(0),
  SiPixelGainCalibrationService_(iConfig),
  recordName_(iConfig.getParameter<std::string>("record")),
  eventno_counter_(0),
  src_( iConfig.getUntrackedParameter<std::string>("src","source")),
  instance_ (iConfig.getUntrackedParameter<std::string>("InputInstance","")),
  maxNfedIds_ ( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelFEDs" )),
  inputconfigfile_( iConfig.getUntrackedParameter<std::string>( "inputFileName","/afs/cern.ch/cms/Tracker/Pixel/forward/ryd/calib_070106d.dat" ) ),
  nrowsmax_( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelRows",80)),
  ncolsmax_( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelColumns",52)),
  nrocsmax_( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelROCs",24)),
  nchannelsmax_( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelFEDs",40)),
  vcal_fitmax_fixed_( iConfig.getUntrackedParameter<unsigned int>( "cutoffVCalFit",100)),
  chisq_threshold_(iConfig.getUntrackedParameter<double>( "chi2CutoffForFileSave",3.0)),
  maximum_gain_(iConfig.getUntrackedParameter<double>("maximumGain",5.)),
  maximum_ped_(iConfig.getUntrackedParameter<double>("maximumPedestal",120.)),
  vcal_fitmin_(256),
  vcal_fitmax_(0),
  detIDmap_size(0) 
{
   //now do what ever initialization is needed
  calib_ = new PixelCalib(inputconfigfile_);
  hitworker_ = new PixelSLinkDataHit();
  // database vars
  ::putenv("CORAL_AUTH_USER=me");
  ::putenv("CORAL_AUTH_PASSWORD=test");
}


SiPixelGainCalibrationAnalysis::~SiPixelGainCalibrationAnalysis()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//


// ------------ method called to for each event  ------------
void
SiPixelGainCalibrationAnalysis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   unsigned int vcal = calib_->vcal_fromeventno(eventno_counter_);
   eventno_counter_++;
   // get tracker geometry

   iSetup.get<TrackerDigiGeometryRecord>().get( geom_ );
   const TrackerGeometry& theTracker(*geom_);

   // Get event setup (to get global transformation)  
   edm::Handle< edm::DetSetVector<PixelDigi> > pixelDigis;
   iEvent.getByLabel( src_, pixelDigis );
   edm::DetSetVector<PixelDigi>::const_iterator digiIter;
   //   edm::LogInfo("SiPixelGainCalibrationAnalysis") << "SiPixelGainCalibrationAnalysis:found " << pixelDigis->size() << " digi hits..." << std::endl;
   for (digiIter = pixelDigis->begin() ; digiIter != pixelDigis->end(); digiIter++){// loop over detector units...
     unsigned int detid = digiIter->id;   // this is the raw idetector ID...
     DetId detId(detid);
     unsigned int detType = detId.det();
     unsigned int detsubId = detId.subdetId();
     if(detsubId!=2){
       edm::LogError("SiPixelGainCalibrationAnalysis")<<" Beware: expected forward pixel detector and looking at detector with subdetector type = " << detsubId << "(1=barrel, 2=forward, else=unknown)" << std::endl;
       //       continue;
     }
     PXFDetId pdetId = PXFDetId(detid);
     edm::LogInfo("SiPixelGainCalibrationAnalysis") << "SiPixelGainCalibrationAnalysis: DetID" << detid << " " << detType << " " << detsubId << " " << detId.rawId() <<", disk " << pdetId.disk() <<  " blade " << pdetId.blade() << " module " << pdetId.module() << " side " << pdetId.side() << " panel " << pdetId.panel() << std::endl;
     //loop over pixels in det unit...
     // get maximum number of rows/columns from det ID:
     
     const PixelGeomDetUnit *theGeomDet = dynamic_cast<const PixelGeomDetUnit*> ( theTracker.idToDet(detId) );   
     int maxcol = theGeomDet->specificTopology().ncolumns();
     int maxrow = theGeomDet->specificTopology().nrows();
     maxcol=52;
     maxrow=80;
     edm::DetSet<PixelDigi>::const_iterator ipix;
     for(ipix = digiIter->data.begin(); ipix!=digiIter->end(); ipix++){
       TempPixelContainer myPixel;
       myPixel.fed_channel =ipix->channel();
       myPixel.roc_id = 0;
       myPixel.dcol_id = 0;
       myPixel.pix_id = detid;
       myPixel.maxrow=maxrow;
       myPixel.maxcol=maxcol;
       myPixel.adc=ipix->adc();
       myPixel.row=ipix->row();
       myPixel.col=ipix->column();
       myPixel.vcal=vcal;
       if(myPixel.vcal<vcal_fitmin_)
	 vcal_fitmin_=myPixel.vcal;
       if(myPixel.vcal>vcal_fitmax_)
	 vcal_fitmax_=myPixel.vcal;
       //       edm::LogInfo("SiPixelGainCalibrationAnalysis") << "vcal:" << myPixel.vcal << " column:" << myPixel.col << " row:" << myPixel.row << " adc:" << myPixel.adc << " detID:" << myPixel.pix_id << std::endl;
	 
       fill(myPixel);
     }
   }
}

//*****************
void SiPixelGainCalibrationAnalysis::init(const TempPixelContainer &aPixel ){
  detIDmap_[aPixel.pix_id]++;
  calib_containers_[detIDmap_[aPixel.pix_id]].init(aPixel.fed_channel,aPixel.roc_id,calib_->nVcal(),calib_->vcal_first(), calib_->vcal_last(),calib_->vcal_step(),aPixel.maxcol,aPixel.maxrow);
  calib_containers_[detIDmap_[aPixel.pix_id]].setDetID(aPixel.pix_id);
  detIDmap_size = detIDmap_[aPixel.pix_id];
  //  edm::LogInfo("SiPixelGainCalibrationAnalysis") << "creating pixel at ROC with ID " << aPixel.pix_id << ", counter " << detIDmap_[aPixel.pix_id] << std::endl;
  
  return;
}
//*****************
void SiPixelGainCalibrationAnalysis::fill(const TempPixelContainer & aPixel)
{// method that does all the entry from the object
 
  if(detIDmap_[aPixel.pix_id]==0){
    //    edm::LogInfo("SiPixelGainCalibrationAnalysis") << "!!!!!!!!!!!!\nadding pixel with ID: " << aPixel.pix_id << std::endl;
    init(aPixel);
  }
  unsigned int i = detIDmap_[aPixel.pix_id];
  
  //  edm::LogInfo("SiPixelGainCalibrationAnalysis") << "SiPixelGainCalibrationAnalysis::fill() : filling pixel, entry at " << i << " det ID " << detIDmap_[i] << std::endl;
  calib_containers_[i].fill(aPixel.row,aPixel.col,aPixel.vcal,aPixel.adc);
  // keep track of minimum and maximum range...
  if(aPixel.vcal<vcal_fitmin_){
    vcal_fitmin_=aPixel.vcal;
  }
  if(aPixel.vcal>vcal_fitmax_)
    vcal_fitmax_=aPixel.vcal;
}
// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelGainCalibrationAnalysis::beginJob(const edm::EventSetup&)
{
  //  edm::LogInfo("SiPixelGainCalibrationAnalysis") <<"beginjob: " << nrowsmax_ << " " << ncolsmax_ << " " << nrocsmax_ << " " << nchannelsmax_ << std::endl;
  assert(nrowsmax_ <= 80);
  assert(ncolsmax_ <= 52);
  assert(nrocsmax_ <= 24);
  assert(nchannelsmax_ <= 40);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelGainCalibrationAnalysis::endJob() {
  //  edm::LogInfo("SiPixelGainCalibrationAnalysis") << "!!!!!!!!!!!\nstarting end loop, looking at " <<  detIDmap_.size() << " different detector IDs" << std::endl;
  // this is where all the fitting etc is done
  if(vcal_fitmax_>vcal_fitmax_fixed_)
    vcal_fitmax_=vcal_fitmax_fixed_;
  if(vcal_fitmin_>vcal_fitmax_)
    vcal_fitmin_=calib_->vcal_first();
  if(vcal_fitmin_>=vcal_fitmax_){
    vcal_fitmin_=0;
    vcal_fitmax_=256;
  }
  
  SiPixelGainCalibration_ = new SiPixelGainCalibration();// create database objects


  
  //  edm::LogInfo("SiPixelGainCalibrationAnalysis") << "fitting function in range : " << vcal_fitmin_ << " " << vcal_fitmax_ << std::endl;
  fitfunction = new TF1("fitfunction","pol1",vcal_fitmin_*0.99,vcal_fitmax_*1.01);// straight line just overlapping vcal range
  fancyfitfunction = new TF1("fancyfitfunction","[0]+(([2]-[0])*0.5*(1+TMath::Erf((x-[3])/([1]*sqrt(x)))))",vcal_fitmin_*0.99,vcal_fitmax_*1.01);
  fitfunction->SetParameters(0,1);
  fancyfitfunction->SetParameters(100,3,250,50);
  // now do the big loop over everything...

  edm::LogInfo("SiPixelGainCalibrationAnalysis") << "number of calibrations: " << detIDmap_.size() << std::endl;
  edm::Service<TFileService> therootfileservice;
  uint32_t tempcounter=0;
  for(std::map<unsigned int, unsigned int>::iterator detiditer = detIDmap_.begin(); detiditer!=detIDmap_.end(); detiditer++){
    edm::LogInfo("SiPixelGainCalibrationAnalysis") << detiditer->first << " " << detiditer->second << std::endl;
    unsigned int itot = detiditer->second;
    uint32_t detid = calib_containers_[itot].getDetID();
    edm::LogInfo("SiPixelGainCalibrationAnalysis") << detid << " and from MAP: " << itot << " " << detIDmap_[itot] << std::endl;
    
    std::string worktitle =calib_containers_[itot].getTitle().Data();
    std::string tempstring = worktitle+"ped";
    std::string temptitle = "pedestals for " +worktitle;
    TH2F *peds = therootfileservice->make<TH2F>(tempstring.c_str(),temptitle.c_str(),nrowsmax_,0,nrowsmax_,ncolsmax_,0,ncolsmax_);
    tempstring = worktitle+"ped1d";
    TH1F *peds1d = therootfileservice->make<TH1F>(tempstring.c_str(),temptitle.c_str(),256,0,256);
    tempstring = worktitle+"gain";
    temptitle = "gain for "+worktitle;
    TH2F *gains = therootfileservice->make<TH2F>(tempstring.c_str(),temptitle.c_str(),nrowsmax_,0,nrowsmax_,ncolsmax_,0,ncolsmax_);
    tempstring = worktitle+"gain1d";
    TH1F *gains1d = therootfileservice->make<TH1F>(tempstring.c_str(),temptitle.c_str(),100,0,10);
    //    TFileDirectory chanrocsubdir=therootfileservice->mkdir(worktitle);
    
    std::vector<char> theSiPixelGainCalibration;
    for(unsigned int irow=0; irow<nrowsmax_; ++irow){
      for(unsigned int icol=0; icol<ncolsmax_;++icol){
	if(!calib_containers_[itot].isvalid(irow,icol))
	  continue;
	edm::LogInfo("SiPixelGainCalibrationAnalysis") << irow << " " << icol << " " << temptitle << " "<<itot << " " << detid << std::endl;
	//TH1F *hist = (TH1F*) calib_containers_[ichan][iroc].getHistoFileService(chanrocsubdir,irow,icol);
	therootfileservice->mkdir("pixels");
	TH1F *hist = (TH1F*)calib_containers_[itot].gethisto(irow,icol);
	
	if(!hist)
	  continue;
	hist->Fit(fitfunction,"RQ0");
	//	  calib_containers_[ichan][iroc].fit(irow,icol,fitfunction);
	// do some check on fitfunction...
	CalParameters theParameters;
	theParameters.ped=fitfunction->GetParameter(0);
	theParameters.gain=fitfunction->GetParameter(1);
	peds->Fill(irow,icol,theParameters.ped);
	gains->Fill(irow,icol,theParameters.gain);
	peds1d->Fill(theParameters.ped);
	gains1d->Fill(theParameters.gain);
	if(0){// don't do this
	  //	  if(fitfunction->GetChisquare()>chisq_threshold_){// save to file 
	  fancyfitfunction->SetParameter(0,theParameters.ped);
	  fancyfitfunction->SetParameter(1,theParameters.gain);
	  hist->Fit(fancyfitfunction);
	  edm::LogInfo("SiPixelGainCalibrationAnalysis") << "chisquare etc..: " << fitfunction->GetChisquare() << " " << fitfunction->GetNDF() << std::endl;
	  edm::LogInfo("SiPixelGainCalibrationAnalysis") << "new chisquare etc..: " << fancyfitfunction->GetChisquare() << " " << fancyfitfunction->GetNDF() << std::endl;
	  theParameters.ped=fancyfitfunction->GetParameter(0);
	  theParameters.gain=fancyfitfunction->GetParameter(1);
	}
	float theEncodedGain  = SiPixelGainCalibrationService_.encodeGain(theParameters.gain);
	float theEncodedPed   = SiPixelGainCalibrationService_.encodePed (theParameters.ped);
	edm::LogInfo("SiPixelGainCalibrationAnalysis") << "gains: " << theEncodedGain << " " << theParameters.gain << std::endl;
	edm::LogInfo("SiPixelGainCalibrationAnalysis") << "peds: " << theEncodedPed << " " << theParameters.ped << std::endl;
	SiPixelGainCalibration_->setData( theEncodedPed, theEncodedGain, theSiPixelGainCalibration);
      }
    }
    SiPixelGainCalibration::Range range(theSiPixelGainCalibration.begin(),theSiPixelGainCalibration.end());
    if( !SiPixelGainCalibration_->put(detid,range,ncolsmax_) )
      edm::LogError("SiPixelGainCalibrationAnalysis")<<"[SiPixelGainCalibrationAnalysis: detid already exists in database!!!"<<std::endl;
    
  }// loop over detids
  edm::LogInfo("SiPixelGainCalibrationAnalysis") << "number of calibrations: " << detIDmap_.size() << std::endl;
  for(std::map<unsigned int, unsigned int>::iterator detiditer = detIDmap_.begin(); detiditer!=detIDmap_.end(); detiditer++)
    edm::LogInfo("SiPixelGainCalibrationAnalysis") << detiditer->first << " " << detiditer->second << std::endl;
  // code copied more-or-less directly from CondTools/SiPixel/test/SiPixelCondObjBuilder.cc
  edm::LogInfo(" --- writing to DB!");
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if(!mydbservice.isAvailable() ){
    edm::LogError("db service unavailable");
    return;
  } 
  else { edm::LogInfo("DB service OK"); }
  
  try{
    if( mydbservice->isNewTagRequest(recordName_) ){
      mydbservice->createNewIOV<SiPixelGainCalibration>(SiPixelGainCalibration_, mydbservice->endOfTime(),recordName_);
    } 
    else {
      mydbservice->appendSinceTime<SiPixelGainCalibration>(SiPixelGainCalibration_, mydbservice->currentTime(),recordName_);
    }
    edm::LogInfo(" --- all OK");
  } 
  catch(const cond::Exception& er){
    edm::LogError("SiPixelCondObjBuilder")<<er.what()<<std::endl;
  } 
  catch(const std::exception& er){
    edm::LogError("SiPixelCondObjBuilder")<<"caught std::exception "<<er.what()<<std::endl;
  }
  catch(...){
    edm::LogError("SiPixelCondObjBuilder")<<"Funny error"<<std::endl;
  }
  
}

// additional methods...

