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
// $Id: SiPixelGainCalibrationAnalysis.cc,v 1.3 2007/08/01 20:33:05 fblekman Exp $
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
#include "TRandom.h"
//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelGainCalibrationAnalysis::SiPixelGainCalibrationAnalysis(const edm::ParameterSet& iConfig):
  recordName_(iConfig.getParameter<std::string>("record")),
  conf_(iConfig),
  appendMode_(conf_.getUntrackedParameter<bool>("appendDatabaseMode",false)),
  SiPixelGainCalibration_(0),
  SiPixelGainCalibrationService_(iConfig),
  eventno_counter_(0),
  src_( iConfig.getUntrackedParameter<std::string>("src","source")),
  instance_ (iConfig.getUntrackedParameter<std::string>("InputInstance","")),
  maxNfedIds_ ( iConfig.getUntrackedParameter<uint32_t>( "numberOfPixelFEDs" )),
  inputconfigfile_( iConfig.getUntrackedParameter<std::string>( "inputFileName","/afs/cern.ch/cms/Tracker/Pixel/forward/ryd/calib_070106d.dat" ) ),
  nrowsmax_( iConfig.getUntrackedParameter<uint32_t>( "numberOfPixelRows",160)),
  ncolsmax_( iConfig.getUntrackedParameter<uint32_t>( "numberOfPixelColumns",260)),
  nrocsmax_( iConfig.getUntrackedParameter<uint32_t>( "numberOfPixelROCs",24)),
  nchannelsmax_( iConfig.getUntrackedParameter<uint32_t>( "numberOfPixelFEDs",40)),
  chisq_threshold_(iConfig.getUntrackedParameter<double>( "chi2CutoffForFileSave",3.0)),
  vcal_fitmin_(256),
  vcal_fitmax_(0),
  vcal_fitmax_fixed_( iConfig.getUntrackedParameter<uint32_t>( "cutoffVCalFit",100)),
  maximum_ped_(iConfig.getUntrackedParameter<double>("maximumPedestal",120.)),
  maximum_gain_(iConfig.getUntrackedParameter<double>("maximumGain",5.)),
  detIDmap_size(-1),
  test_(iConfig.getUntrackedParameter<bool>("dummyData",false)),
  fitfuncrootformula_(iConfig.getUntrackedParameter<std::string>("rootFunctionForFit","pol1"))
{
   //now do what ever initialization is needed
  
  // std::auto_ptr <PixelCalib> bla( new PixelCalib(inputconfigfile_));
  PixelCalib tempcalib(inputconfigfile_);
  calib_ =tempcalib;

  // database vars
  ::putenv("CORAL_AUTH_USER=me");
  ::putenv("CORAL_AUTH_PASSWORD=test");
  if(test_)
    edm::LogInfo("INFO") << "Using test configuration, dummy data being entered into histograms..." << std::endl;
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
   uint32_t vcal = calib_.vcal_fromeventno(eventno_counter_);
   uint32_t state = eventno_counter_ / calib_.nTriggers();  
 
   //    std::cout << vcal << " " << calib_.vcal(state) << " " << calib_.nTriggers() << " " << calib_.nVcal() << " " << calib_.vcal_step() << " " << calib_.vcal_first() << " " << calib_.nTriggersTotal() << std::endl;
   vcal = calib_.vcal(state);
   // std::cout << "end of event number: "<< eventno_counter_ << ", state: " << state << " " << calib_.vcal_fromeventno(eventno_counter_) << " " << vcal << std::endl;
  
   // get tracker geometry

   iSetup.get<TrackerDigiGeometryRecord>().get( geom_ );
   const TrackerGeometry& theTracker(*geom_);
   // Get event setup (to get global transformation)  
   edm::Handle< edm::DetSetVector<PixelDigi> > pixelDigis;
   iEvent.getByLabel( src_, pixelDigis );
   edm::DetSetVector<PixelDigi>::const_iterator digiIter;
   //   edm::LogInfo("SiPixelGainCalibrationAnalysis") << "SiPixelGainCalibrationAnalysis:found " << pixelDigis->size() << " digi hits..." << std::endl;
   for (digiIter = pixelDigis->begin() ; digiIter != pixelDigis->end(); digiIter++){// loop over detector units...
     uint32_t detid = digiIter->id;   // this is the raw idetector ID...
     DetId detId(detid);
     uint32_t detType = detId.det();
     uint32_t detsubId = detId.subdetId();
     if(detsubId!=2){
       edm::LogError("SiPixelGainCalibrationAnalysis")<<" Beware: expected forward pixel detector and looking at detector with subdetector type = " << detsubId << "(1=barrel, 2=forward, else=unknown). Det Type = " << detType <<  std::endl;
       //       continue;
     }
     PXFDetId pdetId = PXFDetId(detid);
     const PixelGeomDetUnit *theGeomDet = dynamic_cast<const PixelGeomDetUnit*> ( theTracker.idToDet(detId) );   
     int maxcol = theGeomDet->specificTopology().ncolumns();
     int maxrow = theGeomDet->specificTopology().nrows();
     uint32_t vcal = calib_.vcal_fromeventno(eventno_counter_);
     uint32_t nvcal = calib_.nVcal();
     uint32_t vcalstep=calib_.vcal_step();
     uint32_t vcalfirst=calib_.vcal_first();
     uint32_t vcallast=calib_.vcal_last();
     uint32_t ntriggers = calib_.nTriggers();
     TempPixelContainer myPixel;
     myPixel.pix_id = detid;
     myPixel.vcal_first=vcalfirst;
     myPixel.vcal_last=vcallast;
     myPixel.vcal_step=vcalstep;
     myPixel.maxrow=maxrow;
     myPixel.maxcol=maxcol;
     myPixel.vcal=vcal;
     myPixel.ntriggers=ntriggers;
     myPixel.nvcal= nvcal;
     if(test_){
       if(eventno_counter_>nvcal)
	 continue;
       myPixel.vcal = myPixel.vcal_first+(myPixel.vcal_step*eventno_counter_);
       myPixel.ntriggers=1;
       for(uint32_t icol=0.4*myPixel.maxcol; icol< 0.6*myPixel.maxcol; icol++){
	 for(uint32_t irow=0.4*myPixel.maxrow;irow<0.6*myPixel.maxrow; irow++){
	   
	   myPixel.adc=myPixel.vcal;
	   myPixel.row=irow;
	   myPixel.col=icol;
	   fill(myPixel);
	   irow++;
	 }
	 icol++;
       }
     }
     else{// if no test
       edm::DetSet<PixelDigi>::const_iterator ipix;   
       for(ipix = digiIter->data.begin(); ipix!=digiIter->end(); ipix++){
	 
	 myPixel.roc_id = 0;
	 myPixel.dcol_id = 0;
	 myPixel.adc=ipix->adc();
	 myPixel.row=ipix->row();
	 myPixel.col=ipix->column();
	 fill(myPixel);
       }
     }
   } 
     
   eventno_counter_++;
}

//*****************

void SiPixelGainCalibrationAnalysis::init(const TempPixelContainer &aPixel ){

  // std::cout << detIDmap_[aPixel.pix_id] << " " << aPixel.pix_id << " " << sname << " " << detIDmap_size << std::endl;
  if(detIDmap_[aPixel.pix_id]>calib_containers_.size() || (detIDmap_[aPixel.pix_id]==0 && calib_containers_.size()!=0) || detIDmap_size==-1 ){
    detIDmap_[aPixel.pix_id]=calib_containers_.size(); 
    TString name = "Pixel ID ";
    name+= aPixel.pix_id;
    std::string sname = name.Data();
    //    PixelROCGainCalib * gaincalib = new PixelROCGainCalib(aPixel.maxrow,aPixel.maxcol);
 
    calib_containers_.push_back(PixelROCGainCalib(aPixel.maxrow,aPixel.maxcol,aPixel.nvcal));
    calib_containers_[calib_containers_.size()-1].init(sname, aPixel.pix_id,aPixel.nvcal,aPixel.vcal_first, aPixel.vcal_last,aPixel.vcal_step,aPixel.maxcol,aPixel.maxrow,aPixel.ntriggers,therootfileservice_);
    detIDmap_size=calib_containers_.size();
  }
  return;
}
//*****************
void SiPixelGainCalibrationAnalysis::fill(const TempPixelContainer & aPixel)
{// method that does all the entry from the object
  
  uint32_t idet = detIDmap_[aPixel.pix_id];
  //  std::cout << idet << " " << aPixel.pix_id << std::endl;
  if(idet==0 && detIDmap_.size()!=0){
    //    edm::LogInfo("SiPixelGainCalibrationAnalysis")
    //    std::cout<< "!!!!!!!!!!!!\nadding pixel with det ID: " << aPixel.pix_id << std::endl;
    init(aPixel);
  }
  else if(idet==0){
    //    std::cout<< "!!!!!!!!!!!!\nadding pixel with det ID: " << aPixel.pix_id << std::endl;
    init(aPixel);
  }
  idet = detIDmap_[aPixel.pix_id];
  
  
  
  
  //  edm::LogInfo("SiPixelGainCalibrationAnalysis") 
  //calculate vcal point:

  calib_containers_[idet].fillVcal(aPixel.row,aPixel.col,aPixel.vcal,aPixel.adc,test_); 
  edm::LogInfo("DEBUG")  << "SiPixelGainCalibrationAnalysis::fill() : filling pixel, entry at col,row "<< aPixel.col<<","<<aPixel.row << " det ID " << aPixel.pix_id << " vcal = " << aPixel.vcal << " " << aPixel.adc << ", label " << idet << " in det map : " << detIDmap_[aPixel.pix_id] << std::endl;
}
// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelGainCalibrationAnalysis::beginJob(const edm::EventSetup&)
{
  //  edm::LogInfo("SiPixelGainCalibrationAnalysis") <<"beginjob: " << nrowsmax_ << " " << ncolsmax_ << " " << nrocsmax_ << " " << nchannelsmax_ << std::endl;

}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelGainCalibrationAnalysis::endJob() {
  edm::LogInfo("SiPixelGainCalibrationAnalysis")  << "!!!!!!!!!!!\nstarting end loop, looking at " <<  detIDmap_.size() << " different detector IDs" << std::endl;
  // this is where all the fitting etc is done
  
  
  vcal_fitmin_ =  calib_.vcal_first();
  vcal_fitmax_ =  calib_.vcal_last();
  if(vcal_fitmax_>vcal_fitmax_fixed_)
    vcal_fitmax_=vcal_fitmax_fixed_;
  TF1 *func = new TF1("func",fitfuncrootformula_.c_str(),vcal_fitmin_,vcal_fitmax_);
  func->SetParameters(0,1);
  edm::LogInfo("SiPixelGainCalibrationAnalysis") << "fitting function in range : " << vcal_fitmin_ << " " << vcal_fitmax_ << std::endl;
   //fancyfitfunction ( new TF1("fancyfitfunction","[0]+(([2]-[0*0.5*(1+TMath::Erf((x-[3])/([1]*sqrt(x)))))",0,256) );
  // now do the big loop over everything...

  SiPixelGainCalibration_ = new SiPixelGainCalibration();// create database objects
  
  for(uint32_t itot=0; itot<calib_containers_.size(); itot++){
    if(calib_containers_[itot].getNentries()==0){
      std::cout << "det ID : " << calib_containers_[itot].getDetID() << " is empty ..." << std::endl;
      continue;
    }
    std::cout << "det ID : " << calib_containers_[itot].getDetID() << " is NOT empty, it has " << calib_containers_[itot].getNentries() << " entries" <<  std::endl;
    uint32_t detid = calib_containers_[itot].getDetID();
    uint32_t ncols = calib_containers_[itot].getNcols();
    uint32_t nrows = calib_containers_[itot].getNrows();
    std::string worktitle =calib_containers_[itot].getTitle();
    std::string tempstring = worktitle+"_ped";
    std::string temptitle = "pedestals for " +worktitle;
    TH2F *peds = therootfileservice_->make<TH2F>(tempstring.c_str(),temptitle.c_str(),nrows,0,nrows,ncols,0,ncols);
    peds->SetDrawOption("colz");
    tempstring = worktitle+"_ped_1d";
    TH1F *peds1d = therootfileservice_->make<TH1F>(tempstring.c_str(),temptitle.c_str(),256,0,256);
    tempstring = worktitle+"_gain";
    temptitle = "gain for "+worktitle;
    TH2F *gains = therootfileservice_->make<TH2F>(tempstring.c_str(),temptitle.c_str(),nrows,0,nrows,ncols,0,ncols);
    gains->SetDrawOption("colz");
    tempstring = worktitle+"_gain_1d";
    TH1F *gains1d =therootfileservice_->make<TH1F>(tempstring.c_str(),temptitle.c_str(),100,0,10);
    
    std::vector<char> theSiPixelGainCalibration;
    for(uint32_t irow=0; irow<nrows; ++irow){
      for(uint32_t icol=0; icol<ncols;++icol){
	if(!calib_containers_[itot].isfilled(irow,icol)){

	  edm::LogInfo("DEBUG") << "detector : " << detid << " row " << irow << " col " << icol << " is  empty!!!" << std::endl;
	}
	else{
	  edm::LogInfo("DEBUG") << "detector : " << detid << " row " << irow << " col " << icol << " is NOT empty!!!" << std::endl;
	  TH1F *gr = (TH1F*)calib_containers_[itot].gethisto(irow,icol,therootfileservice_);
	  if(gr->GetEntries()<gr->GetNbinsX()*.5){// if less than half of the bins are filled: who cares!
	    calib_containers_[itot].printsummary(irow,icol);
	    TString newname = gr->GetName();
	    newname+=" NO FIT";
	    gr->SetName(newname);
	  }
	  else{
	    gr->Fit(func,"RQ");
	    if(func->GetChisquare()>0){
	      CalParameters theParameters;
	      theParameters.ped=func->GetParameter(0);
	      theParameters.gain=func->GetParameter(1);
	      peds->Fill(irow,icol,theParameters.ped);
	      gains->Fill(irow,icol,theParameters.gain);
	      peds1d->Fill(theParameters.ped);
	      gains1d->Fill(theParameters.gain);
	      
	      float theEncodedGain  = SiPixelGainCalibrationService_.encodeGain(theParameters.gain);
	      float theEncodedPed   = SiPixelGainCalibrationService_.encodePed (theParameters.ped);
	      edm::LogInfo("SiPixelGainCalibrationAnalysis") << "gains: " << theEncodedGain << " " << theParameters.gain << std::endl;
	      edm::LogInfo("SiPixelGainCalibrationAnalysis") << "peds: " << theEncodedPed << " " << theParameters.ped << std::endl;
	      SiPixelGainCalibration_->setData( theEncodedPed, theEncodedGain, theSiPixelGainCalibration);
	    }
	  }// good histogram gr.
	}
      }// loop over columns
    }//loop over rows
    SiPixelGainCalibration::Range range(theSiPixelGainCalibration.begin(),theSiPixelGainCalibration.end());
    if( !SiPixelGainCalibration_->put(detid,range,ncols) )
      edm::LogError("SiPixelGainCalibrationAnalysis")<<"[SiPixelGainCalibrationAnalysis: detid already exists in database!!!"<<std::endl;
    
  }// loop over detids

  
  // code copied more-or-less directly from CondTools/SiPixel/test/SiPixelCondObjBuilder.cc
  std::cout << " NOW filling database, this can take a while..." << std::endl;
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
  // end of database filling...

  // clean up:
  
  calib_containers_.clear();
  
}

// additional methods...

