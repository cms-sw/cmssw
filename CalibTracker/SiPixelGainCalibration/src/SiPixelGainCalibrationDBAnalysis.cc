// -*- C++ -*-
//
// Package:    SiPixelGainCalibrationDBAnalysis
// Class:      SiPixelGainCalibrationDBAnalysis
// 
/**\class SiPixelGainCalibrationDBAnalysis SiPixelGainCalibrationDBAnalysis.cc CalibTracker/SiPixelGainCalibrationDBAnalysis/src/SiPixelGainCalibrationDBAnalysis.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Mon May  7 14:22:37 CEST 2007
// $Id: SiPixelGainCalibrationDBAnalysis.cc,v 1.1 2007/05/20 18:08:09 fblekman Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
// other classes
#include "CalibTracker/SiPixelGainCalibration/interface/SiPixelGainCalibrationDBAnalysis.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
// root classes
#include "TF1.h"
#include "TFile.h"
//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelGainCalibrationDBAnalysis::SiPixelGainCalibrationDBAnalysis(const edm::ParameterSet& iConfig):
  conf_(iConfig),
  appendMode_(conf_.getUntrackedParameter<bool>("appendDatabaseMode",false)),
  SiPixelGainCalibration_(0),
  SiPixelGainCalibrationService_(iConfig),
  recordName_(iConfig.getParameter<std::string>("record")),
  eventno_counter_(0),
  maxNfedIds_ ( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelFEDs" )),
  inputconfigfile_( iConfig.getUntrackedParameter<std::string>( "inputFileName","/afs/cern.ch/cms/Tracker/Pixel/forward/ryd/calib_070106d.dat" ) ),
  rootoutputfilename_( iConfig.getUntrackedParameter<std::string>( "rootFileName","histograms_for_monitoring.root" ) ),
  nrowsmax_( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelRows",80)),
  ncolsmax_( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelColumns",52)),
  nrocsmax_( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelROCs",24)),
  nchannelsmax_( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelFEDs",40)),
  vcal_fitmax_fixed_( iConfig.getUntrackedParameter<unsigned int>( "cutoffVCalFit",100)),
  chisq_threshold_(iConfig.getUntrackedParameter<double>( "chi2CutoffForFileSave",3.0)),
  maximum_gain_(iConfig.getUntrackedParameter<double>("maximumGain",5.)),
  maximum_ped_(iConfig.getUntrackedParameter<double>("maximumPedestal",120.)),
  vcal_fitmin_(256),
  vcal_fitmax_(0)
   
{
   //now do what ever initialization is needed
  calib_ = new PixelCalib(inputconfigfile_);
  hitworker_ = new PixelSLinkDataHit();
  // database vars
  ::putenv("CORAL_AUTH_USER=me");
  ::putenv("CORAL_AUTH_PASSWORD=test"); 
}


SiPixelGainCalibrationDBAnalysis::~SiPixelGainCalibrationDBAnalysis()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//


// ------------ method called to for each event  ------------
void
SiPixelGainCalibrationDBAnalysis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   unsigned int vcal = calib_->vcal_fromeventno(eventno_counter_);
   eventno_counter_++;
  
   Handle<FEDRawDataCollection> fedRawDataCollection;
   iEvent.getByType(fedRawDataCollection);// nasty, we should do a label check.... TO BE FIXED
   for (unsigned int ifedId=0;ifedId<maxNfedIds_;++ifedId){
     const FEDRawData& fedRawData = fedRawDataCollection->FEDData( ifedId );
     datasize_=fedRawData.size();
     for(datacounter_=0;datacounter_<datasize_/8;++datacounter_){
       //       unsigned long long thedata = get_next_block(fedRawData.data(),datacounter_);
       hitworker_->SetTheData(get_next_block(fedRawData.data(),datacounter_));
       if(!hitworker_->IsValidData())
	 continue;
       // now hitworker_ is filled with the next two hits...
       for(unsigned int ihit=0; ihit<2; ++ihit){
	 hitworker_->SetHit(ihit);
	 if(!hitworker_->IsValidData())
	   continue;
	 TempPixelContainer myPixel;
	 myPixel.fed_channel = hitworker_->GetFEDChannel();
	 myPixel.roc_id = hitworker_->GetROC_ID();
	 myPixel.dcol_id = hitworker_->GetDcol_ID();
	 myPixel.pix_id = hitworker_->GetPix_ID();
	 myPixel.adc=hitworker_->GetADC_count();
	 myPixel.row=hitworker_->GetPixelRow();
	 myPixel.col=myPixel.dcol_id*2+myPixel.pix_id%2;
	 myPixel.vcal=vcal;
	 if(myPixel.vcal<vcal_fitmin_)
	   vcal_fitmin_=myPixel.vcal;
	 if(myPixel.vcal>vcal_fitmax_)
	   vcal_fitmax_=myPixel.vcal;
	 //	   std::cout << myPixel.vcal << " " << myPixel.col << " " << myPixel.row << " " << myPixel.adc << std::endl;
	 
	 fill(myPixel);
       }
     }// while valid data is coming out
   }// end of loop over feds
}

//*****************
void SiPixelGainCalibrationDBAnalysis::init(const TempPixelContainer & aPixel ){
  if(rocgainused_[aPixel.fed_channel][aPixel.roc_id])
    return;
  rocgainused_[aPixel.fed_channel][aPixel.roc_id]=true;
  calib_containers_[aPixel.fed_channel][aPixel.roc_id].init(aPixel.fed_channel,aPixel.roc_id,calib_->nVcal(),calib_->vcal_first(), calib_->vcal_last(),calib_->vcal_step(),52,80);
  return;
}
//*****************
void SiPixelGainCalibrationDBAnalysis::fill(const TempPixelContainer & aPixel)
{// method that does all the entry from the object
  if(!rocgainused_[aPixel.fed_channel][aPixel.roc_id])
    init(aPixel);
  calib_containers_[aPixel.fed_channel][aPixel.roc_id].fill(aPixel.row,aPixel.col,aPixel.vcal,aPixel.adc);
 
}
// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelGainCalibrationDBAnalysis::beginJob(const edm::EventSetup&)
{
  std::cout <<"beginjob: " << nrowsmax_ << " " << ncolsmax_ << " " << nrocsmax_ << " " << nchannelsmax_ << std::endl;
  assert(nrowsmax_ <= 80);
  assert(ncolsmax_ <= 52);
  assert(nrocsmax_ <= 24);
  assert(nchannelsmax_ <= 40);
  for(unsigned int ichan=0; ichan<nchannelsmax_;++ichan){
    for(unsigned int iroc=0; iroc<nrocsmax_;++iroc){
      rocgainused_[ichan][iroc]=false;
    }
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelGainCalibrationDBAnalysis::endJob() {
  std::cout << "starting end loop:" << std::endl;
  // this is where all the fitting etc is done
  if(vcal_fitmax_>vcal_fitmax_fixed_)
    vcal_fitmax_=vcal_fitmax_fixed_;
  if(vcal_fitmin_>vcal_fitmax_)
    vcal_fitmin_=calib_->vcal_first();
  if(vcal_fitmin_>vcal_fitmax_){
    vcal_fitmin_=0;
    vcal_fitmax_=256;
  }
  SiPixelGainCalibration_ = new SiPixelGainCalibration();// create database objects


  
  std::cout << "fitting function in range : " << vcal_fitmin_ << " " << vcal_fitmax_ << std::endl;
  TF1 *fitfunction = new TF1("fitfunction","pol1",vcal_fitmin_*0.99,vcal_fitmax_*1.01);// straight line just overlapping vcal range
  fitfunction->SetParameters(0,1);
  TF1 *fancyfitfunction = new TF1("fancyfitfunction","[0]+(([2]-[0])*0.5*(1+TMath::Erf((x-[3])/([1]*sqrt(x)))))",vcal_fitmin_*0.99,vcal_fitmax_*1.01);
  fancyfitfunction->SetParameters(100,3,250,50);
  // now do the big loop over everything...
  TFile *rootfile = new TFile(rootoutputfilename_.c_str(),"recreate");
  gROOT->cd();
  uint32_t tempcounter=0;
  for(unsigned int ichan=0; ichan<nchannelsmax_;++ichan){
    for(unsigned int iroc=0; iroc<nrocsmax_; ++iroc){
      if(!rocgainused_[ichan][iroc])
	continue; 
      std::vector<char> theSiPixelGainCalibration;
      for(unsigned int irow=0; irow<nrowsmax_; ++irow){
	for(unsigned int icol=0; icol<ncolsmax_;++icol){
	  if(!calib_containers_[ichan][iroc].isvalid(irow,icol))
	    continue;
	  //	  std::cout << irow << " " << icol << " " << ichan << " " << iroc << std::endl;
	  gROOT->cd();
	  //	  rootfile->cd();
	  TH1F *hist = (TH1F*)calib_containers_[ichan][iroc].gethisto(irow,icol);
	  if(!hist)
	    continue;
	  hist->Fit(fitfunction,"RQ0");
	  //	  calib_containers_[ichan][iroc].fit(irow,icol,fitfunction);
	  // do some check on fitfunction...
	  CalParameters theParameters;
	  theParameters.ped=fitfunction->GetParameter(0);
	  theParameters.gain=fitfunction->GetParameter(1);

	  if(fitfunction->GetChisquare()>chisq_threshold_){// save to file 
	    fancyfitfunction->SetParameter(0,theParameters.ped);
 	    fancyfitfunction->SetParameter(1,theParameters.gain);
 	    hist->Fit(fancyfitfunction);
	    std::cout << "chisquare etc..: " << fitfunction->GetChisquare() << " " << fitfunction->GetNDF() << std::endl;
	    std::cout << "new chisquare etc..: " << fancyfitfunction->GetChisquare() << " " << fancyfitfunction->GetNDF() << std::endl;
	    theParameters.ped=fancyfitfunction->GetParameter(0);
 	    theParameters.gain=fancyfitfunction->GetParameter(1);
	    rootfile->cd();
	    hist->Write(hist->GetName());
	  }
	  float theEncodedGain  = SiPixelGainCalibrationService_.encodeGain(theParameters.gain);
	  float theEncodedPed   = SiPixelGainCalibrationService_.encodePed (theParameters.ped);
	  std::cout << "gains: " << theEncodedGain << " " << theParameters.gain << std::endl;
	  std::cout << "peds: " << theEncodedPed << " " << theParameters.ped << std::endl;
	  SiPixelGainCalibration_->setData( theEncodedPed, theEncodedGain, theSiPixelGainCalibration);
	}
      }
      uint32_t detid = tempcounter++;
      std::cout << detid << std::endl;
      SiPixelGainCalibration::Range range(theSiPixelGainCalibration.begin(),theSiPixelGainCalibration.end());
      if( !SiPixelGainCalibration_->put(detid,range,ncolsmax_) )
	edm::LogError("SiPixelCondObjBuilder")<<"[SiPixelCondObjBuilder::analyze] detid already exists"<<std::endl;
    }// loop over rocs
  }// loop over channels
  rootfile->Write();
  rootfile->Close(); 
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

