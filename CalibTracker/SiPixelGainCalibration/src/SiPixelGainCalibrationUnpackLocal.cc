// -*- C++ -*-
//
// Package:    SiPixelGainCalibrationUnpackLocal
// Class:      SiPixelGainCalibrationUnpackLocal
// 
/**\class SiPixelGainCalibrationUnpackLocal SiPixelGainCalibrationUnpackLocal.cc SiPixelGainCalibrationUnpackLocal/SiPixelGainCalibrationUnpackLocal/src/SiPixelGainCalibrationUnpackLocal.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Thu Apr 26 10:38:32 CEST 2007
// $Id$
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
// calibration information and worker classes
#include "CalibTracker/SiPixelGainCalibration/interface/PixelCalib.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibHists.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelSLinkDataHit.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include <string>
#include <vector>
#include <iostream>
#include "TFile.h"
#include "TF1.h"
#include "CalibTracker/SiPixelGainCalibration/interface/SiPixelGainCalibrationUnpackLocal.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelGainCalibrationUnpackLocal::SiPixelGainCalibrationUnpackLocal(const edm::ParameterSet& iConfig) 
  : eventno_counter(0),
    inputfile_( iConfig.getUntrackedParameter<std::string>( "inputFileName","/afs/cern.ch/cms/Tracker/Pixel/forward/ryd/calib_070106d.dat" ) ), 
    outputfilename_( iConfig.getUntrackedParameter<std::string>( "outputRootFileName","histograms_for_monitoring.root" ) ), 
    calib_(0),
    nrowsmax_( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelRows",80)),
    ncolsmax_( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelColumns",52)),
    nrocsmax_( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelROCs",24)),
    nchannelsmax_( iConfig.getUntrackedParameter<unsigned int>( "numberOfPixelFEDs",40)),
    vcalminfit_( iConfig.getUntrackedParameter<unsigned int>( "minVCalFit",15))
,
    vcalmaxfit_( iConfig.getUntrackedParameter<unsigned int>( "maxVCalFit",100)),
    save_everything_( iConfig.getUntrackedParameter<bool>( "saveAllHistos",false)),
    database_access_( iConfig.getUntrackedParameter<bool>( "doDatabase",false))//,
								    //    pixelCalibrationDBService_( iConfig )
{
  edm::LogVerbatim("") << "***********************************" << std::endl;
  //now do what ever initialization is needed

  // make some ugly asserts so the user can't do anything too stupid

  ::putenv("CORAL_AUTH_USER=me");
  ::putenv("CORAL_AUTH_PASSWORD=test"); 
  
  assert(nrowsmax_ <= 80);
  assert(ncolsmax_ <= 52);
  assert(nrocsmax_ <= 24);
  assert(nchannelsmax_ <= 40);
  for(unsigned int iroc=0; iroc < nrocsmax_; ++iroc)
    for(unsigned int ichan = 0; ichan< nchannelsmax_; ++ichan)
      rocgainused_[ichan][iroc]=false;
  
  
}


SiPixelGainCalibrationUnpackLocal::~SiPixelGainCalibrationUnpackLocal()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
SiPixelGainCalibrationUnpackLocal::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
    
  
  unsigned int vcal = calib_->vcal_fromeventno(eventno_counter);
  eventno_counter++;
  Handle<FEDRawDataCollection> fedRawDataCollection;
  iEvent.getByType(fedRawDataCollection);// nasty, we should do a label check.... TO BE FIXED
 
  //I know that we are the pixels, fedid=0-39, but this
  //should be defined in some headerfile.
  PixelSLinkDataHit hitworker; //Who needs hex when you can have a class that thinks for you!
 
  for (unsigned int fed_id=0;fed_id<40;fed_id++){
    const FEDRawData& fedRawData = fedRawDataCollection->FEDData( fed_id );

    //   edm::LogVerbatim("") << "fedid="<<fed_id<<" size="<<fedRawData.size()<<std::endl;

    unsigned int datasize=fedRawData.size();

    const unsigned char* dataptr=fedRawData.data();

    for( unsigned int i=0;i<datasize/8;i++ ){

      
	
      unsigned long long data=((const unsigned long long*)dataptr)[i];
      //edm::LogVerbatim("") << "i="<<i<<" data="<<std::hex<<data<<std::dec<<std::endl;
      
      hitworker.SetTheData(data);
      if( !hitworker.IsValidData() )
	continue;
      for(int ihit=0; ihit<2; ++ihit){
	hitworker.SetHit(ihit);
	if(!hitworker.IsValidData())
	  continue;
	// now get the stuff out:
	unsigned int fed_channel=hitworker.GetFEDChannel();
	unsigned int roc_id=hitworker.GetROC_ID();
	unsigned int dcol_id=hitworker.GetDcol_ID();
	unsigned int pix_id=hitworker.GetPix_ID();
	unsigned int adc=hitworker.GetADC_count();
	unsigned int row=hitworker.GetPixelRow();
	unsigned int col=dcol_id*2+pix_id%2;
	if(!hitworker.IsValidHit())
	  continue;
	rocgainused_[fed_channel-1][roc_id-1]=true;
	rocgain_[fed_channel-1][roc_id-1].fill(row,col,vcal,adc);

      }
    }

  }

}


// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelGainCalibrationUnpackLocal::beginJob(const edm::EventSetup&)
{

  edm::LogVerbatim("") << "In SiPixelGainCalibrationUnpackLocal::beginJob" << std::endl;
  edm::LogVerbatim("") << "Reading in PixelCalib file: " << inputfile_ << std::endl;
  calib_=new PixelCalib(inputfile_);

  int nvcal=calib_->nVcal();	
  edm::LogVerbatim("") << calib_->vcal_first() << " " << calib_->vcal_last() << " " << calib_->vcal_step() << std::endl;
  edm::LogVerbatim("") << "number of calibrations: " << nvcal << std::endl;
  edm::LogVerbatim("")<< "number of triggers : " << calib_->nTriggersPerPattern() << std::endl;
  edm::LogVerbatim("") << "number of triggers (total): " << calib_->nTriggersTotal() << std::endl;
  for(unsigned int linkid=0;linkid<nchannelsmax_;linkid++){
    for(unsigned int rocid=0;rocid<nrocsmax_;rocid++){
      rocgain_[linkid][rocid]. init(linkid,rocid,nvcal);
      rocgain_[linkid][rocid]. setVCalRange(calib_->vcal_first(),calib_->vcal_last(),calib_->vcal_step());
    }
  }
    edm::LogVerbatim("") << "Leaving SiPixelGainCalibrationUnpackLocal::beginJob" << std::dec;
  edm::LogVerbatim("") << "***********************************" << std::endl;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelGainCalibrationUnpackLocal::endJob() {


  edm::LogVerbatim("") << "***********************************" << std::endl;
  edm::LogVerbatim("") << "In SiPixelGainCalibrationUnpackLocal::endJob" << std::dec;  


  // to be fixed, now uses standard pixel ID for each type of module
  
  outputfileformonitoring = new TFile(outputfilename_.c_str(),"recreate");
  edm::LogVerbatim("") << "Creating root output file with histograms: " << outputfileformonitoring->GetName() << std::endl;
  edm::LogVerbatim("") << "***********************************" << std::endl;
  // and plot/fit things:
  for(unsigned int linkid=0;linkid<nchannelsmax_;++linkid){
    for(unsigned int rocid=0;rocid<nrocsmax_;++rocid){ 
      if(!rocgainused_[linkid][rocid])
	continue;
      rocgain_[linkid][rocid].setFitRange(vcalminfit_,vcalminfit_);
      TString histonamehere = "overview_link_";
      histonamehere+=linkid;
      histonamehere+="_ROC_";
      histonamehere+=rocid;
      histonamehere.ReplaceAll(" ","");
      outputfileformonitoring->cd();
      // create overview plot
      TString histonametemp= histonamehere+"_slope";
      roc_summary_histos_slope[linkid][rocid] = new TH2F(histonametemp,histonametemp,nrowsmax_,0.,nrowsmax_,ncolsmax_,0.,ncolsmax_);
      histonametemp= histonamehere+"_intersect";
      roc_summary_histos_intersect[linkid][rocid] = new TH2F(histonametemp,histonametemp,nrowsmax_,0.,nrowsmax_,ncolsmax_,0.,ncolsmax_);
      roc_summary_histos_slope[linkid][rocid]->SetXTitle("row number");
      roc_summary_histos_slope[linkid][rocid]->SetYTitle("column number");
      roc_summary_histos_intersect[linkid][rocid]->SetXTitle("row number");
      roc_summary_histos_intersect[linkid][rocid]->SetYTitle("column number");
      for(unsigned int irow=0; irow<nrowsmax_; ++irow){
	for(unsigned int icol=0; icol<ncolsmax_; ++icol){
	  if(!rocgain_[linkid][rocid].filled(irow,icol))
	    continue;
	  TF1* temp = rocgain_[linkid][rocid].fit(irow,icol);
	  float gain=temp->GetParameter(1);
	  float ped=temp->GetParameter(0);
// 	  if(database_access_){
// 	    gain = pixelCalibrationDBService_ .encodeGain(temp->GetParameter(1));
// 	    ped = pixelCalibrationDBService_ .encodePed(temp->GetParameter(0));
// 	  }
	  std::cout << gain << " " << ped << " " << temp->GetParameter(1) << " "<< temp->GetParameter(0) << std::endl;
	  if(temp==0)
	    continue;
	  roc_summary_histos_slope[linkid][rocid]->Fill(irow,icol,gain);
	  roc_summary_histos_intersect[linkid][rocid]->Fill(irow,icol,ped);
	  //	  edm::LogVerbatim("") << "Uncertainties, fractional:" <<temp->GetParameter(0)/temp->GetParError(0) << ", " << temp->GetParameter(1)/temp->GetParameter(1) << " [%]"<< std::endl;
	  if(save_everything_)
	    rocgain_[linkid][rocid].save(irow,icol,outputfileformonitoring);
	}
      }
      // do it this way so the range is known..
      histonametemp= histonamehere+"_spreadslope";
      roc_summary_histos_slopevals[linkid][rocid] = new TH1F(histonametemp,histonamehere,100,0.8*roc_summary_histos_slope[linkid][rocid]->GetMinimum(),1.2*roc_summary_histos_slope[linkid][rocid]->GetMaximum());
      roc_summary_histos_slopevals[linkid][rocid]->SetXTitle("slope of fit");
      roc_summary_histos_slopevals[linkid][rocid]->SetYTitle("entries (pixels)");
      histonametemp= histonamehere+"_spreadint";
      roc_summary_histos_startvals[linkid][rocid] = new TH1F(histonametemp,histonamehere,100,0.8*roc_summary_histos_intersect[linkid][rocid]->GetMinimum(),1.2*roc_summary_histos_intersect[linkid][rocid]->GetMaximum());
      roc_summary_histos_startvals[linkid][rocid]->SetXTitle("slope of fit");
      roc_summary_histos_startvals[linkid][rocid]->SetYTitle("entries (pixels)");
      for(unsigned int irow=0; irow<nrowsmax_; ++irow){
	for(unsigned int icol=0; icol<ncolsmax_; ++icol){
	 
	  roc_summary_histos_startvals[linkid][rocid]->Fill(roc_summary_histos_intersect[linkid][rocid]->GetBinContent(irow+1,icol+1));
	  roc_summary_histos_slopevals[linkid][rocid]->Fill(roc_summary_histos_slope[linkid][rocid]->GetBinContent(irow+1,icol+1));
							    
	}
      }
    }
  }
  if(database_access_){
    // Write into DB
    
  }
  outputfileformonitoring->Write();
  outputfileformonitoring->Close();
  // and delete everything
  delete outputfileformonitoring;
  edm::LogVerbatim("") << "leaving SiPixelGainCalibrationUnpackLocal::endJob" << std::dec;  
 //  for(unsigned int linkid=0;linkid<nchannelsmax_;++linkid){
//     for(unsigned int rocid=0;rocid<nrocsmax_;++rocid){  
//       if(!rocgainused_[linkid][rocid])
// 	continue;
//       rocgain_[linkid][rocid].cleanup();
//     }
//   }

  if (calib_!=0) delete calib_; calib_=0;
  edm::LogVerbatim("") << "***********************************" << std::endl;
}
