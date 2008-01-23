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
//         Created:  Wed Nov 14 15:02:06 CET 2007
// $Id: SiPixelGainCalibrationAnalysis.cc,v 1.5 2007/12/20 18:02:31 fblekman Exp $
//
//

// user include files
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "SiPixelGainCalibrationAnalysis.h"
#include "TLinearFitter.h"
#include <sstream>
//
// constructors and destructor
//
SiPixelGainCalibrationAnalysis::SiPixelGainCalibrationAnalysis(const edm::ParameterSet& iConfig):
  SiPixelOfflineCalibAnalysisBase(iConfig),
  conf_(iConfig),
  bookkeeper_(),
  bookkeeper_pixels_(),
  reject_badpoints_(iConfig.getUntrackedParameter<bool>("suppressZeroAndPlateausInFit",true)),
  reject_badpoints_frac_(iConfig.getUntrackedParameter<double>("suppressZeroAndPlateausInFitFrac",0)),
  chi2Threshold_(iConfig.getUntrackedParameter<double>("minChi2forHistSave",10)),
  maxGainInHist_(iConfig.getUntrackedParameter<double>("maxGainInHist",10)),
  maxChi2InHist_(iConfig.getUntrackedParameter<double>("maxChi2InHist",25)),
  filldb_(iConfig.getUntrackedParameter<bool>("writeDatabase",false)),
  recordName_(conf_.getParameter<std::string>("record")),
  appendMode_(conf_.getUntrackedParameter<bool>("appendMode",true)),
  theGainCalibrationDbInput_(0),
  theGainCalibrationDbInputService_(iConfig)
{
  ::putenv("CORAL_AUTH_USER=me");
  ::putenv("CORAL_AUTH_PASSWORD=test");   
}

SiPixelGainCalibrationAnalysis::~SiPixelGainCalibrationAnalysis()
{
}
// member functions
//
// ------------ method called once each job just before starting event loop  ------------
void SiPixelGainCalibrationAnalysis::calibrationSetup(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelGainCalibrationAnalysis::calibrationEnd() {

  // this is where we loop over all histograms and save the database objects
  for(std::map<uint32_t,std::map<std::string, MonitorElement *> >::const_iterator idet=bookkeeper_.begin(); idet!= bookkeeper_.end(); ++idet){
    // now filling stuff
    //    std::cout << "now looking at detid " << idet->first << std::endl;
  }
  if(filldb_)
    fillDatabase();
}
//-----------method to fill the database
void SiPixelGainCalibrationAnalysis::fillDatabase(){

  uint32_t nchannels=0;
  uint32_t nmodules=0;
  for(std::map<uint32_t,std::map<std::string, MonitorElement *> >::const_iterator idet=bookkeeper_.begin(); idet!= bookkeeper_.end(); ++idet){
    uint32_t detid=idet->first;
    // Get the module sizes.
    int nrows = bookkeeper_[detid]["gain2d"]->getNbinsY();
    int ncols = bookkeeper_[detid]["ped2d"]->getNbinsX();   
    
    std::vector<char> theSiPixelGainCalibration;

    // Loop over columns and rows of this DetID
    for(int i=0; i<ncols; i++) {
      for(int j=0; j<nrows; j++) {
	nchannels++;
	     
	float ped = bookkeeper_[detid]["ped2d"]->getBinContent(i,j);
	float gain = bookkeeper_[detid]["gain2d"]->getBinContent(i,j);

	theGainCalibrationDbInput_->setData( ped , gain , theSiPixelGainCalibration);
      }
    }

    SiPixelGainCalibration::Range range(theSiPixelGainCalibration.begin(),theSiPixelGainCalibration.end());
    if( !theGainCalibrationDbInput_->put(detid,range,ncols) )
      edm::LogError("SiPixelGainCalibrationAnalysis")<<"warning: detid already exists"<<std::endl;
  }
  std::cout << " ---> PIXEL Modules  " << nmodules  << std::endl;
  std::cout << " ---> PIXEL Channels " << nchannels << std::endl;

  edm::LogInfo(" --- writing to DB!");
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if(!mydbservice.isAvailable() ){
    edm::LogError("db service unavailable");
    return;
    if( mydbservice->isNewTagRequest(recordName_) ){
      mydbservice->createNewIOV<SiPixelGainCalibration>(
							theGainCalibrationDbInput_, mydbservice->endOfTime(),recordName_);
    } else {
      mydbservice->appendSinceTime<SiPixelGainCalibration>(
							   theGainCalibrationDbInput_, mydbservice->currentTime(),recordName_);
    }
    edm::LogInfo(" --- all OK");
  } 
}
// ------------ method called to do fits to all objects available  ------------
bool
SiPixelGainCalibrationAnalysis::doFits(uint32_t detid, std::vector<SiPixelCalibDigi>::const_iterator ipix)
{

  // first, fill the input arrays to the TLinearFitter.
  double xvals[100];
  double yvals[100];
  double yerrvals[100];
  float xvalsall[200];
  float yvalsall[200];
  float yerrvalsall[200];
  int npoints=0;
  for(uint32_t ipoint = 0; ipoint < ipix->getnpoints(); ++ipoint){
    float minrange = 255.*ipix->getnentries(ipoint)*reject_badpoints_frac_;
    float maxrange = (255*ipix->getnentries(ipoint)) - minrange;
    if(ipoint<200){

      xvalsall[ipoint]=vCalValues_[ipoint];
      yvalsall[ipoint]=ipix->getsum(ipoint);
      yerrvalsall[ipoint]=ipix->getsumsquares(ipoint)-pow(ipix->getsum(ipoint),2);
      if(ipix->getnentries(ipoint)>0){
	yvalsall[ipoint]/=ipix->getnentries(ipoint);
	yerrvalsall[ipoint]=sqrt(fabs(yerrvalsall[ipoint]))/pow(ipix->getnentries(ipoint),2);
      }
    }
    else
      continue;
    if(ipix->getsum(ipoint)<=minrange && reject_badpoints_)
      continue;
    if(ipix->getsum(ipoint)>=maxrange && reject_badpoints_)
       continue;
    if(npoints>=100)
      continue;
    xvals[npoints]=xvalsall[ipoint];
    yvals[npoints]=yvalsall[ipoint];
    yerrvals[npoints]=yerrvalsall[ipoint];

    //std::cout << xvals[npoints] << " "<< yvals[npoints] << " " << yerrvals[npoints] << std::endl;
    npoints++;

    
  }
  if(npoints<2){
    return false;
  }
  TLinearFitter fitter(2,"pol1");
  fitter.AssignData(npoints,2,xvals,yvals,yerrvals);

  // and do the fit:
  int result = fitter.Eval();
  if(result==1)
    return false;
  // it is also possible to do fitter.EvalRobust(), at which point outlyers are ignored
  
  float slope = fitter.GetParameter(1);
  float intercept = fitter.GetParameter(0);
  float chi2 = fitter.GetChisquare();
  chi2/=(float)npoints;
    
  bookkeeper_[detid]["gain_1d"]->Fill(slope);
  bookkeeper_[detid]["gain_2d"]->Fill(ipix->col(),ipix->row(),slope);
  bookkeeper_[detid]["ped_1d"]->Fill(intercept);
  bookkeeper_[detid]["ped_2d"]->Fill(ipix->col(),ipix->row(),intercept);
  bookkeeper_[detid]["chi2_1d"]->Fill(chi2);
  bookkeeper_[detid]["chi2_2d"]->Fill(ipix->col(),ipix->row(),chi2);
  //  std::cout << "leaving doFits" << std::endl;
  
  if(chi2>chi2Threshold_ && chi2Threshold_>0.){
    setDQMDirectory(detid);
    std::ostringstream pixelinfo;
    pixelinfo << "row_" << ipix->row() << "_col_" << ipix->col();
    std::string tempname=translateDetIdToString(detid);
    tempname+="_";
    tempname+=pixelinfo.str();
     // and book the histo
    bookkeeper_pixels_[detid][pixelinfo.str()] =  bookDQMHistogram1D(pixelinfo.str(),tempname,ipix->getnpoints()-1,xvalsall);
    for(uint32_t ii=0; ii<ipix->getnpoints(); ++ii){
      bookkeeper_pixels_[detid][pixelinfo.str()]->setBinContent(ii+1,yvalsall[ii]);
      bookkeeper_pixels_[detid][pixelinfo.str()]->setBinError(ii+1,yerrvalsall[ii]);
    }
  }
  //  edm::LogInfo("SiPixelGainCalibrationAnalysis") << "looking at Det ID " << detid << " "<< translateDetIdToString(detid) << ", pixel " << ipix->row() << " , " << ipix->col() << " gain= " << slope << ", pedestal= " << intercept << " chi2/NDOF= " << chi2 << " fit result value " << result << std::endl;
  return true;
}
// ------------ method called to do fill new detids  ------------
void 
SiPixelGainCalibrationAnalysis::newDetID(uint32_t detid)
{
  setDQMDirectory(detid);
  std::string tempname=translateDetIdToString(detid);
  //std::cout << "creating new histograms..."<< tempname << std::endl;
  bookkeeper_[detid]["gain_1d"] = bookDQMHistogram1D("gain_1d_"+tempname,"gain for "+tempname,100,0.,maxGainInHist_);
  bookkeeper_[detid]["gain_2d"] = bookDQMHistoPlaquetteSummary2D("gain_2d_"+tempname,"gain for "+tempname,detid);
  bookkeeper_[detid]["ped_1d"] = bookDQMHistogram1D("pedestal_1d_"+tempname,"pedestal for "+tempname,256,0.,256.);
  bookkeeper_[detid]["ped_2d"] = bookDQMHistoPlaquetteSummary2D("pedestal_2d_"+tempname,"pedestal for "+tempname,detid);
  bookkeeper_[detid]["chi2_1d"] = bookDQMHistogram1D("chi2_1d_"+tempname,"#chi^{2}/NDOF for "+tempname,100,0.,maxChi2InHist_);
  bookkeeper_[detid]["chi2_2d"] = bookDQMHistoPlaquetteSummary2D("chi2_2d_"+tempname,"#chi^{2}/NDOF for "+tempname,detid);

  //std::cout << "leaving new detid" << std::endl;
}
//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelGainCalibrationAnalysis);
