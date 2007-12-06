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
// $Id: SiPixelGainCalibrationAnalysis.cc,v 1.1 2007/11/27 14:34:01 fblekman Exp $
//
//

// user include files
//#include "CalibTracker/SiPixelGainCalibration/interface/SiPixelGainCalibrationAnalysis.h"
#include "SiPixelGainCalibrationAnalysis.h"
//
// constructors and destructor
//
SiPixelGainCalibrationAnalysis::SiPixelGainCalibrationAnalysis(const edm::ParameterSet& iConfig):
  SiPixelOfflineCalibAnalysisBase(iConfig),
  reject_badpoints_(iConfig.getUntrackedParameter<bool>("suppressZeroAndPlateausInFit",true)),
  reject_badpoints_frac_(iConfig.getUntrackedParameter<double>("suppressZeroAndPlateausInFitFrac",0))
  //  recordName_(conf_.getParameter<std::string>("record")),
  //  appendMode_(conf_.getUntrackedParameter<bool>("appendMode",true)),
  //  theGainCalibrationDbInput_(0),
  //  theGainCalibrationDbInputService_(iConfig)
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
void 
SiPixelGainCalibrationAnalysis::calibrationSetup(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelGainCalibrationAnalysis::calibrationEnd() {

  // this is where we loop over all histograms and save the database objects
  for(std::map<uint32_t,std::map<std::string, MonitorElement *> >::const_iterator idet=bookkeeper_.begin(); idet!= bookkeeper_.end(); ++idet){
    // now filling stuff
    std::cout << "now looking at detid " << idet->first << std::endl;
    
  }
}

// ------------ method called to do fits to all objects available  ------------
bool
SiPixelGainCalibrationAnalysis::doFits(uint32_t detid, std::vector<SiPixelCalibDigi>::const_iterator ipix)
{std::cout << "looking at Det ID" << detid << ", pixel " << ipix->row() << "," << ipix->col() << std::endl;
  // inspired by Tony Kelly's analytical fit code
  double xpoints_mean_sum=0;
  double xpoints_sqmean_sum=0;
  double ypoints_mean_sum=0;
  double ypoints_sqmean_sum=0;
  double xpoints_mean=0;
  double xpoints_sqmean=0;
  double xpoints_meansq=0;
  double ypoints_mean=0;
  double ypoints_sqmean=0;
  double ypoints_meansq=0;
  //The following variables are for the regression line, defaulted to a horizontal line
  double slope_numerator=0;
  double slope_denominator=1;
  double slope = 0;
  double intercept=0;
  double regression_ydenominator=1;
  // regression is necesary for error calculation but not included yet.
  //double regression=0;
  //  double regression_square=0;
  double npoints_used=0;
  
  
  for(uint32_t ipoint = 0; ipoint < ipix->getnpoints(); ++ipoint){
    double minrange = 255.*ipix->getnentries(ipoint)*reject_badpoints_frac_;
    double maxrange = (255*ipix->getnentries(ipoint)) - minrange;
    if(ipix->getsum(ipoint)<=minrange && reject_badpoints_)
      continue;
    if(ipix->getsum(ipoint)>=maxrange && reject_badpoints_)
       continue;
    xpoints_mean_sum += vCalValues_[ipoint];
    ypoints_mean_sum += ipix->getsum(ipoint);
    xpoints_sqmean_sum += vCalValues_[ipoint]*vCalValues_[ipoint];
    ypoints_sqmean_sum += ipix->getsum(ipoint)*ipix->getsum(ipoint);
    npoints_used++;
  }
  if(npoints_used==0){
    return false;
  }
  xpoints_mean = xpoints_mean_sum/npoints_used;
  xpoints_sqmean=xpoints_sqmean_sum/npoints_used;
  xpoints_meansq=xpoints_mean*xpoints_mean;
  ypoints_mean = ypoints_mean_sum/npoints_used;
  ypoints_sqmean=ypoints_sqmean_sum/npoints_used;
  ypoints_meansq=ypoints_mean*ypoints_mean;

  for(uint32_t ipoint = 0; ipoint < ipix->getnpoints(); ++ipoint){ 
    if(ipix->getsum(ipoint)==0 && reject_badpoints_)
      continue;
    if(ipix->getsum(ipoint)>=250*ipix->getnentries(ipoint) && reject_badpoints_)
      continue;
    slope_numerator += (vCalValues_[ipoint]-xpoints_mean)*(ipix->getsum(ipoint)-ypoints_mean);
    slope_denominator += (vCalValues_[ipoint]-xpoints_mean)*(vCalValues_[ipoint]-xpoints_mean);
    regression_ydenominator += (ipix->getsum(ipoint)-ypoints_mean)*(ipix->getsum(ipoint)-ypoints_mean);
  }
  slope = slope_numerator/slope_denominator;
  intercept = ypoints_mean-(slope*xpoints_mean); 

  // numbering hard-coded in SiPixelGainCalibrationAnalysis::newDetID for now
  bookkeeper_[detid]["gain1d"]->Fill(slope);
  bookkeeper_[detid]["gain2d"]->Fill(ipix->col(),ipix->row(),slope);
  bookkeeper_[detid]["ped1d"]->Fill(intercept);
  bookkeeper_[detid]["ped2d"]->Fill(ipix->col(),ipix->row(),intercept);
  return true;
}
// ------------ method called to do fill new detids  ------------
void 
SiPixelGainCalibrationAnalysis::newDetID(short detid)
{
  setDQMDirectory(detid);
  std::string tempname=translateDetIdToString(detid);
  bookkeeper_[detid]["gain1d"] = bookDQMHistogram1D("gain1d_"+tempname,"gain for "+tempname,100,0.,100.);
  bookkeeper_[detid]["gain2d"] = bookDQMHistoPlaquetteSummary2D("gain2d_"+tempname,"gain for "+tempname,detid);
  bookkeeper_[detid]["ped1d"] = bookDQMHistogram1D("pedestal1d_"+tempname,"pedestal for "+tempname,256,0.,256.);
  bookkeeper_[detid]["ped2d"] = bookDQMHistoPlaquetteSummary2D("pedestal2d_"+tempname,"pedestal for "+tempname,detid);

}
//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelGainCalibrationAnalysis);
