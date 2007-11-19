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
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CalibTracker/SiPixelTools/interface/SiPixelOfflineCalibAnalysisBase.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// class decleration
//

class SiPixelGainCalibrationAnalysis : public SiPixelOfflineCalibAnalysisBase {
   public:
      explicit SiPixelGainCalibrationAnalysis(const edm::ParameterSet&);
      ~SiPixelGainCalibrationAnalysis();

      virtual bool doFits(uint32_t detid, std::vector<SiPixelCalibDigi>::const_iterator ipix);


   private:
      
      virtual void calibrationSetup(const edm::EventSetup& iSetup);
      
  //      void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      virtual void newDetID(short detid);

      // ----------member data --------------------------- 

  // more class members used to keep track of the histograms
  std::map<uint32_t,std::vector<MonitorElement *> > bookkeeper_;
 //  std::map<uint32_t,MonitorElement *> bookkeeper_gain1d_; 
//   std::map<uint32_t,MonitorElement *> bookkeeper_gain2d_; 
//   std::map<uint32_t,MonitorElement *> bookkeeper_ped1d_; 
//   std::map<uint32_t,MonitorElement *> bookkeeper_ped2d_; 

  // flags
  bool reject_badpoints_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelGainCalibrationAnalysis::SiPixelGainCalibrationAnalysis(const edm::ParameterSet& iConfig):
  reject_badpoints_(iConfig.getUntrackedParameter<bool>("suppressZeroAndPlateausInFit",true))
{
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
SiPixelGainCalibrationAnalysis::endJob() {

  // this is where we loop over all 

}

// ------------ method called to do fits to all objects available  ------------
bool
SiPixelGainCalibrationAnalysis::doFits(uint32_t detid, std::vector<SiPixelCalibDigi>::const_iterator ipix)
{
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
    if(ipix->getsum(ipoint)==0 && reject_badpoints_)
      continue;
    if(ipix->getsum(ipoint)>=250*ipix->getnentries(ipoint) && reject_badpoints_)
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
  bookkeeper_[detid][0]->Fill(slope);
  bookkeeper_[detid][1]->Fill(ipix->col(),ipix->row(),slope);
  bookkeeper_[detid][2]->Fill(intercept);
  bookkeeper_[detid][3]->Fill(ipix->col(),ipix->row(),intercept);
  return true;
}
// ------------ method called to do fill new detids  ------------
void 
SiPixelGainCalibrationAnalysis::newDetID(short detid)
{
  setDQMDirectory(detid);
  std::string tempname=translateDetIdToString(detid);
  std::vector<MonitorElement *> entries(4);// hard-code the number of calibrations.
  bookkeeper_[detid]=entries;
  bookkeeper_[detid][0] = bookDQMHistogram1D("gain1d_"+tempname,"gain for "+tempname,100,0.,100.);
  bookkeeper_[detid][1] = bookDQMHistoPlaquetteSummary2D("gain2d_"+tempname,"gain for "+tempname,detid);
  bookkeeper_[detid][2] = bookDQMHistogram1D("pedestal1d_"+tempname,"pedestal for "+tempname,256,0.,256.);
  bookkeeper_[detid][3] = bookDQMHistoPlaquetteSummary2D("pedestal2d_"+tempname,"pedestal for "+tempname,detid);

}
//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelGainCalibrationAnalysis);
