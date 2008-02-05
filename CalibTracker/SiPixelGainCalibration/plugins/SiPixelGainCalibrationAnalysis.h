// -*- C++ -*-
//
// Package:    SiPixelGainCalibrationAnalysis
// Class:      SiPixelGainCalibrationAnalysis
// 
/**\class SiPixelGainCalibrationAnalysis SiPixelGainCalibrationAnalysis.h CalibTracker/SiPixelGainCalibrationAnalysis/interface/SiPixelGainCalibrationAnalysis.h

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Wed Nov 14 15:02:06 CET 2007
// $Id: SiPixelGainCalibrationAnalysis.h,v 1.7 2008/01/29 23:56:48 fblekman Exp $
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
#include "CalibFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "TLinearFitter.h"

//
// class decleration
//

class SiPixelGainCalibrationAnalysis : public SiPixelOfflineCalibAnalysisBase {
public:
  explicit SiPixelGainCalibrationAnalysis(const edm::ParameterSet& iConfig);
  ~SiPixelGainCalibrationAnalysis();

  void doSetup(const edm::ParameterSet&);
  virtual bool doFits(uint32_t detid, std::vector<SiPixelCalibDigi>::const_iterator ipix);

  virtual bool checkCorrectCalibrationType();

private:
      
  virtual void calibrationSetup(const edm::EventSetup& iSetup);
      
  virtual void calibrationEnd();
  virtual void newDetID(uint32_t detid);
  void fillDatabase();
  void printSummary();
  std::vector<float> CalculateAveragePerColumn(uint32_t detid, std::string label);
  // ----------member data --------------------------- 
  edm::ParameterSet conf_;
  // more class members used to keep track of the histograms
  std::map<uint32_t,std::map<std::string, MonitorElement *> > bookkeeper_;
  std::map<uint32_t,std::map<std::string, MonitorElement *> > bookkeeper_pixels_;

  // fitter
  int nfitparameters_;
  std::string fitfunction_;

  // flags
  bool reject_badpoints_;
  bool savePixelHists_;
  bool reject_plateaupoints_;
  bool reject_single_entries_;
  double reject_badpoints_frac_;
  double chi2Threshold_;
  double chi2ProbThreshold_;
  double maxGainInHist_;
  double maxChi2InHist_;
  bool saveALLHistograms_;
  bool sum_ped_cols_;
  bool sum_gain_cols_;
  bool filldb_;
  
  // parameters for database output  
  std::string  recordName_;
  bool appendMode_;
  SiPixelGainCalibration *theGainCalibrationDbInput_;
  SiPixelGainCalibrationService theGainCalibrationDbInputService_;

};
