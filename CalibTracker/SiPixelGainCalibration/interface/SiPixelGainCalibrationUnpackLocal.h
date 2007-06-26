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
// $Id: SiPixelGainCalibrationUnpackLocal.h,v 1.1 2007/05/20 18:08:09 fblekman Exp $
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

#include "CalibTracker/SiPixelGainCalibration/interface/PixelCalib.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibHists.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelSLinkDataHit.h"
//#include "CondTools/SiPixel/interface/SiPixelGainCalibrationService.h"
#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"

#include <string>
#include <vector>
#include <iostream>
#include "TFile.h"
#include "TF1.h"
//
// class declaration
//

class SiPixelGainCalibrationUnpackLocal : public edm::EDAnalyzer {
 public:
  explicit SiPixelGainCalibrationUnpackLocal(const edm::ParameterSet&);
  ~SiPixelGainCalibrationUnpackLocal();


 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;


  // ----------member data ---------------------------
  
  unsigned int eventno_counter;
  std::string inputfile_;  
  std::string outputfilename_;
  PixelCalib* calib_;
  //for now assume only on fed_id!
  PixelROCGainCalibHists rocgain_[40][24];
  bool rocgainused_[40][24];
  TFile *outputfileformonitoring;
  TH2F* roc_summary_histos_slope[40][24];
  TH2F* roc_summary_histos_intersect[40][24];
  TH1F* roc_summary_histos_slopevals[40][24];
  TH1F* roc_summary_histos_startvals[40][24];

  unsigned int nrowsmax_;
  unsigned int ncolsmax_;
  unsigned int nrocsmax_;
  unsigned int nchannelsmax_;

  unsigned int vcalminfit_;
  unsigned int vcalmaxfit_;
  bool save_everything_;
  bool database_access_;
  

    
};
