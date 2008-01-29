#ifndef CalibTracker__SiPixelGainCalibrationDBAnalysis_H_
#define CalibTracker__SiPixelGainCalibrationDBAnalysis_H_
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
// $Id$
//
//



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelCalib.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalib.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelSLinkDataHit.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "CondTools/SiPixel/interface/SiPixelGainCalibrationService.h"

//
// class decleration
//

class SiPixelGainCalibrationDBAnalysis : public edm::EDAnalyzer {
   public:
      explicit SiPixelGainCalibrationDBAnalysis(const edm::ParameterSet&);
      ~SiPixelGainCalibrationDBAnalysis();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // defined as inline below
      unsigned long long get_next_block(const unsigned char * dataptr,unsigned int i);

      // ----------member data ---------------------------
      // internal class for storing parameters
      class CalParameters {
      public:
	float ped;
	float gain;
      };
      class TempPixelContainer {
      public:
	unsigned int fed_channel;
	unsigned int roc_id;
	unsigned int dcol_id;
	unsigned int pix_id;
	unsigned int adc;
	unsigned int row;
	unsigned int col;
	unsigned int vcal;
      };
      PixelCalib *calib_; // keeps track of the calibration constants
      PixelSLinkDataHit *hitworker_; // does unpacking of raw data locally
      unsigned int datasize_;//worker that keeps track of raw data
      unsigned int datacounter_;

      unsigned int eventno_counter_;
      unsigned int maxNfedIds_;
      std::string inputconfigfile_;
      std::string rootoutputfilename_;
      void fill(const TempPixelContainer & aPixel);
      void init(const TempPixelContainer & aPixel);
      // maximum numbers of columns/rows/rocs/channels
      unsigned int nrowsmax_;
      unsigned int ncolsmax_;
      unsigned int nrocsmax_;
      unsigned int nchannelsmax_;
      unsigned int vcal_fitmin_;
      unsigned int vcal_fitmax_;
      unsigned int vcal_fitmax_fixed_;
      double chisq_threshold_;
      double maximum_gain_;
      double maximum_ped_;
      // and the containers
      bool rocgainused_[40][24];// [channel][roc]
      PixelROCGainCalib calib_containers_[40][24];// [channel][roc]
      
      edm::ParameterSet conf_;
      bool appendMode_;
      SiPixelGainCalibration* SiPixelGainCalibration_; // database worker class
      SiPixelGainCalibrationService SiPixelGainCalibrationService_; // additional database worker classes
      std::string recordName_;
      //
/*       double thefancyfitfunction(double *x, double *par);// 0: pedestal; 1:plateau; 2:halfwaypoint; 3:gain */
};

inline unsigned long long SiPixelGainCalibrationDBAnalysis::get_next_block(const unsigned char * dataptr,unsigned int i){
  return ((const unsigned long long*)dataptr)[i];
}

// fancy fit function
/* inline Double_t SiPixelGainCalibrationDBAnalysis::thefancyfitfunction(Double_t *x, Double_t *par){// 0: pedestal; 1:plateau; 2:halfwaypoint; 3:gain */
/*   Double_t ped = par[0]; */
/*   Double_t plateau = par[1]; */
/*   Double_t halfwaypoint = par[2]; */
/*   Double_t gain = par[3]; */
  
/*   Double_t res = ped; */
/*   Double_t turnon = 1+TMath::Erf((x[0]-halfwaypoint)/(gain*sqrt(x[0]))); */
/*   turnon*=0.5*plateau; */
  
/*   return res+turnon; */
/* } */
#endif
