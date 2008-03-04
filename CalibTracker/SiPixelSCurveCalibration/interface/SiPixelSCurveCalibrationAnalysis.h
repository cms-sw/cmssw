#ifndef CALIBTRACKER_SIPIXELSCURVECALIBRATION_SIPIXELSCURVE_CALIBRATION_H
#define CALIBTRACKER_SIPIXELSCURVECALIBRATION_SIPIXELSCURVE_CALIBRATION_H
//
// Package:    SiPixelSCurveCalibrationAnalysis
// Class:      SiPixelSCurveCalibrationAnalysis
// 
/**\class SiPixelSCurveCalibrationAnalysis SiPixelSCurveCalibrationAnalysis.cc CalibTracker/SiPixelSCurveCalibration/src/SiPixelSCurveCalibrationAnalysis.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Evan Klose Friis
//         Created:  Tue Nov 13 13:59:09 CET 2007
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "CalibTracker/SiPixelTools/interface/SiPixelOfflineCalibAnalysisBase.h"
#include "TMinuit.h"
#include <iomanip>
//
enum sCurveHistogramType {
   kSigmaSummary,       //1d
   kSigmas,             //2d
   kThresholdSummary,   //1d
   kThresholds,         //2d
   kChi2Summary,        //1d
   kChi2s,              //2d
   kFitResults,         //2d
   kFitResultSummary   //1d
};

enum sCurveErrorFlag {
   errNoDigi,
   errOK,
   errFlaggedBadByUser,
   errBadChi2Prob,
   errNoTurnOn,
   errAllZeros,
   errFitNonPhysical
};


typedef std::map<sCurveHistogramType, MonitorElement*> sCurveHistogramHolder;
typedef std::map<uint32_t, sCurveHistogramHolder> detIDHistogramMap;

// class decleration
//

class SiPixelSCurveCalibrationAnalysis : public SiPixelOfflineCalibAnalysisBase {
   public:
      explicit SiPixelSCurveCalibrationAnalysis(const edm::ParameterSet& iConfig):SiPixelOfflineCalibAnalysisBase(iConfig){doSetup(iConfig);};
      ~SiPixelSCurveCalibrationAnalysis();
      void doSetup(const edm::ParameterSet&);

      virtual bool doFits(uint32_t detid, std::vector<SiPixelCalibDigi>::const_iterator ipix);

      static std::vector<float> efficiencies_;
      static std::vector<float> effErrors_;

      sCurveErrorFlag estimateSCurveParameters(const std::vector<float>& eff, float& threshold, float& sigma);
      sCurveErrorFlag fittedSCurveSanityCheck(float threshold, float sigma, float amplitude);

   private:
      //configuration options
      bool                      saveCurvesThatFlaggedBad_;
      bool                      write2dHistograms_;
      bool                      write2dFitResult_; 
      std::vector<std::string>  plaquettesToSave_;

      //parameters that define "bad curves"
      double                     minimumChi2prob_;
      double                     minimumThreshold_;
      double                     maximumThreshold_;
      double                     minimumSigma_;
      double                     maximumSigma_;
      double                     minimumEffAsymptote_;
      double                     maximumEffAsymptote_;
      
      //holds histogrms entered
      detIDHistogramMap histograms_;

      virtual void calibrationSetup(const edm::EventSetup& iSetup);
      virtual void newDetID(uint32_t detid);
      //virtual void endJob();  //do nothing


      // ----------member data ---------------------------
};

#endif
