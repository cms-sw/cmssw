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
// $Id: SiPixelSCurveCalibrationAnalysis.h,v 1.18 2008/08/29 14:57:27 fblekman Exp $
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
   errNoDigi,                   //default value (will actually never get passed to the analyzer, but included for consistency when viewing histograms) 
   errOK,                       //everything is OK
   errFlaggedBadByUser,         //fit converged, but parameters are outside a user specified range (i.e. noise (sigma) > 6 ADC counts)
   errBadChi2Prob,              //fit converged, but failed user specified chi2 test
   errFitNonPhysical,           //fit converged, but in a nonsensical region (i.e. vCalMax < threshold < 0, sigma > vCalMax, etc)
   errNoTurnOn,                 //sCurve never turned on above 90%
   errAllZeros                  //sCurve was all zeros.  This shouldn't ever happen, (all zeros would prevent a CalibDigi from being produced) but is included as a potential tool for potential future debugging        
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
      std::vector<float> vCalPointsAsFloats_;  //need to save single histograms

      sCurveErrorFlag estimateSCurveParameters(const std::vector<float>& eff, float& threshold, float& sigma);
      sCurveErrorFlag fittedSCurveSanityCheck(float threshold, float sigma, float amplitude);

      void            buildACurveHistogram(const uint32_t& detid, const uint32_t& row, 
                                           const uint32_t& col, sCurveErrorFlag errorFlag, 
                                           const std::vector<float>& efficiencies, const std::vector<float>& errors);

   private:
      //configuration options
      bool                      useDetectorHierarchyFolders_;
      bool                      saveCurvesThatFlaggedBad_;
      unsigned int              maxCurvesToSave_;               //define maximum number of curves to save, to prevent huge memory consumption
      unsigned int              curvesSavedCounter_;
      bool                      write2dHistograms_;
      bool                      write2dFitResult_; 
      std::vector<std::string>  plaquettesToSave_;
      bool                      printoutthresholds_;
      bool                      writeZeroes_;
      std::string               thresholdfilename_;
      std::map<uint32_t, bool>  detIDsToSave_;      


      //parameters that define "bad curves"
      double                     minimumChi2prob_;
      double                     minimumThreshold_;
      double                     maximumThreshold_;
      double                     minimumSigma_;
      double                     maximumSigma_;
      double                     minimumEffAsymptote_;
      double                     maximumEffAsymptote_;

      //parameters that define histogram size/binning
      double                     maximumThresholdBin_;
      double                     maximumSigmaBin_;

      
      //holds histogrms entered
      detIDHistogramMap histograms_;

      virtual void calibrationSetup(const edm::EventSetup& iSetup);
      virtual bool checkCorrectCalibrationType();
      virtual void newDetID(uint32_t detid);
      void makeThresholdSummary(void);
      virtual void calibrationEnd() ;
      
      // ----------member data ---------------------------
};

#endif
