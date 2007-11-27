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
// $Id:$
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

//
// class decleration
//

class SiPixelGainCalibrationAnalysis : public SiPixelOfflineCalibAnalysisBase {
   public:
  explicit SiPixelGainCalibrationAnalysis(const edm::ParameterSet& iConfig);
  ~SiPixelGainCalibrationAnalysis();

  void doSetup(const edm::ParameterSet&);
      virtual bool doFits(uint32_t detid, std::vector<SiPixelCalibDigi>::const_iterator ipix);


   private:
      
      virtual void calibrationSetup(const edm::EventSetup& iSetup);
      
  //      void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      virtual void newDetID(short detid);

      // ----------member data --------------------------- 

  // more class members used to keep track of the histograms
  uint32_t elementsize_;
  std::map<uint32_t,std::vector<MonitorElement *> > bookkeeper_;

  // flags
  bool reject_badpoints_;
  double reject_badpoints_frac_;
  
  // parameters for database output  
  edm::ParameterSet conf_;
//   std::string  recordName_;
//   bool appendMode_;
//   SiPixelGainCalibration *theGainCalibrationDbInput_;
//   SiPixelGainCalibrationService theGainCalibrationDbInputService_;

};
