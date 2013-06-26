#ifndef ROC__Analyzer_h
#define ROC__Analyzer_h

// -*- C++ -*-
//
// Package:    SiPixelDQMRocLevelAnalyzer
// Class:      SiPixelDQMRocLevelAnalyzer
// 
/**\class SiPixelDQMRocLevelAnalyzer SiPixelDQMRocLevelAnalyzer.cc DQM/SiPixelDQMRocLevelAnalyzer/src/SiPixelDQMRocLevelAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lukas Wehrli
//         Created:  Thu Sep 30 14:03:02 CEST 2008
// $Id: SiPixelDQMRocLevelAnalyzer.h,v 1.1 2010/08/10 08:57:54 ursl Exp $
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
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <string>
#include "TH1D.h"
#include "TFile.h"
#include "math.h"
//
// class decleration
//

class SiPixelDQMRocLevelAnalyzer : public edm::EDAnalyzer {
   public:
      explicit SiPixelDQMRocLevelAnalyzer(const edm::ParameterSet&);
      ~SiPixelDQMRocLevelAnalyzer();


   private:
      virtual void beginJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      //
      void RocSummary(std::string tagname);
      void RocSumOneModule(int maxr, int maxc, MonitorElement* const &me, std::vector<double> &vecCN, std::vector<double> &vecMean, std::vector<double> &vecSD, int &chipNumber);
      void FillRocLevelHistos(TH1F *hrocdep, TH1F *hdist, std::vector<double> &vecx, std::vector<double> &vecy);



      // ----------member data ---------------------------
      edm::ParameterSet conf_;
      DQMStore * dbe;
      edm::Service<TFileService> fs_;

      std::vector<MonitorElement*> mes;
      bool bRS, fRS, bPixelAlive;

      std::vector<double> vbpixCN; 
      std::vector<double> vbpixM;
      std::vector<double> vbpixSD;
      std::vector<double> vfpixCN;
      std::vector<double> vfpixM;
      std::vector<double> vfpixSD;

      //barrel
      TH1F * bhPixelAlive; 
      TH1F * bhPixelAlive_dist;
      TH1F * bhThresholdMean; 
      TH1F * bhThresholdMean_dist; 
      TH1F * bhThresholdRMS; 
      TH1F * bhThresholdRMS_dist; 
      TH1F * bhNoiseMean; 
      TH1F * bhNoiseMean_dist; 
      TH1F * bhNoiseRMS; 
      TH1F * bhNoiseRMS_dist; 
      TH1F * bhGainMean; 
      TH1F * bhGainMean_dist; 
      TH1F * bhGainRMS; 
      TH1F * bhGainRMS_dist; 
      TH1F * bhPedestalMean; 
      TH1F * bhPedestalMean_dist; 
      TH1F * bhPedestalRMS; 
      TH1F * bhPedestalRMS_dist; 
      //endcap
      TH1F * ehPixelAlive; 
      TH1F * ehPixelAlive_dist; 
      TH1F * ehThresholdMean; 
      TH1F * ehThresholdMean_dist; 
      TH1F * ehThresholdRMS; 
      TH1F * ehThresholdRMS_dist; 
      TH1F * ehNoiseMean; 
      TH1F * ehNoiseMean_dist; 
      TH1F * ehNoiseRMS; 
      TH1F * ehNoiseRMS_dist; 
      TH1F * ehGainMean; 
      TH1F * ehGainMean_dist; 
      TH1F * ehGainRMS; 
      TH1F * ehGainRMS_dist; 
      TH1F * ehPedestalMean; 
      TH1F * ehPedestalMean_dist; 
      TH1F * ehPedestalRMS; 
      TH1F * ehPedestalRMS_dist; 

};


#endif
