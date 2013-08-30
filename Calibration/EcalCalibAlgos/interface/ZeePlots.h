#ifndef ZEEPLOTS_H
#define  ZEEPLOTS_H

// -*- C++ -*-
//
// Package:    ElectronCalibration
// Class:      ElectronCalibration
// 
/**\class ElectronCalibration ElectronCalibration.cc Calibration/EcalCalibAlgos/src/ElectronCalibration.cc

 Description: Perform single electron calibration (tested on TB data only).

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lorenzo AGOSTINO, Radek Ofierzynski
//         Created:  Tue Jul 18 12:17:01 CEST 2006
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Calibration/Tools/interface/HouseholderDecomposition.h"
#include "Calibration/Tools/interface/MinL3Algorithm.h"
#include "Calibration/Tools/interface/CalibrationCluster.h"
#include "Calibration/Tools/interface/CalibElectron.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"


// class declaration
//

class ZeePlots {
   public:
      ZeePlots( const char* );
      ~ZeePlots();

      void openFile();

      void bookEleHistograms();
      void bookEleMCHistograms();
      void bookZHistograms();
      void bookZMCHistograms();
      void bookHLTHistograms();
      void bookEleClassesPlots();

      void fillZMCInfo( const HepMC::GenEvent* );
      void fillEleMCInfo( const HepMC::GenEvent* );
      void fillEleInfo( const reco::GsfElectronCollection* );
      void fillHLTInfo( edm::Handle<edm::TriggerResults> );
      void fillZInfo(std::pair<calib::CalibElectron*,calib::CalibElectron*> myZeeCandidate);
      void fillEleClassesPlots( calib::CalibElectron*);
      
      void writeEleHistograms();
      void writeZHistograms();
      void writeMCEleHistograms();
      void writeMCZHistograms();
      void writeHLTHistograms();
      void writeEleClassesPlots();

 private:

      TFile* file_;
      const char* fileName_;
 
      TH1F*  h1_gen_ZMass_;
      TH1F*  h1_gen_ZRapidity_;
      TH1F*  h1_gen_ZEta_;
      TH1F*  h1_gen_ZPhi_;
      TH1F*  h1_gen_ZPt_;
      
      TH1F* h1_FiredTriggers_;
      TH1F* h1_HLT1Electron_FiredEvents_ ;
      TH1F* h1_HLT2Electron_FiredEvents_;
      TH1F* h1_HLT2ElectronRelaxed_FiredEvents_ ;
      TH1F* h1_HLT1Electron_HLT2Electron_FiredEvents_;
      TH1F* h1_HLT1Electron_HLT2ElectronRelaxed_FiredEvents_;
      TH1F* h1_HLT2Electron_HLT2ElectronRelaxed_FiredEvents_;
      TH1F* h1_HLT1Electron_HLT2Electron_HLT2ElectronRelaxed_FiredEvents_;
      TH1F* h1_HLTVisitedEvents_;

      TH1F* h1_mcEle_Energy_;
      TH1F* h1_mcElePt_;
      TH1F* h1_mcEleEta_;
      TH1F* h1_mcElePhi_;
      
      TH1F* h1_recoEleEnergy_;
      TH1F* h1_recoElePt_;
      TH1F* h1_recoEleEta_;
      TH1F* h1_recoElePhi_;
      TH1F* h1_nEleReco_;


      TH1F* h1_reco_ZEta_;
      TH1F* h1_reco_ZTheta_;
      TH1F* h1_reco_ZRapidity_;
      TH1F* h1_reco_ZPhi_;
      TH1F* h1_reco_ZPt_;      

      TH1F* h1_occupancyVsEtaGold_;
      TH1F* h1_occupancyVsEtaSilver_;
      TH1F* h1_occupancyVsEtaCrack_;
      TH1F* h1_occupancyVsEtaShower_;
      
};
#endif
