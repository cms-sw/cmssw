// -*- C++ -*-
//
// Package:    TestGEMSegmentAnalyzer
// Class:      TestGEMSegmentAnalyzer
// 
/**\class TestGEMSegmentAnalyzer TestGEMSegmentAnalyzer.cc MyAnalyzers/TestGEMSegmentAnalyzer/src/TestGEMSegmentAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/

// system include files
#include <memory>
#include <fstream>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>


// root include files
#include "TFile.h"
#include "TDirectoryFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TApplication.h"

// user include files
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/GEMRecHit/interface/GEMSegmentCollection.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHitCollection.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

//
// class declaration
//

class TestGEMSegmentAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TestGEMSegmentAnalyzer(const edm::ParameterSet&);
      ~TestGEMSegmentAnalyzer();



   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&);

      // ----------member data ---------------------------
  edm::ESHandle<GEMGeometry> gemGeom;
  edm::ESHandle<CSCGeometry> cscGeom;

  edm::EDGetTokenT<reco::GenParticleCollection> GENParticle_Token;
  edm::EDGetTokenT<edm::HepMCProduct>           HEPMCCol_Token;
  edm::EDGetTokenT<edm::SimTrackContainer>      SIMTrack_Token;
  // edm::EDGetTokenT<edm::PSimHitContainer>    SIMHit_Token;
  edm::EDGetTokenT<CSCSegmentCollection>        CSCSegment_Token;
  edm::EDGetTokenT<GEMSegmentCollection>        GEMSegment_Token;
  edm::EDGetTokenT<GEMRecHitCollection>         GEMRecHit_Token;

  std::string rootFileName;
  std::unique_ptr<TFile> outputfile;

  bool printSegmntInfo, printResidlInfo, printSimHitInfo, printEventOrder;


  std::unique_ptr<TDirectoryFile> GEMSegment_GE11_Pos_and_Dir, GEMSegment_GE21_Pos_and_Dir;
  std::unique_ptr<TDirectoryFile> GEMSegment_GE11, GEMSegment_GE21;
  std::unique_ptr<TDirectoryFile> GEMSegment_GE21_2hits, GEMSegment_GE21_4hits;
  std::unique_ptr<TDirectoryFile> GEMSegment_SimHitPlots, GEMSegment_NoiseReductionPlots, GEMSegment_PositionPlots;

  std::unique_ptr<TH1F> GEN_eta, GEN_phi, SIM_eta, SIM_phi;

  std::unique_ptr<TH1F> GE11_Pos_eta, GE11_Pos_phi, GE11_Dir_eta, GE11_Dir_phi, GE21_Pos_eta, GE21_Pos_phi, GE21_Dir_eta, GE21_Dir_phi;
  std::unique_ptr<TH1F> ME11_Pos_eta, ME11_Pos_phi, ME11_Dir_eta, ME11_Dir_phi, ME21_Pos_eta, ME21_Pos_phi, ME21_Dir_eta, ME21_Dir_phi;
  std::unique_ptr<TH1F> GE11_LocPos_x, GE11_LocPos_y, GE11_GloPos_x, GE11_GloPos_y, GE11_GloPos_r, GE11_GloPos_p, GE11_GloPos_t;
  std::unique_ptr<TH1F> GE21_LocPos_x, GE21_LocPos_y, GE21_GloPos_x, GE21_GloPos_y, GE21_GloPos_r, GE21_GloPos_p, GE21_GloPos_t;

  std::unique_ptr<TH1F> Delta_Pos_SIM_GE11_eta, Delta_Pos_SIM_GE11_phi, Delta_Pos_SIM_GE11_BX;
  std::unique_ptr<TH1F> Delta_Pos_SIM_GE21_eta, Delta_Pos_SIM_GE21_phi, Delta_Pos_SIM_GE21_BX;

  std::unique_ptr<TH1F> Delta_Pos_ME11_GE11_eta, Delta_Pos_ME11_GE11_phi, Delta_Dir_ME11_GE11_eta, Delta_Dir_ME11_GE11_phi, Delta_Dir_ME21_GE21_eta_2hits, Delta_Dir_ME21_GE21_phi_2hits;
  std::unique_ptr<TH1F> Delta_Pos_ME21_GE21_eta, Delta_Pos_ME21_GE21_phi, Delta_Dir_ME21_GE21_eta, Delta_Dir_ME21_GE21_phi, Delta_Dir_ME21_GE21_eta_4hits, Delta_Dir_ME21_GE21_phi_4hits;

  std::unique_ptr<TH1F> GE11_Dir_eta_2hits, GE11_Dir_phi_2hits;
  std::unique_ptr<TH1F> GE21_Pos_eta_2hits, GE21_Pos_phi_2hits, GE21_Dir_eta_2hits, GE21_Dir_phi_2hits;
  std::unique_ptr<TH1F> GE21_Pos_eta_4hits, GE21_Pos_phi_4hits, GE21_Dir_eta_4hits, GE21_Dir_phi_4hits;

  std::unique_ptr<TH1F> NumSegs_GE11_pos, NumSegs_GE11_neg, NumSegs_GE21_pos, NumSegs_GE21_neg;
  std::unique_ptr<TH1F> NumSegs_ME11_pos, NumSegs_ME11_neg, NumSegs_ME21_pos, NumSegs_ME21_neg;

  std::unique_ptr<TH1F> GE11_numhits, GE21_numhits, GE11_BX, GE21_BX;

  std::unique_ptr<TH1F> GE11_fitchi2, GE11_fitndof, GE11_fitchi2ndof;
  std::unique_ptr<TH1F> GE11_Residuals_x, GE11_Residuals_l1_x, GE11_Residuals_l2_x, GE11_Pull_x, GE11_Pull_l1_x, GE11_Pull_l2_x;
  std::unique_ptr<TH1F> GE11_Residuals_y, GE11_Residuals_l1_y, GE11_Residuals_l2_y, GE11_Pull_y, GE11_Pull_l1_y, GE11_Pull_l2_y;

  std::unique_ptr<TH1F> GE21_fitchi2, GE21_fitndof, GE21_fitchi2ndof;
  std::unique_ptr<TH1F> GE21_Residuals_x, GE21_Residuals_l1_x, GE21_Residuals_l2_x, GE21_Residuals_l3_x, GE21_Residuals_l4_x;
  std::unique_ptr<TH1F> GE21_Pull_x, GE21_Pull_l1_x, GE21_Pull_l2_x, GE21_Pull_l3_x, GE21_Pull_l4_x;
  std::unique_ptr<TH1F> GE21_Residuals_y, GE21_Residuals_l1_y, GE21_Residuals_l2_y, GE21_Residuals_l3_y, GE21_Residuals_l4_y;
  std::unique_ptr<TH1F> GE21_Pull_y, GE21_Pull_l1_y, GE21_Pull_l2_y, GE21_Pull_l3_y, GE21_Pull_l4_y;

  std::unique_ptr<TH1F> GE21_2hits_fitchi2, GE21_2hits_fitndof, GE21_2hits_fitchi2ndof;
  std::unique_ptr<TH1F> GE21_2hits_Residuals_x, GE21_2hits_Residuals_l1_x, GE21_2hits_Residuals_l2_x, GE21_2hits_Pull_x, GE21_2hits_Pull_l1_x, GE21_2hits_Pull_l2_x;
  std::unique_ptr<TH1F> GE21_2hits_Residuals_y, GE21_2hits_Residuals_l1_y, GE21_2hits_Residuals_l2_y, GE21_2hits_Pull_y, GE21_2hits_Pull_l1_y, GE21_2hits_Pull_l2_y;

  std::unique_ptr<TH1F> GE21_4hits_fitchi2, GE21_4hits_fitndof, GE21_4hits_fitchi2ndof;
  std::unique_ptr<TH1F> GE21_4hits_Residuals_x, GE21_4hits_Residuals_l1_x, GE21_4hits_Residuals_l2_x, GE21_4hits_Residuals_l3_x, GE21_4hits_Residuals_l4_x;
  std::unique_ptr<TH1F> GE21_4hits_Pull_x, GE21_4hits_Pull_l1_x, GE21_4hits_Pull_l2_x, GE21_4hits_Pull_l3_x, GE21_4hits_Pull_l4_x;
  std::unique_ptr<TH1F> GE21_4hits_Residuals_y, GE21_4hits_Residuals_l1_y, GE21_4hits_Residuals_l2_y, GE21_4hits_Residuals_l3_y, GE21_4hits_Residuals_l4_y;
  std::unique_ptr<TH1F> GE21_4hits_Pull_y, GE21_4hits_Pull_l1_y, GE21_4hits_Pull_l2_y, GE21_4hits_Pull_l3_y, GE21_4hits_Pull_l4_y;

  std::unique_ptr<TH2F> GE11_Pos_Dir_eta, GE11_Pos_Dir_phi, GE21_Pos_Dir_eta, GE21_Pos_Dir_phi, GE21_Pos_Dir_eta_2hits, GE21_Pos_Dir_phi_2hits, GE21_Pos_Dir_eta_4hits, GE21_Pos_Dir_phi_4hits;
  std::unique_ptr<TH2F> ME11_Pos_Dir_eta, ME11_Pos_Dir_phi, ME21_Pos_Dir_eta, ME21_Pos_Dir_phi;

  /*
  // Aim: prove that construction of a segment reduces the background level in GEM system
  // Make a plot of # signal rechits / # total rechits
  // Make a plot of # signal segments / # total segments
  // maybe use DigiSimLinks or Delta R matching with GEN or SIM  
  std::unique_ptr<TH1F> GE11_RecHits_Pos, GE11_RecHits_Neg, GE21_RecHits_Pos, GE21_RecHits_Neg,
  std::unique_ptr<TH1F> GE11_RecHBX0_Pos, GE11_RecHBX0_Neg, GE21_RecHBX0_Pos, GE21_RecHBX0_Neg,
  std::unique_ptr<TH1F> GE11_AllSegm_Pos, GE11_AllSegm_Neg, GE21_AllSegm_Pos, GE21_AllSegm_Neg,
  std::unique_ptr<TH1F> GE11_SegmBX0_Pos, GE11_SegmBX0_Neg, GE21_SegmBX0_Pos, GE21_SegmBX0_Neg,
  */

  /*
  std::unique_ptr<TH1F> GE11_RecHits_PerEvent,      GE21_RecHits_PerEvent;
  std::unique_ptr<TH1F> GE11_RecHits_BX0_PerEvent,  GE21_RecHits_BX0_PerEvent;
  std::unique_ptr<TH1F> GE11_Segments_PerEvent,     GE21_Segments_PerEvent;
  std::unique_ptr<TH1F> GE11_Segments_BX0_PerEvent, GE21_Segments_BX0_PerEvent;
  */

  std::unique_ptr<TH2F> GE11_Pos_XY, GE11_Neg_XY, GE21_Pos_XY, GE21_Neg_XY, GEM_Pos_RZ, GEM_Neg_RZ;
  std::unique_ptr<TH1F> GE11_CheckSegmentDirection, GE21_CheckSegmentDirection, GE21_CheckSegmentDirection_2hits, GE21_CheckSegmentDirection_4hits;

  std::unique_ptr<TH1F> SIM_SimHitEta;
  std::unique_ptr<TH1F> GE11_SimHitEta,    GE21_SimHitEta;
  std::unique_ptr<TH1F> GE11_SimHitEta_1D, GE21_SimHitEta_1D;

  std::unique_ptr<TH1F> GEM_AverageRecHitsPerEvent, GEM_AverageSimHitsPerEvent, GEM_NoiseFraction, GEM_AverageRecHitsPerEvent_BX0, GEM_NoiseFraction_BX0;
  int nEvents; std::vector<int> nGEMrh, nGEMsh, nGEMrhbx0;

};

//
// constants, enums and typedefs
//
// constructors and destructor
//
TestGEMSegmentAnalyzer::TestGEMSegmentAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  GENParticle_Token = consumes<reco::GenParticleCollection>(edm::InputTag("genParticles"));
  HEPMCCol_Token    = consumes<edm::HepMCProduct>(edm::InputTag("generator"));
  SIMTrack_Token    = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  consumesMany<edm::PSimHitContainer>();
  // SIMHit_Token     = consumes<edm::PSimHitContainer>(edm::InputTag("g4SimHits"));
  CSCSegment_Token  = consumes<CSCSegmentCollection>(edm::InputTag("cscSegments"));
  GEMSegment_Token  = consumes<GEMSegmentCollection>(edm::InputTag("gemSegments"));
  GEMRecHit_Token   = consumes<GEMRecHitCollection>(edm::InputTag("gemRecHits"));

  printSegmntInfo = iConfig.getUntrackedParameter<bool>("printSegmntInfo");
  printResidlInfo = iConfig.getUntrackedParameter<bool>("printResidlInfo");
  printSimHitInfo = iConfig.getUntrackedParameter<bool>("printSimHitInfo");
  printEventOrder = iConfig.getUntrackedParameter<bool>("printEventOrder");



  rootFileName  = iConfig.getUntrackedParameter<std::string>("RootFileName");
  outputfile.reset(TFile::Open(rootFileName.c_str(), "RECREATE"));

  GEMSegment_GE11_Pos_and_Dir    = std::unique_ptr<TDirectoryFile>(new TDirectoryFile("GEMSegment_GE11_Pos_and_Dir", "GEMSegment_GE11_Pos_and_Dir"));
  GEMSegment_GE21_Pos_and_Dir    = std::unique_ptr<TDirectoryFile>(new TDirectoryFile("GEMSegment_GE21_Pos_and_Dir", "GEMSegment_GE21_Pos_and_Dir"));
  GEMSegment_GE11                = std::unique_ptr<TDirectoryFile>(new TDirectoryFile("GEMSegment_GE11", "GEMSegment_GE11"));
  GEMSegment_GE21                = std::unique_ptr<TDirectoryFile>(new TDirectoryFile("GEMSegment_GE21", "GEMSegment_GE21"));
  GEMSegment_GE21_2hits          = std::unique_ptr<TDirectoryFile>(new TDirectoryFile("GEMSegment_GE21_2hits", "GEMSegment_GE21_2hits"));
  GEMSegment_GE21_4hits          = std::unique_ptr<TDirectoryFile>(new TDirectoryFile("GEMSegment_GE21_4hits", "GEMSegment_GE21_4hits"));
  GEMSegment_SimHitPlots         = std::unique_ptr<TDirectoryFile>(new TDirectoryFile("GEMSegment_SimHitPlots", "GEMSegment_SimHitPlots"));
  GEMSegment_NoiseReductionPlots = std::unique_ptr<TDirectoryFile>(new TDirectoryFile("GEMSegment_NoiseReductionPlots", "GEMSegment_NoiseReductionPlots"));
  GEMSegment_PositionPlots       = std::unique_ptr<TDirectoryFile>(new TDirectoryFile("GEMSegment_PositionPlots", "GEMSegment_PositionPlots"));


  GEN_eta = std::unique_ptr<TH1F>(new TH1F("GEN_eta","GEN_eta",100,-2.50,2.50));
  GEN_phi = std::unique_ptr<TH1F>(new TH1F("GEN_phi","GEN_phi",144,-3.14,3.14));
  SIM_eta = std::unique_ptr<TH1F>(new TH1F("SIM_eta","SIM_eta",100,-2.50,2.50));
  SIM_phi = std::unique_ptr<TH1F>(new TH1F("SIM_phi","SIM_phi",144,-3.14,3.14));

  GE11_Pos_eta = std::unique_ptr<TH1F>(new TH1F("GE11_Pos_eta","GE11_Pos_eta",100,-2.50,2.50));
  GE11_Pos_phi = std::unique_ptr<TH1F>(new TH1F("GE11_Pos_phi","GE11_Pos_phi",144,-3.14,3.14));
  GE11_Dir_eta = std::unique_ptr<TH1F>(new TH1F("GE11_Dir_eta","GE11_Dir_eta",100,-2.50,2.50));
  GE11_Dir_phi = std::unique_ptr<TH1F>(new TH1F("GE11_Dir_phi","GE11_Dir_phi",144,-3.14,3.14));

  GE21_Pos_eta = std::unique_ptr<TH1F>(new TH1F("GE21_Pos_eta","GE21_Pos_eta",100,-2.50,2.50));
  GE21_Pos_phi = std::unique_ptr<TH1F>(new TH1F("GE21_Pos_phi","GE21_Pos_phi",144,-3.14,3.14));
  GE21_Dir_eta = std::unique_ptr<TH1F>(new TH1F("GE21_Dir_eta","GE21_Dir_eta",100,-2.50,2.50));
  GE21_Dir_phi = std::unique_ptr<TH1F>(new TH1F("GE21_Dir_phi","GE21_Dir_phi",144,-3.14,3.14));

  ME11_Pos_eta = std::unique_ptr<TH1F>(new TH1F("ME11_Pos_eta","ME11_Pos_eta",100,-2.50,2.50));
  ME11_Pos_phi = std::unique_ptr<TH1F>(new TH1F("ME11_Pos_phi","ME11_Pos_phi",144,-3.14,3.14));
  ME11_Dir_eta = std::unique_ptr<TH1F>(new TH1F("ME11_Dir_eta","ME11_Dir_eta",100,-2.50,2.50));
  ME11_Dir_phi = std::unique_ptr<TH1F>(new TH1F("ME11_Dir_phi","ME11_Dir_phi",144,-3.14,3.14));

  ME21_Pos_eta = std::unique_ptr<TH1F>(new TH1F("ME21_Pos_eta","ME21_Pos_eta",100,-2.50,2.50));
  ME21_Pos_phi = std::unique_ptr<TH1F>(new TH1F("ME21_Pos_phi","ME21_Pos_phi",144,-3.14,3.14));
  ME21_Dir_eta = std::unique_ptr<TH1F>(new TH1F("ME21_Dir_eta","ME21_Dir_eta",100,-2.50,2.50));
  ME21_Dir_phi = std::unique_ptr<TH1F>(new TH1F("ME21_Dir_phi","ME21_Dir_phi",144,-3.14,3.14));

  GE11_Dir_eta_2hits = std::unique_ptr<TH1F>(new TH1F("GE11_Dir_eta_2hits","GE11_Dir_eta_2hits",200,-50,50));
  GE11_Dir_phi_2hits = std::unique_ptr<TH1F>(new TH1F("GE11_Dir_phi_2hits","GE11_Dir_phi_2hits",144,-3.14,3.14));

  GE21_Pos_eta_2hits = std::unique_ptr<TH1F>(new TH1F("GE21_Pos_eta_2hits","GE21_Pos_eta_2hits",100,-2.50,2.50));
  GE21_Pos_phi_2hits = std::unique_ptr<TH1F>(new TH1F("GE21_Pos_phi_2hits","GE21_Pos_phi_2hits",144,-3.14,3.14));
  GE21_Dir_eta_2hits = std::unique_ptr<TH1F>(new TH1F("GE21_Dir_eta_2hits","GE21_Dir_eta_2hits",200,-50,50));
  GE21_Dir_phi_2hits = std::unique_ptr<TH1F>(new TH1F("GE21_Dir_phi_2hits","GE21_Dir_phi_2hits",144,-3.14,3.14));

  GE21_Pos_eta_4hits = std::unique_ptr<TH1F>(new TH1F("GE21_Pos_eta_4hits","GE21_Pos_eta_4hits",100,-2.50,2.50));
  GE21_Pos_phi_4hits = std::unique_ptr<TH1F>(new TH1F("GE21_Pos_phi_4hits","GE21_Pos_phi_4hits",144,-3.14,3.14));
  GE21_Dir_eta_4hits = std::unique_ptr<TH1F>(new TH1F("GE21_Dir_eta_4hits","GE21_Dir_eta_4hits",200,-50,50));
  GE21_Dir_phi_4hits = std::unique_ptr<TH1F>(new TH1F("GE21_Dir_phi_4hits","GE21_Dir_phi_4hits",144,-3.14,3.14));

  Delta_Pos_SIM_GE11_eta = std::unique_ptr<TH1F>(new TH1F("Delta_Pos_SIM_GE11_eta","Delta_Pos_SIM_GE11_eta",100,-0.5,0.5));
  Delta_Pos_SIM_GE11_phi = std::unique_ptr<TH1F>(new TH1F("Delta_Pos_SIM_GE11_phi","Delta_Pos_SIM_GE11_phi",100,-0.5,0.5));
  Delta_Pos_SIM_GE21_eta = std::unique_ptr<TH1F>(new TH1F("Delta_Pos_SIM_GE21_eta","Delta_Pos_SIM_GE21_eta",100,-0.5,0.5));
  Delta_Pos_SIM_GE21_phi = std::unique_ptr<TH1F>(new TH1F("Delta_Pos_SIM_GE21_phi","Delta_Pos_SIM_GE21_phi",100,-0.5,0.5));

  Delta_Pos_ME11_GE11_eta = std::unique_ptr<TH1F>(new TH1F("Delta_Pos_ME11_GE11_eta","Delta_Pos_ME11_GE11_eta",100,-0.5,0.5));
  Delta_Pos_ME11_GE11_phi = std::unique_ptr<TH1F>(new TH1F("Delta_Pos_ME11_GE11_phi","Delta_Pos_ME11_GE11_phi",100,-0.5,0.5));
  Delta_Pos_ME21_GE21_eta = std::unique_ptr<TH1F>(new TH1F("Delta_Pos_ME21_GE21_eta","Delta_Pos_ME21_GE21_eta",100,-0.5,0.5));
  Delta_Pos_ME21_GE21_phi = std::unique_ptr<TH1F>(new TH1F("Delta_Pos_ME21_GE21_phi","Delta_Pos_ME21_GE21_phi",100,-0.5,0.5));

  Delta_Dir_ME11_GE11_eta = std::unique_ptr<TH1F>(new TH1F("Delta_Dir_ME11_GE11_eta","Delta_Dir_ME11_GE11_eta",100,-0.5,0.5));
  Delta_Dir_ME11_GE11_phi = std::unique_ptr<TH1F>(new TH1F("Delta_Dir_ME11_GE11_phi","Delta_Dir_ME11_GE11_phi",100,-3.5,3.5));
  Delta_Dir_ME21_GE21_eta = std::unique_ptr<TH1F>(new TH1F("Delta_Dir_ME21_GE21_eta","Delta_Dir_ME21_GE21_eta",100,-0.5,0.5));
  Delta_Dir_ME21_GE21_phi = std::unique_ptr<TH1F>(new TH1F("Delta_Dir_ME21_GE21_phi","Delta_Dir_ME21_GE21_phi",100,-3.5,3.5));

  Delta_Dir_ME21_GE21_eta_2hits = std::unique_ptr<TH1F>(new TH1F("Delta_Dir_ME21_GE21_eta_2hits","Delta_Dir_ME21_GE21_eta_2hits",100,-50,50));
  Delta_Dir_ME21_GE21_phi_2hits = std::unique_ptr<TH1F>(new TH1F("Delta_Dir_ME21_GE21_phi_2hits","Delta_Dir_ME21_GE21_phi_2hits",100,-3.5,3.5));
  Delta_Dir_ME21_GE21_eta_4hits = std::unique_ptr<TH1F>(new TH1F("Delta_Dir_ME21_GE21_eta_4hits","Delta_Dir_ME21_GE21_eta_4hits",100,-0.5,0.5));
  Delta_Dir_ME21_GE21_phi_4hits = std::unique_ptr<TH1F>(new TH1F("Delta_Dir_ME21_GE21_phi_4hits","Delta_Dir_ME21_GE21_phi_4hits",100,-3.5,3.5));

  Delta_Pos_SIM_GE11_BX = std::unique_ptr<TH1F>(new TH1F("Delta_Pos_SIM_GE11_BX","Delta_Pos_SIM_GE11_BX",11,-5.5,5.5));
  Delta_Pos_SIM_GE21_BX = std::unique_ptr<TH1F>(new TH1F("Delta_Pos_SIM_GE21_BX","Delta_Pos_SIM_GE21_BX",11,-5.5,5.5));

  NumSegs_GE11_pos = std::unique_ptr<TH1F>(new TH1F("NumSegs_GE11_pos","NumSegs_GE11_pos",11,-0.5,10.5));
  NumSegs_GE11_neg = std::unique_ptr<TH1F>(new TH1F("NumSegs_GE11_neg","NumSegs_GE11_neg",11,-0.5,10.5));
  NumSegs_GE21_pos = std::unique_ptr<TH1F>(new TH1F("NumSegs_GE21_pos","NumSegs_GE21_pos",11,-0.5,10.5));
  NumSegs_GE21_neg = std::unique_ptr<TH1F>(new TH1F("NumSegs_GE21_neg","NumSegs_GE21_neg",11,-0.5,10.5));

  NumSegs_ME11_pos = std::unique_ptr<TH1F>(new TH1F("NumSegs_ME11_pos","NumSegs_ME11_pos",11,-0.5,10.5));
  NumSegs_ME11_neg = std::unique_ptr<TH1F>(new TH1F("NumSegs_ME11_neg","NumSegs_ME11_neg",11,-0.5,10.5));
  NumSegs_ME21_pos = std::unique_ptr<TH1F>(new TH1F("NumSegs_ME21_pos","NumSegs_ME21_pos",11,-0.5,10.5));
  NumSegs_ME21_neg = std::unique_ptr<TH1F>(new TH1F("NumSegs_ME21_neg","NumSegs_ME21_neg",11,-0.5,10.5));

  GE11_LocPos_x = std::unique_ptr<TH1F>(new TH1F("GE11_LocPos_x","GE11_LocPos_x",100,-50,50));
  GE11_LocPos_y = std::unique_ptr<TH1F>(new TH1F("GE11_LocPos_y","GE11_LocPos_y",100,-50,50));
  GE11_GloPos_x = std::unique_ptr<TH1F>(new TH1F("GE11_GloPos_x","GE11_GloPos_x", 70, 0,350));
  GE11_GloPos_y = std::unique_ptr<TH1F>(new TH1F("GE11_GloPos_y","GE11_GloPos_y", 70, 0,350));
  GE11_GloPos_r = std::unique_ptr<TH1F>(new TH1F("GE11_GloPos_r","GE11_GloPos_r",100, 100,350));
  GE11_GloPos_p = std::unique_ptr<TH1F>(new TH1F("GE11_GloPos_p","GE11_GloPos_p",144,-3.14,3.14));
  GE11_GloPos_t = std::unique_ptr<TH1F>(new TH1F("GE11_GloPos_t","GE11_GloPos_t", 72, 0.00,3.14));

  GE21_LocPos_x = std::unique_ptr<TH1F>(new TH1F("GE21_LocPos_x","GE21_LocPos_x",100,-50,50));
  GE21_LocPos_y = std::unique_ptr<TH1F>(new TH1F("GE21_LocPos_y","GE21_LocPos_y",100,-200,200));
  GE21_GloPos_x = std::unique_ptr<TH1F>(new TH1F("GE21_GloPos_x","GE21_GloPos_x", 70, 0,350));
  GE21_GloPos_y = std::unique_ptr<TH1F>(new TH1F("GE21_GloPos_y","GE21_GloPos_y", 70, 0,350));
  GE21_GloPos_r = std::unique_ptr<TH1F>(new TH1F("GE21_GloPos_r","GE21_GloPos_r",100, 100,350));
  GE21_GloPos_p = std::unique_ptr<TH1F>(new TH1F("GE21_GloPos_p","GE21_GloPos_p",144,-3.14,3.14));
  GE21_GloPos_t = std::unique_ptr<TH1F>(new TH1F("GE21_GloPos_t","GE21_GloPos_t", 72, 0.00,3.14));

  GE11_fitchi2        = std::unique_ptr<TH1F>(new TH1F("GE11_chi2","GE11_chi2",11,-0.5,10.5)); 
  GE11_fitndof        = std::unique_ptr<TH1F>(new TH1F("GE11_ndf","GE11_ndf",11,-0.5,10.5)); 
  GE11_fitchi2ndof    = std::unique_ptr<TH1F>(new TH1F("GE11_chi2Vsndf","GE11_chi2Vsndf",50,0.,5.)); 
  GE11_numhits        = std::unique_ptr<TH1F>(new TH1F("GE11_NumberOfHits","GE11_NumberOfHits",11,-0.5,10.5)); 
  GE11_BX             = std::unique_ptr<TH1F>(new TH1F("GE11_BX","GE11_BX",11,-5.5,5.5));
  GE11_Residuals_x    = std::unique_ptr<TH1F>(new TH1F("xGE11_Res","xGE11_Res",100,-0.5,0.5));
  GE11_Residuals_l1_x = std::unique_ptr<TH1F>(new TH1F("xGE11_Res_l1","xGE11_Res_l1",100,-0.5,0.5));
  GE11_Residuals_l2_x = std::unique_ptr<TH1F>(new TH1F("xGE11_Res_l2","xGE11_Res_l2",100,-0.5,0.5));
  GE11_Pull_x    = std::unique_ptr<TH1F>(new TH1F("xGE11_Pull","xGE11_Pull",100,-5.,5.));
  GE11_Pull_l1_x = std::unique_ptr<TH1F>(new TH1F("xGE11_Pull_l1","xGE11_Pull_l1",100,-5.,5.));
  GE11_Pull_l2_x = std::unique_ptr<TH1F>(new TH1F("xGE11_Pull_l2","xGE11_Pull_l2",100,-5.,5.));
  GE11_Residuals_y    = std::unique_ptr<TH1F>(new TH1F("yGE11_Res","yGE11_Res",100,-5.,5.));
  GE11_Residuals_l1_y = std::unique_ptr<TH1F>(new TH1F("yGE11_Res_l1","yGE11_Res_l1",100,-5.,5.));
  GE11_Residuals_l2_y = std::unique_ptr<TH1F>(new TH1F("yGE11_Res_l2","yGE11_Res_l2",100,-5.,5.));
  GE11_Pull_y    = std::unique_ptr<TH1F>(new TH1F("yGE11_Pull","yGE11_Pull",100,-5.,5.));
  GE11_Pull_l1_y = std::unique_ptr<TH1F>(new TH1F("yGE11_Pull_l1","yGE11_Pull_l1",100,-5.,5.));
  GE11_Pull_l2_y = std::unique_ptr<TH1F>(new TH1F("yGE11_Pull_l2","yGE11_Pull_l2",100,-5.,5.));

  GE21_fitchi2        = std::unique_ptr<TH1F>(new TH1F("GE21_chi2","GE21_chi2",11,0.5,10.5)); 
  GE21_fitndof        = std::unique_ptr<TH1F>(new TH1F("GE21_ndf","GE21_ndf",11,-0.5,10.5)); 
  GE21_fitchi2ndof    = std::unique_ptr<TH1F>(new TH1F("GE21_chi2Vsndf","GE21_chi2Vsndf",50,0.,5.)); 
  GE21_numhits        = std::unique_ptr<TH1F>(new TH1F("GE21_NumberOfHits","GE21_NumberOfHits",11,-0.5,10.5)); 
  GE21_BX             = std::unique_ptr<TH1F>(new TH1F("GE21_BX","GE21_BX",11,-5.5,5.5));
  GE21_Residuals_x    = std::unique_ptr<TH1F>(new TH1F("xGE21_Res","xGE21_Res",100,-0.5,0.5));
  GE21_Residuals_l1_x = std::unique_ptr<TH1F>(new TH1F("xGE21_Res_l1","xGE21_Res_l1",100,-0.5,0.5));
  GE21_Residuals_l2_x = std::unique_ptr<TH1F>(new TH1F("xGE21_Res_l2","xGE21_Res_l2",100,-0.5,0.5));
  GE21_Residuals_l3_x = std::unique_ptr<TH1F>(new TH1F("xGE21_Res_l3","xGE21_Res_l3",100,-0.5,0.5));
  GE21_Residuals_l4_x = std::unique_ptr<TH1F>(new TH1F("xGE21_Res_l4","xGE21_Res_l4",100,-0.5,0.5));
  GE21_Pull_x    = std::unique_ptr<TH1F>(new TH1F("xGE21_Pull","xGE21_Pull",100,-5.,5.));
  GE21_Pull_l1_x = std::unique_ptr<TH1F>(new TH1F("xGE21_Pull_l1","xGE21_Pull_l1",100,-5.,5.));
  GE21_Pull_l2_x = std::unique_ptr<TH1F>(new TH1F("xGE21_Pull_l2","xGE21_Pull_l2",100,-5.,5.));
  GE21_Pull_l3_x = std::unique_ptr<TH1F>(new TH1F("xGE21_Pull_l3","xGE21_Pull_l3",100,-5.,5.));
  GE21_Pull_l4_x = std::unique_ptr<TH1F>(new TH1F("xGE21_Pull_l4","xGE21_Pull_l4",100,-5.,5.));
  GE21_Residuals_y    = std::unique_ptr<TH1F>(new TH1F("yGE21_Res","yGE21_Res",100,-5.,5.));
  GE21_Residuals_l1_y = std::unique_ptr<TH1F>(new TH1F("yGE21_Res_l1","yGE21_Res_l1",100,-5.,5.));
  GE21_Residuals_l2_y = std::unique_ptr<TH1F>(new TH1F("yGE21_Res_l2","yGE21_Res_l2",100,-5.,5.));
  GE21_Residuals_l3_y = std::unique_ptr<TH1F>(new TH1F("yGE21_Res_l3","yGE21_Res_l3",100,-5.,5.));
  GE21_Residuals_l4_y = std::unique_ptr<TH1F>(new TH1F("yGE21_Res_l4","yGE21_Res_l4",100,-5.,5.));
  GE21_Pull_y    = std::unique_ptr<TH1F>(new TH1F("yGE21_Pull","yGE21_Pull",100,-5.,5.));
  GE21_Pull_l1_y = std::unique_ptr<TH1F>(new TH1F("yGE21_Pull_l1","yGE21_Pull_l1",100,-5.,5.));
  GE21_Pull_l2_y = std::unique_ptr<TH1F>(new TH1F("yGE21_Pull_l2","yGE21_Pull_l2",100,-5.,5.));
  GE21_Pull_l3_y = std::unique_ptr<TH1F>(new TH1F("yGE21_Pull_l3","yGE21_Pull_l3",100,-5.,5.));
  GE21_Pull_l4_y = std::unique_ptr<TH1F>(new TH1F("yGE21_Pull_l4","yGE21_Pull_l4",100,-5.,5.));

  GE21_2hits_fitchi2 = std::unique_ptr<TH1F>(new TH1F("GE21_2hits_chi2","GE21_2hits_chi2",11,-0.5,10.5)); 
  GE21_2hits_fitndof = std::unique_ptr<TH1F>(new TH1F("GE21_2hits_ndf","GE21_2hits_ndf",11,-0.5,10.5)); 
  GE21_2hits_fitchi2ndof = std::unique_ptr<TH1F>(new TH1F("GE21_2hits_chi2Vsndf","GE21_2hits_chi2Vsndf",50,0.,5.)); 
  GE21_2hits_Residuals_x    = std::unique_ptr<TH1F>(new TH1F("xGE21_2hits_Res","xGE21_2hits_Res",100,-0.5,0.5));
  GE21_2hits_Residuals_l1_x = std::unique_ptr<TH1F>(new TH1F("xGE21_2hits_Res_l1","xGE21_2hits_Res_l1",100,-0.5,0.5));
  GE21_2hits_Residuals_l2_x = std::unique_ptr<TH1F>(new TH1F("xGE21_2hits_Res_l2","xGE21_2hits_Res_l2",100,-0.5,0.5));
  GE21_2hits_Pull_x    = std::unique_ptr<TH1F>(new TH1F("xGE21_2hits_Pull","xGE21_2hits_Pull",100,-5.,5.));
  GE21_2hits_Pull_l1_x = std::unique_ptr<TH1F>(new TH1F("xGE21_2hits_Pull_l1","xGE21_2hits_Pull_l1",100,-5.,5.));
  GE21_2hits_Pull_l2_x = std::unique_ptr<TH1F>(new TH1F("xGE21_2hits_Pull_l2","xGE21_2hits_Pull_l2",100,-5.,5.));
  GE21_2hits_Residuals_y    = std::unique_ptr<TH1F>(new TH1F("yGE21_2hits_Res","yGE21_2hits_Res",100,-5.,5.));
  GE21_2hits_Residuals_l1_y = std::unique_ptr<TH1F>(new TH1F("yGE21_2hits_Res_l1","yGE21_2hits_Res_l1",100,-5.,5.));
  GE21_2hits_Residuals_l2_y = std::unique_ptr<TH1F>(new TH1F("yGE21_2hits_Res_l2","yGE21_2hits_Res_l2",100,-5.,5.));
  GE21_2hits_Pull_y    = std::unique_ptr<TH1F>(new TH1F("yGE21_2hits_Pull","yGE21_2hits_Pull",100,-5.,5.));
  GE21_2hits_Pull_l1_y = std::unique_ptr<TH1F>(new TH1F("yGE21_2hits_Pull_l1","yGE21_2hits_Pull_l1",100,-5.,5.));
  GE21_2hits_Pull_l2_y = std::unique_ptr<TH1F>(new TH1F("yGE21_2hits_Pull_l2","yGE21_2hits_Pull_l2",100,-5.,5.));

  GE21_4hits_fitchi2 = std::unique_ptr<TH1F>(new TH1F("GE21_4hits_chi2","GE21_4hits_chi2",11,0.5,10.5)); 
  GE21_4hits_fitndof = std::unique_ptr<TH1F>(new TH1F("GE21_4hits_ndf","GE21_4hits_ndf",11,-0.5,10.5)); 
  GE21_4hits_fitchi2ndof = std::unique_ptr<TH1F>(new TH1F("GE21_4hits_chi2Vsndf","GE21_4hits_chi2Vsndf",50,0.,5.)); 
  GE21_4hits_Residuals_x    = std::unique_ptr<TH1F>(new TH1F("xGE21_4hits_Res","xGE21_4hits_Res",100,-0.5,0.5));
  GE21_4hits_Residuals_l1_x = std::unique_ptr<TH1F>(new TH1F("xGE21_4hits_Res_l1","xGE21_4hits_Res_l1",100,-0.5,0.5));
  GE21_4hits_Residuals_l2_x = std::unique_ptr<TH1F>(new TH1F("xGE21_4hits_Res_l2","xGE21_4hits_Res_l2",100,-0.5,0.5));
  GE21_4hits_Residuals_l3_x = std::unique_ptr<TH1F>(new TH1F("xGE21_4hits_Res_l3","xGE21_4hits_Res_l3",100,-0.5,0.5));
  GE21_4hits_Residuals_l4_x = std::unique_ptr<TH1F>(new TH1F("xGE21_4hits_Res_l4","xGE21_4hits_Res_l4",100,-0.5,0.5));
  GE21_4hits_Pull_x    = std::unique_ptr<TH1F>(new TH1F("xGE21_4hits_Pull","xGE21_4hits_Pull",100,-5.,5.));
  GE21_4hits_Pull_l1_x = std::unique_ptr<TH1F>(new TH1F("xGE21_4hits_Pull_l1","xGE21_4hits_Pull_l1",100,-5.,5.));
  GE21_4hits_Pull_l2_x = std::unique_ptr<TH1F>(new TH1F("xGE21_4hits_Pull_l2","xGE21_4hits_Pull_l2",100,-5.,5.));
  GE21_4hits_Pull_l3_x = std::unique_ptr<TH1F>(new TH1F("xGE21_4hits_Pull_l3","xGE21_4hits_Pull_l3",100,-5.,5.));
  GE21_4hits_Pull_l4_x = std::unique_ptr<TH1F>(new TH1F("xGE21_4hits_Pull_l4","xGE21_4hits_Pull_l4",100,-5.,5.));
  GE21_4hits_Residuals_y    = std::unique_ptr<TH1F>(new TH1F("yGE21_4hits_Res","yGE21_4hits_Res",100,-5.,5.));
  GE21_4hits_Residuals_l1_y = std::unique_ptr<TH1F>(new TH1F("yGE21_4hits_Res_l1","yGE21_4hits_Res_l1",100,-5.,5.));
  GE21_4hits_Residuals_l2_y = std::unique_ptr<TH1F>(new TH1F("yGE21_4hits_Res_l2","yGE21_4hits_Res_l2",100,-5.,5.));
  GE21_4hits_Residuals_l3_y = std::unique_ptr<TH1F>(new TH1F("yGE21_4hits_Res_l3","yGE21_4hits_Res_l3",100,-5.,5.));
  GE21_4hits_Residuals_l4_y = std::unique_ptr<TH1F>(new TH1F("yGE21_4hits_Res_l4","yGE21_4hits_Res_l4",100,-5.,5.));
  GE21_4hits_Pull_y    = std::unique_ptr<TH1F>(new TH1F("yGE21_4hits_Pull","yGE21_4hits_Pull",100,-5.,5.));
  GE21_4hits_Pull_l1_y = std::unique_ptr<TH1F>(new TH1F("yGE21_4hits_Pull_l1","yGE21_4hits_Pull_l1",100,-5.,5.));
  GE21_4hits_Pull_l2_y = std::unique_ptr<TH1F>(new TH1F("yGE21_4hits_Pull_l2","yGE21_4hits_Pull_l2",100,-5.,5.));
  GE21_4hits_Pull_l3_y = std::unique_ptr<TH1F>(new TH1F("yGE21_4hits_Pull_l3","yGE21_4hits_Pull_l3",100,-5.,5.));
  GE21_4hits_Pull_l4_y = std::unique_ptr<TH1F>(new TH1F("yGE21_4hits_Pull_l4","yGE21_4hits_Pull_l4",100,-5.,5.));

  SIM_SimHitEta  =  std::unique_ptr<TH1F>(new TH1F("SIM_SimHitEta", "SIM_SimHitEta", 100,1.50,2.50));
  GE11_SimHitEta =  std::unique_ptr<TH1F>(new TH1F("GE11_SimHitEta","GE11_SimHitEta",100,1.50,2.50));
  GE21_SimHitEta =  std::unique_ptr<TH1F>(new TH1F("GE21_SimHitEta","GE21_SimHitEta",100,1.50,2.50));
  GE11_SimHitEta_1D =  std::unique_ptr<TH1F>(new TH1F("GE11_SimHitEta_1D","GE11_SimHitEta_1D",100,1.50,2.50));
  GE21_SimHitEta_1D =  std::unique_ptr<TH1F>(new TH1F("GE21_SimHitEta_1D","GE21_SimHitEta_1D",100,1.50,2.50));

  GE11_Pos_Dir_eta = std::unique_ptr<TH2F>(new TH2F("GE11_Pos_Dir_eta","GE11_Pos_Dir_eta",100,-2.5,2.5,100,-2.5,2.5));
  GE21_Pos_Dir_eta = std::unique_ptr<TH2F>(new TH2F("GE21_Pos_Dir_eta","GE21_Pos_Dir_eta",100,-2.5,2.5,100,-2.5,2.5));
  ME11_Pos_Dir_eta = std::unique_ptr<TH2F>(new TH2F("ME11_Pos_Dir_eta","ME11_Pos_Dir_eta",100,-2.5,2.5,100,-2.5,2.5));
  ME21_Pos_Dir_eta = std::unique_ptr<TH2F>(new TH2F("ME21_Pos_Dir_eta","ME21_Pos_Dir_eta",100,-2.5,2.5,100,-2.5,2.5));

  GE11_Pos_Dir_phi = std::unique_ptr<TH2F>(new TH2F("GE11_Pos_Dir_phi","GE11_Pos_Dir_phi",144,-3.14,3.14,144,-3.14,3.14));
  GE21_Pos_Dir_phi = std::unique_ptr<TH2F>(new TH2F("GE21_Pos_Dir_phi","GE21_Pos_Dir_phi",144,-3.14,3.14,144,-3.14,3.14));
  ME11_Pos_Dir_phi = std::unique_ptr<TH2F>(new TH2F("ME11_Pos_Dir_phi","ME11_Pos_Dir_phi",144,-3.14,3.14,144,-3.14,3.14));
  ME21_Pos_Dir_phi = std::unique_ptr<TH2F>(new TH2F("ME21_Pos_Dir_phi","ME21_Pos_Dir_phi",144,-3.14,3.14,144,-3.14,3.14));

  GE21_Pos_Dir_eta_2hits = std::unique_ptr<TH2F>(new TH2F("GE21_Pos_Dir_eta_2hits","GE21_Pos_Dir_eta_2hits",100,-2.5,2.5,100,-2.5,2.5));
  GE21_Pos_Dir_eta_4hits = std::unique_ptr<TH2F>(new TH2F("GE21_Pos_Dir_eta_4hits","GE21_Pos_Dir_eta_4hits",100,-2.5,2.5,100,-2.5,2.5));

  GE21_Pos_Dir_phi_2hits = std::unique_ptr<TH2F>(new TH2F("GE21_Pos_Dir_phi_2hits","GE21_Pos_Dir_phi_2hits",144,-3.14,3.14,144,-3.14,3.14));
  GE21_Pos_Dir_phi_4hits = std::unique_ptr<TH2F>(new TH2F("GE21_Pos_Dir_phi_4hits","GE21_Pos_Dir_phi_4hits",144,-3.14,3.14,144,-3.14,3.14));

  GE11_Pos_XY = std::unique_ptr<TH2F>(new TH2F("GE11_Pos_XY","GE11_Pos_XY",150,-300,300, 150,-300,300));
  GE11_Neg_XY = std::unique_ptr<TH2F>(new TH2F("GE11_Neg_XY","GE11_Neg_XY",150,-300,300, 150,-300,300));
  GE21_Pos_XY = std::unique_ptr<TH2F>(new TH2F("GE21_Pos_XY","GE21_Pos_XY",175,-350,350, 175,-350,350));
  GE21_Neg_XY = std::unique_ptr<TH2F>(new TH2F("GE21_Neg_XY","GE21_Neg_XY",175,-350,350, 175,-350,350));

  GEM_Pos_RZ = std::unique_ptr<TH2F>(new TH2F("GEM_Pos_RZ","GEM_Pos_RZ",150,+550,+850, 175,0,350));
  GEM_Neg_RZ = std::unique_ptr<TH2F>(new TH2F("GEM_Neg_RZ","GEM_Neg_RZ",150,-850,-550, 175,0,350));

  GE11_CheckSegmentDirection       = std::unique_ptr<TH1F>(new TH1F("GE11_CheckSegmentDirection","GE11_CheckSegmentDirection",3,-1.5,+1.5));
  GE21_CheckSegmentDirection       = std::unique_ptr<TH1F>(new TH1F("GE21_CheckSegmentDirection","GE21_CheckSegmentDirection",3,-1.5,+1.5));
  GE21_CheckSegmentDirection_2hits = std::unique_ptr<TH1F>(new TH1F("GE21_CheckSegmentDirection_2hits","GE21_CheckSegmentDirection_2hits",3,-1.5,+1.5));
  GE21_CheckSegmentDirection_4hits = std::unique_ptr<TH1F>(new TH1F("GE21_CheckSegmentDirection_4hits","GE21_CheckSegmentDirection_4hits",3,-1.5,+1.5));


  GEM_AverageRecHitsPerEvent = std::unique_ptr<TH1F>(new TH1F("GEM_AverageRecHitsPerEvent","GEM_AverageRecHitsPerEvent",8,0.5,8.5)); // (rh for GE11 & GE2 + segments ) * 2 endcaps
  GEM_AverageSimHitsPerEvent = std::unique_ptr<TH1F>(new TH1F("GEM_AverageSimHitsPerEvent","GEM_AverageSimHitsPerEvent",8,0.5,8.5)); // (rh for GE11 & GE2 + segments ) * 2 endcaps
  GEM_NoiseFraction          = std::unique_ptr<TH1F>(new TH1F("GEM_NoiseFraction",         "GEM_NoiseFraction",8,0.5,8.5));          // (rh for GE11 & GE2 + segments ) * 2 endcaps
  GEM_AverageRecHitsPerEvent_BX0 = std::unique_ptr<TH1F>(new TH1F("GEM_AverageRecHitsPerEvent_BX0","GEM_AverageRecHitsPerEvent_BX0",8,0.5,8.5)); 
  GEM_NoiseFraction_BX0          = std::unique_ptr<TH1F>(new TH1F("GEM_NoiseFraction_BX0",         "GEM_NoiseFraction_BX0",8,0.5,8.5));          
  nEvents = 0; // nGEMrh(8,0); nGEMsh(8,0); // initialize vectors with 8 elements equal to zero ... works only when declaration and initialization are in same step
  for(int i=0; i<8; ++i) {nGEMrh.push_back(0); nGEMsh.push_back(0); nGEMrhbx0.push_back(0);}
}


TestGEMSegmentAnalyzer::~TestGEMSegmentAnalyzer()
{

  outputfile->cd();
  // Event Information
  // -----------------
  NumSegs_GE11_pos->Write();
  NumSegs_GE11_neg->Write();
  NumSegs_GE21_pos->Write();
  NumSegs_GE21_neg->Write();

  NumSegs_ME11_pos->Write();
  NumSegs_ME11_neg->Write();
  NumSegs_ME21_pos->Write();
  NumSegs_ME21_neg->Write();

  GEN_eta->Write();
  GEN_phi->Write();
  SIM_eta->Write();
  SIM_phi->Write();


  // Position and Direction GE11
  // ---------------------------
  GEMSegment_GE11_Pos_and_Dir->cd();
  GE11_Pos_eta->Write();
  GE11_Pos_phi->Write();
  GE11_Dir_eta->Write();
  GE11_Dir_phi->Write();

  ME11_Pos_eta->Write();
  ME11_Pos_phi->Write();
  ME11_Dir_eta->Write();
  ME11_Dir_phi->Write();

  GE11_Dir_eta_2hits->Write();
  GE11_Dir_phi_2hits->Write();

  Delta_Pos_SIM_GE11_BX->Write();
  Delta_Pos_SIM_GE11_eta->Write();
  Delta_Pos_SIM_GE11_phi->Write();

  Delta_Pos_ME11_GE11_eta->Write();
  Delta_Pos_ME11_GE11_phi->Write();
  Delta_Dir_ME11_GE11_eta->Write();
  Delta_Dir_ME11_GE11_phi->Write();

  GE11_LocPos_x->Write(); GE11_LocPos_y->Write(); GE11_GloPos_x->Write(); GE11_GloPos_y->Write(); GE11_GloPos_r->Write(); GE11_GloPos_p->Write(); GE11_GloPos_t->Write();
  GE11_Pos_Dir_eta->Write(); ME11_Pos_Dir_eta->Write(); GE11_Pos_Dir_phi->Write(); ME11_Pos_Dir_phi->Write(); 

  std::cout<<"GE11 Phi Correlation Factor = "<<GE11_Pos_Dir_phi->GetCorrelationFactor(1,2)<<" Covariance = "<<GE11_Pos_Dir_phi->GetCovariance(1,2)<<std::endl;
  std::cout<<"ME11 Phi Correlation Factor = "<<ME11_Pos_Dir_phi->GetCorrelationFactor(1,2)<<" Covariance = "<<ME11_Pos_Dir_phi->GetCovariance(1,2)<<std::endl;

  GE11_CheckSegmentDirection->Write();

  outputfile->cd();

  // Position and Direction GE21
  // ---------------------------
  GEMSegment_GE21_Pos_and_Dir->cd();
  GE21_Pos_eta->Write();
  GE21_Pos_phi->Write();
  GE21_Dir_eta->Write();
  GE21_Dir_phi->Write();

  ME21_Pos_eta->Write();
  ME21_Pos_phi->Write();
  ME21_Dir_eta->Write();
  ME21_Dir_phi->Write();

  GE21_Pos_eta_2hits->Write();
  GE21_Pos_phi_2hits->Write();
  GE21_Dir_eta_2hits->Write();
  GE21_Dir_phi_2hits->Write();

  GE21_Pos_eta_4hits->Write();
  GE21_Pos_phi_4hits->Write();
  GE21_Dir_eta_4hits->Write();
  GE21_Dir_phi_4hits->Write();

  Delta_Pos_SIM_GE21_BX->Write();
  Delta_Pos_SIM_GE21_eta->Write();
  Delta_Pos_SIM_GE21_phi->Write();

  Delta_Pos_ME21_GE21_eta->Write();
  Delta_Pos_ME21_GE21_phi->Write();
  Delta_Dir_ME21_GE21_eta->Write();
  Delta_Dir_ME21_GE21_phi->Write();

  Delta_Dir_ME21_GE21_eta_2hits->Write(); Delta_Dir_ME21_GE21_phi_2hits->Write(); Delta_Dir_ME21_GE21_eta_4hits->Write(); Delta_Dir_ME21_GE21_phi_4hits->Write();
  GE21_LocPos_x->Write(); GE21_LocPos_y->Write(); GE21_GloPos_x->Write(); GE21_GloPos_y->Write(); GE21_GloPos_r->Write(); GE21_GloPos_p->Write(); GE21_GloPos_t->Write();
  GE21_Pos_Dir_eta->Write(); ME21_Pos_Dir_eta->Write(); GE21_Pos_Dir_phi->Write(); ME21_Pos_Dir_phi->Write(); 
  GE21_Pos_Dir_eta_2hits->Write(); GE21_Pos_Dir_phi_2hits->Write(); GE21_Pos_Dir_eta_4hits->Write(); GE21_Pos_Dir_phi_4hits->Write(); 

  std::cout<<"GE21 Phi Correlation Factor = "<<GE21_Pos_Dir_phi->GetCorrelationFactor(1,2)<<" Covariance = "<<GE21_Pos_Dir_phi->GetCovariance(1,2)<<std::endl;
  std::cout<<"ME21 Phi Correlation Factor = "<<ME21_Pos_Dir_phi->GetCorrelationFactor(1,2)<<" Covariance = "<<ME21_Pos_Dir_phi->GetCovariance(1,2)<<std::endl;

  GE21_CheckSegmentDirection->Write(); GE21_CheckSegmentDirection_2hits->Write(); GE21_CheckSegmentDirection_4hits->Write();

  outputfile->cd();

  // All GE11 Segments
  // ------------------
  GEMSegment_GE11->cd();
  GE11_fitchi2->Write();
  GE11_fitndof->Write();
  GE11_fitchi2ndof->Write();
  GE11_numhits->Write();
  GE11_BX->Write();
  GE11_Residuals_x->Write();
  GE11_Residuals_l1_x->Write();
  GE11_Residuals_l2_x->Write();
  GE11_Pull_x->Write();
  GE11_Pull_l1_x->Write();
  GE11_Pull_l2_x->Write();
  GE11_Residuals_y->Write();
  GE11_Residuals_l1_y->Write();
  GE11_Residuals_l2_y->Write();
  GE11_Pull_y->Write();
  GE11_Pull_l1_y->Write();
  GE11_Pull_l2_y->Write();
  outputfile->cd();

  // All GE21 Segments
  // -----------------
  GEMSegment_GE21->cd();
  GE21_fitchi2->Write();
  GE21_fitndof->Write();
  GE21_fitchi2ndof->Write();
  GE21_numhits->Write();
  GE21_BX->Write();
  GE21_Residuals_x->Write();
  GE21_Residuals_l1_x->Write();
  GE21_Residuals_l2_x->Write();
  GE21_Residuals_l3_x->Write();
  GE21_Residuals_l4_x->Write();
  GE21_Pull_x->Write();
  GE21_Pull_l1_x->Write();
  GE21_Pull_l2_x->Write();
  GE21_Pull_l3_x->Write();
  GE21_Pull_l4_x->Write();
  GE21_Residuals_y->Write();
  GE21_Residuals_l1_y->Write();
  GE21_Residuals_l2_y->Write();
  GE21_Residuals_l3_y->Write();
  GE21_Residuals_l4_y->Write();
  GE21_Pull_y->Write();
  GE21_Pull_l1_y->Write();
  GE21_Pull_l2_y->Write();
  GE21_Pull_l3_y->Write();
  GE21_Pull_l4_y->Write();
  outputfile->cd();

  // GE21 Segments with 2 hits
  // -------------------------
  GEMSegment_GE21_2hits->cd();
  GE21_2hits_fitchi2->Write();
  GE21_2hits_fitndof->Write();
  GE21_2hits_fitchi2ndof->Write();
  GE21_2hits_Residuals_x->Write();
  GE21_2hits_Residuals_l1_x->Write();
  GE21_2hits_Residuals_l2_x->Write();
  GE21_2hits_Pull_x->Write();
  GE21_2hits_Pull_l1_x->Write();
  GE21_2hits_Pull_l2_x->Write();
  GE21_2hits_Residuals_y->Write();
  GE21_2hits_Residuals_l1_y->Write();
  GE21_2hits_Residuals_l2_y->Write();
  GE21_2hits_Pull_y->Write();
  GE21_2hits_Pull_l1_y->Write();
  GE21_2hits_Pull_l2_y->Write();
  outputfile->cd();

  // GE21 Segments with 3-4 hits
  // ---------------------------
  GEMSegment_GE21_4hits->cd();
  GE21_4hits_fitchi2->Write();
  GE21_4hits_fitndof->Write();
  GE21_4hits_fitchi2ndof->Write();
  GE21_4hits_Residuals_x->Write();
  GE21_4hits_Residuals_l1_x->Write();
  GE21_4hits_Residuals_l2_x->Write();
  GE21_4hits_Residuals_l3_x->Write();
  GE21_4hits_Residuals_l4_x->Write();
  GE21_4hits_Pull_x->Write();
  GE21_4hits_Pull_l1_x->Write();
  GE21_4hits_Pull_l2_x->Write();
  GE21_4hits_Pull_l3_x->Write();
  GE21_4hits_Pull_l4_x->Write();
  GE21_4hits_Residuals_y->Write();
  GE21_4hits_Residuals_l1_y->Write();
  GE21_4hits_Residuals_l2_y->Write();
  GE21_4hits_Residuals_l3_y->Write();
  GE21_4hits_Residuals_l4_y->Write();
  GE21_4hits_Pull_y->Write();
  GE21_4hits_Pull_l1_y->Write();
  GE21_4hits_Pull_l2_y->Write();
  GE21_4hits_Pull_l3_y->Write();
  GE21_4hits_Pull_l4_y->Write();
  outputfile->cd();

  // SimHit Plots
  // ------------
  GEMSegment_SimHitPlots->cd();
  SIM_SimHitEta->Write();
  GE11_SimHitEta->Write();
  GE21_SimHitEta->Write();
  // GE11_SimHitEta_1D->Sumw2(1);
  // GE21_SimHitEta_1D->Sumw2(1);
  for(int i=0; i<100; ++i) {
    int num1  = GE11_SimHitEta->GetBinContent(i+1); 
    int num2  = GE21_SimHitEta->GetBinContent(i+1); 
    int denom = SIM_SimHitEta->GetBinContent(i+1);
    if(denom > 0) {
      double ave1 = 1.0*num1/denom; double err1 = sqrt(num1)/denom;
      double ave2 = 1.0*num2/denom; double err2 = sqrt(num2)/denom;
      GE11_SimHitEta_1D->SetBinContent(i+1,ave1); GE11_SimHitEta_1D->SetBinError(i+1,err1);
      GE21_SimHitEta_1D->SetBinContent(i+1,ave2); GE21_SimHitEta_1D->SetBinError(i+1,err2);
    }
  }
  GE11_SimHitEta_1D->Write();
  GE21_SimHitEta_1D->Write();
  outputfile->cd();

  // Noise Reduction Plots
  // ---------------------
  GEMSegment_NoiseReductionPlots->cd();
  std::string labels[] = {"GE+1/1 hit", "GE-1/1 hit", "GE+1/1 seg", "GE-1/1 seg", "GE+2/1 hit","GE-2/1 hit", "GE+2/1 seg", "GE-2/1 seg", "GE+2/1", "GE-2/1 seg"};
  for(int i=0; i<8; ++i) {
    // for first estimate of uncertainties I am assuming that the variables are not correlated .... but in fact they are ... 
    if(nEvents>0 && nGEMrh[i]>0)   { GEM_AverageRecHitsPerEvent->SetBinContent(i+1,nGEMrh[i]*1.0/nEvents); GEM_AverageRecHitsPerEvent->SetBinError(i+1,nGEMrh[i]*1.0/nEvents*sqrt(1.0/nGEMrh[i]+1.0/nEvents)); }
    if(nEvents>0 && nGEMsh[i]>0)   { GEM_AverageSimHitsPerEvent->SetBinContent(i+1,nGEMsh[i]*1.0/nEvents); GEM_AverageSimHitsPerEvent->SetBinError(i+1,nGEMsh[i]*1.0/nEvents*sqrt(1.0/nGEMsh[i]+1.0/nEvents)); }
    if(nGEMrh[i]>0 && nGEMsh[i]>0) { GEM_NoiseFraction->SetBinContent(i+1,nGEMrh[i]*1.0/nGEMsh[i]);        GEM_NoiseFraction->SetBinError(i+1,nGEMrh[i]*1.0/nGEMsh[i]*sqrt(1.0/nGEMrh[i]+1.0/nGEMsh[i])); }
    GEM_AverageRecHitsPerEvent->GetXaxis()->SetBinLabel(i+1, labels[i].c_str());
    GEM_AverageSimHitsPerEvent->GetXaxis()->SetBinLabel(i+1, labels[i].c_str());
    GEM_NoiseFraction->GetXaxis()->SetBinLabel(i+1, labels[i].c_str());
    // BX = 0
    if(nEvents>0 && nGEMrhbx0[i]>0) {
      GEM_AverageRecHitsPerEvent_BX0->SetBinContent(i+1,nGEMrhbx0[i]*1.0/nEvents); GEM_AverageRecHitsPerEvent_BX0->SetBinError(i+1,nGEMrhbx0[i]*1.0/nEvents*sqrt(1.0/nGEMrhbx0[i]+1.0/nEvents));
    }
    if(nGEMrhbx0[i]>0 && nGEMsh[i]>0) {
      GEM_NoiseFraction_BX0->SetBinContent(i+1,nGEMrhbx0[i]*1.0/nGEMsh[i]);        GEM_NoiseFraction_BX0->SetBinError(i+1,nGEMrhbx0[i]*1.0/nGEMsh[i]*sqrt(1.0/nGEMrhbx0[i]+1.0/nGEMsh[i]));
    }
    GEM_AverageRecHitsPerEvent_BX0->GetXaxis()->SetBinLabel(i+1, labels[i].c_str());
    GEM_NoiseFraction_BX0->GetXaxis()->SetBinLabel(i+1, labels[i].c_str());
  }
  GEM_AverageRecHitsPerEvent->Write();
  GEM_AverageSimHitsPerEvent->Write();
  GEM_NoiseFraction->Write();
  GEM_AverageRecHitsPerEvent_BX0->Write();
  GEM_NoiseFraction_BX0->Write();
  outputfile->cd();


  // Global Position Plots
  // ---------------------
  GEMSegment_PositionPlots->cd();
  GE11_Pos_XY->Write();   GE11_Neg_XY->Write();  GE21_Pos_XY->Write();  GE21_Neg_XY->Write();  GEM_Pos_RZ->Write();  GEM_Neg_RZ->Write();


  outputfile->Close();
}


//
// member functions
//

// ------------ method called for each event  ------------
void
TestGEMSegmentAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  ++nEvents;

  iSetup.get<MuonGeometryRecord>().get(gemGeom);
  iSetup.get<MuonGeometryRecord>().get(cscGeom);

  // Handles
  // =======
  edm::Handle<reco::GenParticleCollection>      genParticles;
  iEvent.getByToken(GENParticle_Token, genParticles);

  edm::Handle<edm::HepMCProduct> hepmcevent;
  iEvent.getByToken(HEPMCCol_Token, hepmcevent);

  edm::Handle<edm::SimTrackContainer> SimTk;
  iEvent.getByToken(SIMTrack_Token,SimTk);

  std::vector<edm::Handle<edm::PSimHitContainer> > theSimHitContainers;
  iEvent.getManyByType(theSimHitContainers);
  std::vector<PSimHit> theSimHits;
  for (int i = 0; i < int(theSimHitContainers.size()); ++i) {
    theSimHits.insert(theSimHits.end(),theSimHitContainers.at(i)->begin(),theSimHitContainers.at(i)->end());
  }

  edm::Handle<CSCSegmentCollection> cscSegmentCollection;
  iEvent.getByToken(CSCSegment_Token, cscSegmentCollection);

  edm::Handle<GEMSegmentCollection> gemSegmentCollection;
  iEvent.getByToken(GEMSegment_Token, gemSegmentCollection);

  edm::Handle<GEMRecHitCollection> gemRecHits;
  iEvent.getByToken(GEMRecHit_Token,gemRecHits);


  // ================
  // GEM Segments
  // ================

  if(printSegmntInfo) std::cout <<"Number of GEM Segments in this event: "<<gemSegmentCollection->size()<<"\n"<<std::endl;

  // Loop over GEM Segments
  // ======================
  for (auto gems = gemSegmentCollection->begin(); gems != gemSegmentCollection->end(); ++gems) {

    if(printSegmntInfo)    std::cout<< "   Analyzing GEM Segment: \n   ---------------------- \n   "<<(*gems)<<std::endl;

    // obtain GEM DetId from GEMSegment ==> GEM Chamber 
    // and obtain corresponding GEMChamber from GEM Geometry
    // (GE1/1 --> station 1; GE2/1 --> station 3)
    GEMDetId id = gems->gemDetId();
    auto chamb = gemGeom->superChamber(id); 

    // calculate Local & Global Position & Direction of GEM Segment
    auto segLP = gems->localPosition();
    auto segLD = gems->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);


    // obtain constituting GEM rechits
    auto gemrhs = gems->specificRecHits();

    // cout
    if(printSegmntInfo) {
      std::cout <<"   "<<std::endl;
      std::cout <<"   GEM Segment DetID "<<id<<" = "<<id.rawId()<<std::endl;
      std::cout <<"   Locl Position "<< segLP <<" Locl Direction "<<segLD<<std::endl;
      std::cout <<"   Glob Position "<< segGP <<" Glob Direction "<<segGD<<std::endl;
      std::cout <<"   Glob Pos  eta "<<segGP.eta()  << " Glob Pos phi " <<segGP.phi()<<std::endl;
      std::cout <<"   Locl Dir thet "<<segLD.theta()<< " Locl Dir phi " <<segLD.phi()<<std::endl;
      std::cout <<"   Glob Dir thet "<<segGD.theta()<< " Glob Dir phi " <<segGD.phi()<<std::endl;
      std::cout <<"   Chi2 = "<<gems->chi2()<<" ndof = "<<gems->degreesOfFreedom()<<" ==> chi2/ndof = "<<gems->chi2()*1.0/gems->degreesOfFreedom()<<std::endl;
      std::cout <<"   BX = "<<gems->BunchX()<<std::endl;
      std::cout <<"   Number of RecHits "<<gemrhs.size()<<std::endl;
      std::cout <<"   "<<std::endl;
    }
    if(id.station()==1) {
      GE11_BX->Fill(gems->BunchX());
      GE11_fitchi2->Fill(gems->chi2());
      GE11_fitndof->Fill(gems->degreesOfFreedom());
      GE11_fitchi2ndof->Fill(gems->chi2()*1.0/gems->degreesOfFreedom());
      GE11_numhits->Fill(gems->nRecHits());
      /*
      std::cout<<"GE11 CheckSegmentDirection :: X :: "<<(segGP.x()*segGD.x())/fabs(segGP.x()*segGD.x())<<std::endl;
      std::cout<<"GE11 CheckSegmentDirection :: Y :: "<<(segGP.y()*segGD.y())/fabs(segGP.y()*segGD.y())<<std::endl;
      std::cout<<"GE11 CheckSegmentDirection :: Z :: "<<(segGP.z()*segGD.z())/fabs(segGP.z()*segGD.z())<<std::endl;
      */
      GE11_CheckSegmentDirection->Fill((segGP.z()*segGD.z())/fabs(segGP.z()*segGD.z()));
      if(id.region()>0) { GE11_Pos_XY->Fill(segGP.x(), segGP.y()); GEM_Pos_RZ->Fill(segGP.z(), segGP.perp()); ++nGEMrh[2]; if(gems->BunchX()==0) {++nGEMrhbx0[2];}}
      if(id.region()<0) { GE11_Neg_XY->Fill(segGP.x(), segGP.y()); GEM_Neg_RZ->Fill(segGP.z(), segGP.perp()); ++nGEMrh[3]; if(gems->BunchX()==0) {++nGEMrhbx0[3];}}
    }
    else if(id.station()==2 || id.station()==3) {
      GE21_BX->Fill(gems->BunchX());
      GE21_fitchi2->Fill(gems->chi2());
      GE21_fitndof->Fill(gems->degreesOfFreedom());
      GE21_fitchi2ndof->Fill(gems->chi2()*1.0/gems->degreesOfFreedom());
      GE21_numhits->Fill(gems->nRecHits());
      /*
      std::cout<<"GE21 CheckSegmentDirection :: X :: "<<(segGP.x()*segGD.x())/fabs(segGP.x()*segGD.x())<<std::endl;
      std::cout<<"GE21 CheckSegmentDirection :: Y :: "<<(segGP.y()*segGD.y())/fabs(segGP.y()*segGD.y())<<std::endl;
      std::cout<<"GE21 CheckSegmentDirection :: Z :: "<<(segGP.z()*segGD.z())/fabs(segGP.z()*segGD.z())<<std::endl;
      */
      GE21_CheckSegmentDirection->Fill((segGP.z()*segGD.z())/fabs(segGP.z()*segGD.z()));
      if(gems->nRecHits()==2) {
	GE21_2hits_fitchi2->Fill(gems->chi2());
	GE21_2hits_fitndof->Fill(gems->degreesOfFreedom());
	GE21_2hits_fitchi2ndof->Fill(gems->chi2()*1.0/gems->degreesOfFreedom());
	GE21_CheckSegmentDirection_2hits->Fill((segGP.z()*segGD.z())/fabs(segGP.z()*segGD.z()));
      }
      if(gems->nRecHits()>2) {
	GE21_4hits_fitchi2->Fill(gems->chi2());
	GE21_4hits_fitndof->Fill(gems->degreesOfFreedom());
	GE21_4hits_fitchi2ndof->Fill(gems->chi2()*1.0/gems->degreesOfFreedom());
	GE21_CheckSegmentDirection_4hits->Fill((segGP.z()*segGD.z())/fabs(segGP.z()*segGD.z()));
      }
      if(id.region()>0) { GE21_Pos_XY->Fill(segGP.x(), segGP.y()); GEM_Pos_RZ->Fill(segGP.z(), segGP.perp()); ++nGEMrh[6]; if(gems->BunchX()==0) {++nGEMrhbx0[6];}}
      if(id.region()<0) { GE21_Neg_XY->Fill(segGP.x(), segGP.y()); GEM_Neg_RZ->Fill(segGP.z(), segGP.perp()); ++nGEMrh[7]; if(gems->BunchX()==0) {++nGEMrhbx0[7];}}

    }
    else {}


    // loop on rechits ... 
    // ===================
    if(printSegmntInfo) {
      std::cout<<"      GEMRecHit :: "<<" | "<<std::setw(9)<<"SEG ETA"<<" | "<<std::setw(9)<<"SEG PHI"<<" | "<<std::setw(9)<<"SEG BX";
      std::cout<<" | "<<std::setw(9)<<"HIT ETA"<<" | "<<std::setw(9)<<"HIT PHI"<<" | "<<std::setw(9)<<"HIT BX";
      std::cout<<" | "<<std::setw(9)<<"DetId"<<"   "<<std::setw(9)<<"DetId"<<std::endl;
      for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){
	// GEM RecHit DetId & EtaPartition
	auto gemid = rh->gemId();
	auto roll = gemGeom->etaPartition(gemid);
	// GEM RecHit Local & Global Position
	auto rhGP = roll->toGlobal(rh->localPosition()); // GEM RecHit Global Position
	// auto rhLP = chamb->toLocal(rhGP);                // GEM RecHit Local Position in GEM Segment Chamber Frame
      
	std::cout<<"      GEMRecHit :: "<<" | "<<std::setw(9)<<segGD.eta()<<" | "<<std::setw(9)<<segGD.phi()<<" | "<<std::setw(9)<<gems->BunchX()
		 <<" | "<<std::setw(9)<<rhGP.eta()<<" | "<<std::setw(9)<<rhGP.phi()<<" | "<<std::setw(9)<<rh->BunchX()
		 <<" | "<<std::setw(9)<<rh->gemId().rawId()<<" = "<<rh->gemId()<<std::endl;
      }
    }


    // loop on rechits ... 
    // ===================
    if(printResidlInfo) {
      std::cout<<"      GEMRecHit :: "<<" | "<<std::setw(9)<<"ETA"<<" | "<<std::setw(9)<<"PHI"<<" | "<<std::setw(9)<<"BX" ;
      std::cout<<" | "<<std::setw(9)<<"RH X"<<" | "<<std::setw(9)<<"RH Y"<<" | "<<std::setw(9)<<"EXTR X"<<" | "<<std::setw(9)<<"EXTR Y";
      std::cout<<" | "<<std::setw(9)<<"Delta X"<<" | "<<std::setw(9)<<"Delta Y"<<" | "<<std::setw(9)<<"DetId"<<"   "<<std::setw(9)<<"DetId"<<std::endl;

      for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){
	// GEM RecHit DetId & EtaPartition
	auto gemid = rh->gemId();
	auto roll = gemGeom->etaPartition(gemid);
	// GEM RecHit Local & Global Position
	auto rhGP = roll->toGlobal(rh->localPosition()); // GEM RecHit Global Position
	auto rhLP = chamb->toLocal(rhGP);                // GEM RecHit Local Position in GEM Segment Chamber Frame

	// GEM Segment extrapolated to Layer of GEM RecHit in GEM Segment Local Frame
	float xe=0.0, ye=0.0, ze=0.0;
	if(segLD.z() != 0) {
	  xe  = segLP.x()+segLD.x()*rhLP.z()/segLD.z();
	  ye  = segLP.y()+segLD.y()*rhLP.z()/segLD.z();
	  ze = rhLP.z();
	}
	else {
	  xe  = segLP.x();
	  ye  = segLP.y();
	  ze = rhLP.z();
	}
	LocalPoint extrPoint(xe,ye,ze);                          // in segment rest frame

	std::cout<<"      GEMRecHit :: "<<" | "<<std::setw(9)<<rhGP.eta()<<" | "<<std::setw(9)<<rhGP.phi()<<" | "<<std::setw(9)<<rh->BunchX();
	std::cout<<" | "<<std::setw(9)<<rhLP.x()<<" | "<<std::setw(9)<<rhLP.y()<<" | "<<std::setw(9)<<extrPoint.x()<<" | "<<std::setw(9)<<extrPoint.y();
	std::cout<<" | "<<std::setw(9)<<extrPoint.x()-rhLP.x()<<" | "<<std::setw(9)<<extrPoint.y()-rhLP.y();
	std::cout<<" | "<<std::setw(9)<<rh->gemId().rawId()<<" = "<<rh->gemId()<<std::endl;
      }
    }

    // Residuals Calculation
    // =====================
    // take layer local position -> global -> ensemble local position same frame as segment
    for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){

      // GEM RecHit DetId & EtaPartition
      auto gemid = rh->gemId();
      auto roll = gemGeom->etaPartition(gemid);

      // GEM RecHit Local & Global Position
      auto erhLEP = rh->localPositionError();
      auto rhGP = roll->toGlobal(rh->localPosition()); // GEM RecHit Global Position
      auto rhLP = chamb->toLocal(rhGP);               // GEM RecHit Local Position in GEM Segment Chamber Frame

      if(printResidlInfo) { 
	std::cout <<"      const GEMRecHit in DetId "<<gemid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" eta = "<<rhGP.eta()<<" phi = "<<rhGP.phi()<<std::endl;
      }

      // GEM Segment extrapolated to Layer of GEM RecHit in GEM Segment Local Frame
      float xe=0.0, ye=0.0, ze=0.0;
      if(segLD.z() != 0) {
        xe  = segLP.x()+segLD.x()*rhLP.z()/segLD.z();
	ye  = segLP.y()+segLD.y()*rhLP.z()/segLD.z();
        ze = rhLP.z();
      }
      else {
	std::cout <<" Segment Local Direction Z should never be zero !!!"<<std::endl;
        xe  = segLP.x();
	ye  = segLP.y();
        ze = rhLP.z();
      }
      LocalPoint extrPoint(xe,ye,ze);                             // in segment = chamber rest frame
      // auto extSegm = roll->toLocal(chamb->toGlobal(extrPoint)); // in roll restframe ... not necessary ... this is the error we make ....
      
      if(printResidlInfo) {
	std::cout <<"      GEM Layer Id "<<rh->gemId()<<"  error on the local point "<<  erhLEP
		  <<"\n-> Ensemble Rest Frame  RH local  position "<<rhLP<<"  Segment extrapolation "<<extrPoint
		  <<"\n-> Layer Rest Frame  RH local  position "<<rhLP<<"  Segment extrapolation "<</*extSegm*/extrPoint
		  <<std::endl;
      }

      if(printResidlInfo) {
	std::cout<<" x Residual = "<<rhLP.x()-/*extSegm*/extrPoint.x()<<std::endl;
	std::cout<<" y Residual = "<<rhLP.y()-/*extSegm*/extrPoint.y()<<std::endl;
	std::cout<<" x Pull     = "<<(rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx())<<std::endl;
	std::cout<<" y Pull     = "<<(rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy())<<std::endl;
      }

      if(gemid.station()==1) {
	GE11_Residuals_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	GE11_Residuals_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	GE11_Pull_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	GE11_Pull_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	switch (gemid.layer()){
	case 1:
	  GE11_Residuals_l1_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	  GE11_Residuals_l1_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	  GE11_Pull_l1_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	  GE11_Pull_l1_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	  break;
	case 2:
	  GE11_Residuals_l2_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	  GE11_Residuals_l2_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	  GE11_Pull_l2_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	  GE11_Pull_l2_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	  break;
	  std::cout <<"      Unphysical GEM layer "<<gemid<<std::endl;
	}
      }
      else if(gemid.station()==2 || gemid.station()==3) {
	GE21_Residuals_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	GE21_Residuals_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	GE21_Pull_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	GE21_Pull_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	if(gemid.station()==2) {
	  switch (gemid.layer()){
	  case 1:
	    GE21_Residuals_l1_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	    GE21_Residuals_l1_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	    GE21_Pull_l1_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	    GE21_Pull_l1_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	    break;
	  case 2:
	    GE21_Residuals_l2_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	    GE21_Residuals_l2_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	    GE21_Pull_l2_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	    GE21_Pull_l2_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	    break;
	  default:
	  std::cout <<"      Unphysical GEM layer "<<gemid<<std::endl;
	  }
	}
	else if (gemid.station()==3) {
	  switch (gemid.layer()) {
	  case 1:
	    GE21_Residuals_l3_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	    GE21_Residuals_l3_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	    GE21_Pull_l3_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	    GE21_Pull_l3_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	    break;
	  case 2:
	    GE21_Residuals_l4_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	    GE21_Residuals_l4_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	    GE21_Pull_l4_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	    GE21_Pull_l4_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	    break;
	  default:
	    std::cout <<"      Unphysical GEM layer "<<gemid<<std::endl;
	  }
	}
      }
      else {}

      // Now divide GE21 in 2 hits vs 3-4 hits
      // -------------------------------------
      // GE21 segments with only 2 hits in GE21 Long
      if(gemid.station()==3 && gemrhs.size()==2) {
	GE21_2hits_Residuals_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	GE21_2hits_Residuals_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	GE21_2hits_Pull_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	GE21_2hits_Pull_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	if(gemid.station()==3) {
	  switch (gemid.layer()){
	  case 1:
	    GE21_2hits_Residuals_l1_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	    GE21_2hits_Residuals_l1_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	    GE21_2hits_Pull_l1_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	    GE21_2hits_Pull_l1_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	    break;
	  case 2:
	    GE21_2hits_Residuals_l2_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	    GE21_2hits_Residuals_l2_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	    GE21_2hits_Pull_l2_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	    GE21_2hits_Pull_l2_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	    break;
	  default:
	  std::cout <<"      Unphysical GEM layer "<<gemid<<std::endl;
	  }
	}
      }
      // GE21 segments with 3 or 4 hits in GE21 Short and Long
      if((gemid.station()==2 || gemid.station()==3) && gemrhs.size()>2) {
	GE21_4hits_Residuals_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	GE21_4hits_Residuals_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	GE21_4hits_Pull_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	GE21_4hits_Pull_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	if(gemid.station()==2) {
	  switch (gemid.layer()){
	  case 1:
	    GE21_4hits_Residuals_l1_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	    GE21_4hits_Residuals_l1_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	    GE21_4hits_Pull_l1_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	    GE21_4hits_Pull_l1_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	    break;
	  case 2:
	    GE21_4hits_Residuals_l2_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	    GE21_4hits_Residuals_l2_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	    GE21_4hits_Pull_l2_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	    GE21_4hits_Pull_l2_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	    break;
	  default:
	  std::cout <<"      Unphysical GEM layer "<<gemid<<std::endl;
	  }
	}
	else if (gemid.station()==3) {
	  switch (gemid.layer()) {
	  case 1:
	    GE21_4hits_Residuals_l3_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	    GE21_4hits_Residuals_l3_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	    GE21_4hits_Pull_l3_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	    GE21_4hits_Pull_l3_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	    break;
	  case 2:
	    GE21_4hits_Residuals_l4_x->Fill(rhLP.x()-/*extSegm*/extrPoint.x());
	    GE21_4hits_Residuals_l4_y->Fill(rhLP.y()-/*extSegm*/extrPoint.y());
	    GE21_4hits_Pull_l4_x->Fill((rhLP.x()-/*extSegm*/extrPoint.x())/sqrt(erhLEP.xx()));
	    GE21_4hits_Pull_l4_y->Fill((rhLP.y()-/*extSegm*/extrPoint.y())/sqrt(erhLEP.yy()));
	    break;
	  default:
	    std::cout <<"      Unphysical GEM layer "<<gemid<<std::endl;
	  }
	}
      }
    }
    if(printResidlInfo) std::cout<<"\n"<<std::endl;
  }
  if(printResidlInfo) std::cout<<"\n"<<std::endl;


  // Loop over all GEM RecHits
  // =========================
  for (GEMRecHitCollection::const_iterator recHit = gemRecHits->begin(); recHit != gemRecHits->end(); recHit++) {
    // std::cout<<"RecHit with RAWID "<<(*recHit).rawId()<<" is a DetId from detector "<<(*recHit).geographicalId().det()<<" subdetector "<<(*recHit).geographicalId().subdetId()<<std::endl;
    // std::cout<<"==> is a GEMRecHit with GEMDetId"<<(*recHit).gemId()<<std::endl;
    GEMDetId id = (GEMDetId)(*recHit).gemId();
    int      bx = (*recHit).BunchX();
    if(id.station()==1) {
      if(id.region()>0) {++nGEMrh[0]; if(bx==0) {++nGEMrhbx0[0];}}
      if(id.region()<0) {++nGEMrh[1]; if(bx==0) {++nGEMrhbx0[1];}}
    }
    if(id.station()>1) {
      if(id.region()>0) {++nGEMrh[4]; if(bx==0) {++nGEMrhbx0[4];}}
      if(id.region()<0) {++nGEMrh[5]; if(bx==0) {++nGEMrhbx0[5];}}
    }
    // std::cout<<"==> all OK for this GEMRecHit"<<std::endl;
  }


  // Plot some SimHit properties
  // ===========================
  // int Num_GE11Hits = 0;
  // int Num_GE21Hits = 0;

  // Strategy:
  // ---------
  // Select the two muons from the SimTrack Container
  // and search for the corresponding GEMSimHits to this SimTrack

  // Loop first over the SimTrack Container
  for (edm::SimTrackContainer::const_iterator it = SimTk->begin(); it != SimTk->end(); ++it) {
    std::unique_ptr<SimTrack> simtrack = std::unique_ptr<SimTrack>(new SimTrack(*it));
    if(fabs(simtrack->type()) != 13) continue;
    double simtrack_eta     = simtrack->momentum().eta();
    double simtrack_trackId = simtrack->trackId();
    if(printSimHitInfo) {
      std::cout<<"SIM Muon: id = "<<std::setw(2)<<it->type()<<" | trackId = "<<std::setw(9)<<it->trackId();
      std::cout<<" | eta = "<<std::setw(9)<<it->momentum().eta()<<" | phi = "<<std::setw(9)<<it->momentum().phi();
      std::cout<<" | pt = "<<std::setw(9)<<it->momentum().pt()<<std::endl;
    }
    SIM_SimHitEta->Fill(fabs(simtrack_eta));

    int nGE11PosHits = 0, nGE11NegHits = 0, nGE21PosHits = 0, nGE21NegHits = 0;

    // Then loop over the SimHit Container
    for (std::vector<PSimHit>::const_iterator iHit = theSimHits.begin(); iHit != theSimHits.end(); ++iHit) {
      DetId theDetUnitId((*iHit).detUnitId());
      DetId simdetid= DetId((*iHit).detUnitId());
      /*
	int pid            = (*iHit).particleType();
	int process        = (*iHit).processType();
	double time        = (*iHit).timeOfFlight();
	double log_time    = log10((*iHit).timeOfFlight());
	double log_energy  = log10((*iHit).momentumAtEntry().perp()*1000); // MeV
	double log_deposit = log10((*iHit).energyLoss()*1000000);          // keV
      */
      int simhit_trackId = (*iHit).trackId();

      if(simdetid.det()==DetId::Muon &&  simdetid.subdetId()== MuonSubdetId::GEM){ // Only GEMs

	GEMDetId gemid(theDetUnitId);
	const GEMEtaPartition* etapart = gemGeom->etaPartition(gemid);
	GlobalPoint GEMGlobalPoint = etapart->toGlobal((*iHit).localPosition());
	// GlobalPoint GEMGlobalEntry = GEMSurface.toGlobal((*iHit).entryPoint());
	// GlobalPoint GEMGlobalExit  = GEMSurface.toGlobal((*iHit).exitPoint());
	// double GEMGlobalEntryExitDZ = fabs(GEMGlobalEntry.z()-GEMGlobalExit.z());
	// double GEMLocalEntryExitDZ  = fabs((*iHit).entryPoint().z()-(*iHit).exitPoint().z());
	
	if(simhit_trackId == simtrack_trackId) {
	  if(printSimHitInfo) {
	    std::cout<<"GEM SimHit in "<<std::setw(12)<<(int)gemid<<std::setw(24)<<gemid<<" from simtrack with trackId = "<<std::setw(9)<<(*iHit).trackId();
	    std::cout<<" | time t = "<<std::setw(12)<<(*iHit).timeOfFlight()<<" | phi = "<<std::setw(12)<<GEMGlobalPoint.phi()<<" | eta = "<<std::setw(12)<<GEMGlobalPoint.eta();
	    // std::cout<<" | global position = "<<GEMGlobalPoint;
	    std::cout<<""<<std::endl;
	  }
	  if(gemid.station()==1) {
	    GE11_SimHitEta->Fill(fabs(simtrack_eta));
	    if(gemid.region()>0) { ++nGEMsh[0]; ++nGE11PosHits;}
	    if(gemid.region()<0) { ++nGEMsh[1]; ++nGE11NegHits;}

	  }
	  else {
	    GE21_SimHitEta->Fill(fabs(simtrack_eta));
	    if(gemid.region()>0) { ++nGEMsh[4]; ++nGE21PosHits; }
	    if(gemid.region()<0) { ++nGEMsh[5]; ++nGE21NegHits; }
	  }
	}
      }
    } // end loop SimHit container
    // if number of simhits > 1, assume that a sim segment could have been constructed
    if(nGE11PosHits > 1) ++nGEMsh[2];
    if(nGE11NegHits > 1) ++nGEMsh[3];
    if(nGE21PosHits > 1) ++nGEMsh[6];
    if(nGE21NegHits > 1) ++nGEMsh[7];
  }




  // Lets try to follow the muon trajectory
  // Piece of code optimized for single muon gun
  // Check first negative endcap, then positive endcap
  // Print position & direction of GEM & CSC segments
  // GE1/1 - ME1/1 - GE2/1 - ME2/1

  // ===============
  // !!! Warning !!!
  // ===============
  // This code is written for tests with single muon signatures 
  // i.e. SingleMuon Gun with one muon in each endcap
  // This code will not work properly for multi-muon signatures
  // or for addition of PU-muons
  // to improve behaviour in confrontation with noise
  // the segment closest to the simtrack (eta,phi) will be used
  // ===============

  /*
  edm::Handle<reco::GenParticleCollection>      genParticles;
  iEvent.getByLabel("genParticles", genParticles);
  std::vector<SimTrack> theSimTracks;
  edm::Handle<edm::SimTrackContainer> SimTk;
  iEvent.getByLabel("g4SimHits",SimTk);
  theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
  edm::Handle<CSCSegmentCollection> cscSegmentCollection;
  iEvent.getByLabel("cscSegments", cscSegmentCollection);

  std::vector<SimTrack> theSimTracks;
  theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
  */

  // Containers
  // std::vector< std::unique_ptr<reco::GenParticle> > GEN_muons_pos, GEN_muons_neg;
  std::vector< std::unique_ptr<HepMC::GenParticle> >   GEN_muons_pos, GEN_muons_neg;
  std::vector< std::unique_ptr<SimTrack> >             SIM_muons_pos, SIM_muons_neg;   
  std::vector< std::unique_ptr<GEMSegment> >           GE11_segs_pos, GE11_segs_neg, GE21_segs_pos, GE21_segs_neg;
  std::vector< std::unique_ptr<CSCSegment> >           ME11_segs_pos, ME11_segs_neg, ME21_segs_pos, ME21_segs_neg;

  // Gen & Sim Muon
  // for(unsigned int i=0; i<genParticles->size(); ++i) {
  // std::unique_ptr<reco::GenParticle> g = &((*genParticles)[i]);
  /*
  for(reco::GenParticleCollection::const_iterator it=genParticles->begin(); it != genParticles->end(); ++it) {
    std::unique_ptr<reco::GenParticle> genpart = std::unique_ptr<reco::GenParticle>(new reco::GenParticle(*it));
    if (genpart->status() != 3) continue;
    if (fabs(genpart->pdgId()) != 13) continue;
    if (genpart->eta() < 0.0)      { GEN_muons_neg.push_back(std::move(genpart)); }
    else if (genpart->eta() > 0.0) { GEN_muons_pos.push_back(std::move(genpart)); }
    else {}
  }
  */
  // working with reco::GenParticle seems not to work anymore
  // compilation crash:
  // undefined reference to `reco::GenParticle::~GenParticle()'
  // undefined reference to `vtable for reco::GenParticle'
  // problem can be avoided by working with HepMC::GenEvent


  // const HepMC::GenEvent * myGenEvent = new HepMC::GenEvent(*(hepmcevent->GetEvent())); // old style, pre c++11 std::unique_ptr<>
  std::unique_ptr<const HepMC::GenEvent> myGenEvent = std::unique_ptr<const HepMC::GenEvent>(new HepMC::GenEvent(*(hepmcevent->GetEvent())));
  for(HepMC::GenEvent::particle_const_iterator it = myGenEvent->particles_begin(); it != myGenEvent->particles_end(); ++it) {
    std::unique_ptr<HepMC::GenParticle> genpart = std::unique_ptr<HepMC::GenParticle>(new HepMC::GenParticle(*(*it)));
    if (fabs(genpart->pdg_id()) != 13) continue;
    GEN_eta->Fill(genpart->momentum().eta()); 
    GEN_phi->Fill(genpart->momentum().phi()); 
    if (genpart->momentum().eta() < 0.0)      { GEN_muons_neg.push_back(std::move(genpart)); }
    else if (genpart->momentum().eta() > 0.0) { GEN_muons_pos.push_back(std::move(genpart)); }
    else {}
    // pointer is moved into vector and does not exist anymore at this point. access the vector if needed.
  }
  // std::cout<<"Saved GenParticles :: size = "<<GEN_muons_pos.size()+GEN_muons_neg.size()<<std::endl;

  // for (std::vector<SimTrack>::const_iterator iTrack = theSimTracks.begin(); iTrack != theSimTracks.end(); ++iTrack) {
  for (edm::SimTrackContainer::const_iterator it = SimTk->begin(); it != SimTk->end(); ++it) {
    std::unique_ptr<SimTrack> simtrack = std::unique_ptr<SimTrack>(new SimTrack(*it));
    if(fabs(simtrack->type()) != 13) continue;
    SIM_eta->Fill(simtrack->momentum().eta()); 
    SIM_phi->Fill(simtrack->momentum().phi()); 
    if(simtrack->momentum().eta() < 0.0)      { SIM_muons_neg.push_back(std::move(simtrack)); }
    else if(simtrack->momentum().eta() > 0.0) { SIM_muons_pos.push_back(std::move(simtrack)); }
    else {}
    // pointer is moved into vector and does not exist anymore at this point. access the vector if needed.
  }
  // std::cout<<"Saved SimTracks :: size = "<<SIM_muons_pos.size()+SIM_muons_neg.size()<<std::endl;

  // GEM
  for (GEMSegmentCollection::const_iterator it = gemSegmentCollection->begin(); it != gemSegmentCollection->end(); ++it) {
    GEMDetId id = it->gemDetId();
    std::unique_ptr<GEMSegment> gemseg = std::unique_ptr<GEMSegment>(new GEMSegment(*it));
    if(id.region()==-1 && id.station()==1) { GE11_segs_neg.push_back(std::move(gemseg)); }
    else if(id.region()==+1 && id.station()==1) { GE11_segs_pos.push_back(std::move(gemseg)); }
    else if(id.region()==-1 && id.station()==3) { GE21_segs_neg.push_back(std::move(gemseg)); }
    else if(id.region()==+1 && id.station()==3) { GE21_segs_pos.push_back(std::move(gemseg)); }
    else {}
  }
  // std::cout<<"Saved GEMSegments :: size = "<<GE21_segs_pos.size()+GE11_segs_pos.size()+GE21_segs_pos.size()+GE21_segs_neg.size()<<std::endl;

  // CSC
  for (CSCSegmentCollection::const_iterator it = cscSegmentCollection->begin(); it!=cscSegmentCollection->end(); ++it){
    CSCDetId id = it->cscDetId();
    std::unique_ptr<CSCSegment> cscseg = std::unique_ptr<CSCSegment>(new CSCSegment(*it));
    if(id.endcap()==2 && id.station()==1)      { ME11_segs_neg.push_back(std::move(cscseg)); }
    else if(id.endcap()==1 && id.station()==1) { ME11_segs_pos.push_back(std::move(cscseg)); }
    else if(id.endcap()==2 && id.station()==2) { ME21_segs_neg.push_back(std::move(cscseg)); }
    else if(id.endcap()==1 && id.station()==2) { ME21_segs_pos.push_back(std::move(cscseg)); }
    else {}

    //    std::cout<<" CSC Segment in Event :: DetId = "<<(it->cscDetId()).rawId()<<" = "<<cscsegmentIt->cscDetId()<<" Time :: "<<cscsegmentIt->time()<<std::endl;
    //    std::cout<<" CSC Segment Details = "<<*cscsegmentIt<<" Time :: "<<cscsegmentIt->time()<<std::endl;
  }       
  // std::cout<<"Saved CSCSegments :: size = "<<ME21_segs_pos.size()+ME11_segs_pos.size()+ME21_segs_pos.size()+ME21_segs_neg.size()<<std::endl;



  // ---------------
  // Negative Endcap
  // ---------------
  double SIM_Pos_eta_neg = 0.0,  SIM_Pos_phi_neg = 0.0;
  double GE11_Pos_eta_neg = 0.0, GE11_Pos_phi_neg = 0.0, GE11_Dir_eta_neg = 0.0, GE11_Dir_phi_neg = 0.0; int GE11_NumSegs_neg = 0;
  double GE21_Pos_eta_neg = 0.0, GE21_Pos_phi_neg = 0.0, GE21_Dir_eta_neg = 0.0, GE21_Dir_phi_neg = 0.0; int GE21_NumSegs_neg = 0;
  double ME11_Pos_eta_neg = 0.0, ME11_Pos_phi_neg = 0.0, ME11_Dir_eta_neg = 0.0, ME11_Dir_phi_neg = 0.0; int ME11_NumSegs_neg = 0;
  double ME21_Pos_eta_neg = 0.0, ME21_Pos_phi_neg = 0.0, ME21_Dir_eta_neg = 0.0, ME21_Dir_phi_neg = 0.0; int ME21_NumSegs_neg = 0;
  int GE11_chamber_neg = 0, GE21_chamber_neg = 0, ME11_chamber_neg = 0, ME21_chamber_neg = 0;
  int GE21_nhits_neg = 0;
  float GE11_BX_neg = -10.0, GE21_BX_neg = -10.0;

  if(printEventOrder) {
    std::cout<<" Overview along the path of the muon :: neg endcap "<<"\n"<<" ------------------------------------------------- "<<std::endl;
    // for(std::vector< std::unique_ptr<reco::GenParticle> >::const_iterator it = GEN_muons_neg.begin(); it!=GEN_muons_neg.end(); ++it) {
    for(std::vector< std::unique_ptr<HepMC::GenParticle> >::const_iterator it = GEN_muons_neg.begin(); it!=GEN_muons_neg.end(); ++it) {
      // std::cout<<"GEN Muon: id = "<<std::setw(2)<<(*it)->pdgId()/*<<" | index = "<<std::setw(9)<<(*it)->index()*/;
      // std::cout<<" | eta = "<<std::setw(9)<<(*it)->eta()<<" | phi = "<<std::setw(9)<<(*it)->phi();
      // std::cout<<" | pt = "<<std::setw(9)<<(*it)->pt()<<" | st = "<<std::setw(2)<<(*it)->status()<<std::endl;
    // std::cout<<"in the loop"<<std::endl;
      std::cout<<"GEN Muon: id = "<<std::setw(2)<<(*it)->pdg_id()/*<<" | index = "<<std::setw(9)<<(*it)->index()*/;
      std::cout<<" | eta = "<<std::setw(9)<<(*it)->momentum().eta()<<" | phi = "<<std::setw(9)<<(*it)->momentum().phi();
      std::cout<<" | pt = "<<std::setw(9)<<(*it)->momentum().perp()<<" | st = "<<std::setw(2)<<(*it)->status()<<std::endl;
    }
  }
  for(std::vector< std::unique_ptr<SimTrack> >::const_iterator it = SIM_muons_neg.begin(); it!=SIM_muons_neg.end(); ++it) {
    if(printEventOrder) {
      std::cout<<"SIM Muon: id = "<<std::setw(2)<<(*it)->type()/*<<" | index = "<<std::setw(9)<<(*it)->genpartIndex()*/;
      std::cout<<" | eta = "<<std::setw(9)<<(*it)->momentum().eta()<<" | phi = "<<std::setw(9)<<(*it)->momentum().phi();
      std::cout<<" | pt = "<<std::setw(9)<<(*it)->momentum().pt()<<std::endl;
    }
    SIM_Pos_eta_neg = (*it)->momentum().eta();
    SIM_Pos_phi_neg = (*it)->momentum().phi();
  }
  for(std::vector< std::unique_ptr<GEMSegment> >::const_iterator it = GE11_segs_neg.begin(); it!=GE11_segs_neg.end(); ++it) {
    GEMDetId id = (*it)->gemDetId();
    auto chamb = gemGeom->superChamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto gemrhs = (*it)->specificRecHits();

    if(printEventOrder) {
      std::cout <<"GE1/1 Segment:"<<std::endl;
      std::cout <<"   GEMSegmnt in DetId "<<id<<" bx = "<<(*it)->BunchX()<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<std::endl;
      std::cout <<"        and dir eta = "<<std::setw(9)<<segGD.eta()<<" phi = "<<std::setw(9)<<segGD.phi()<<" with glob dir = "<<segGD<<std::endl;
      std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
      std::cout << "   Number of RecHits "<<gemrhs.size()<<std::endl;
    }
    ++GE11_NumSegs_neg; GE11_chamber_neg = id.chamber();

    double deltaR_old = sqrt(pow(SIM_Pos_eta_neg-GE11_Pos_eta_neg,2)+pow(SIM_Pos_phi_neg-GE11_Pos_phi_neg,2));
    double deltaR_new = sqrt(pow(SIM_Pos_eta_neg-segGP.eta(),2)+pow(SIM_Pos_phi_neg-segGP.phi(),2));

    if(deltaR_new < deltaR_old) {
      GE11_Pos_eta_neg = segGP.eta();  GE11_Pos_phi_neg = segGP.phi(); 
      GE11_Dir_eta_neg = segGD.eta();  GE11_Dir_phi_neg = segGD.phi();
      GE11_BX_neg = (*it)->BunchX();
    }

    GE11_Pos_eta->Fill(segGP.eta()); GE11_Pos_phi->Fill(segGP.phi()); 
    /* GE11_Dir_eta->Fill(segGD.eta()); */ GE11_Dir_phi->Fill(segGD.phi()); 
    if(GE11_Dir_eta_neg < -2.50) GE11_Dir_eta->Fill(-2.495);
    else GE11_Dir_eta->Fill(GE11_Dir_eta_neg);
    GE11_Dir_eta_2hits->Fill(segGD.eta()); GE11_Dir_phi_2hits->Fill(segGD.phi());

    GE11_LocPos_x->Fill(segLP.x());
    GE11_LocPos_y->Fill(segLP.y());
    GE11_GloPos_x->Fill(segGP.x());
    GE11_GloPos_y->Fill(segGP.y());
    GE11_GloPos_r->Fill(segGP.transverse());  // transverse = perp = sqrt (x*x+y*y)
    GE11_GloPos_p->Fill(segGP.phi().value()); // angle in radians, for angle in degrees take phi().degrees()
    GE11_GloPos_t->Fill(segGP.theta());       // theta

    for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){
      auto gemid = rh->gemId();
      auto roll = gemGeom->etaPartition(gemid);
      auto rhLP = rh->localPosition();
      auto rhGP = roll->toGlobal(rhLP);
      if(printEventOrder) {
	std::cout <<"      GEMRecHit in DetId "<<gemid<<" bx = "<<rh->BunchX()<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<" with glob pos = "<<rhGP<<std::endl;
      }
    }
  }
  for(std::vector< std::unique_ptr<CSCSegment> >::const_iterator it = ME11_segs_neg.begin(); it!=ME11_segs_neg.end(); ++it) {
    CSCDetId id = (*it)->cscDetId();
    // Special ... skip CSC if segment is not in chamber with same number (so same phi)
    if(id.chamber() != GE11_chamber_neg) continue;

    auto chamb = cscGeom->chamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto cscrhs = (*it)->specificRecHits();

    ++ME11_NumSegs_neg; ME11_chamber_neg = id.chamber();

    if(printEventOrder) {
      std::cout <<"ME1/1 Segment:"<<std::endl;
      std::cout <<"   CSCSegmnt in DetId "<<id<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<std::endl;
      std::cout <<"        and dir eta = "<<std::setw(9)<<segGD.eta()<<" phi = "<<std::setw(9)<<segGD.phi()<<" with glob dir = "<<segGD<<std::endl;
      std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
      std::cout << "   Number of RecHits "<<cscrhs.size()<<std::endl;
    }


    double deltaR_old = sqrt(pow(SIM_Pos_eta_neg-ME11_Pos_eta_neg,2)+pow(SIM_Pos_phi_neg-ME11_Pos_phi_neg,2));
    double deltaR_new = sqrt(pow(SIM_Pos_eta_neg-segGP.eta(),2)+pow(SIM_Pos_phi_neg-segGP.phi(),2));

    if(deltaR_new < deltaR_old){
    // if(ME11_chamber_neg==GE11_chamber_neg) {
      ME11_Pos_eta_neg = segGP.eta();  ME11_Pos_phi_neg = segGP.phi();
      ME11_Dir_eta_neg = segGD.eta();  ME11_Dir_phi_neg = segGD.phi();
    }
    ME11_Pos_eta->Fill(segGP.eta()); ME11_Pos_phi->Fill(segGP.phi()); 
    ME11_Dir_eta->Fill(segGD.eta()); ME11_Dir_phi->Fill(segGD.phi()); 

    for (auto rh = cscrhs.begin(); rh!= cscrhs.end(); rh++){
      auto cscid = rh->cscDetId();
      auto roll = cscGeom->chamber(cscid);
      auto rhLP = rh->localPosition();
      auto rhGP = roll->toGlobal(rhLP);
      if(printEventOrder) {
	std::cout <<"      CSCRecHit in DetId "<<cscid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<std::endl;
      }
    }
  }
  for(std::vector< std::unique_ptr<GEMSegment> >::const_iterator it = GE21_segs_neg.begin(); it!=GE21_segs_neg.end(); ++it) {
    GEMDetId id = (*it)->gemDetId();
    auto chamb = gemGeom->superChamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto gemrhs = (*it)->specificRecHits();
    if(printEventOrder) {
      std::cout <<"GE2/1 Segment:"<<std::endl;
      std::cout <<"   GEMSegmnt in DetId "<<id<<" bx = "<<(*it)->BunchX()<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<std::endl;
      std::cout <<"        and dir eta = "<<std::setw(9)<<segGD.eta()<<" phi = "<<std::setw(9)<<segGD.phi()<<" with glob dir = "<<segGD<<std::endl;
      std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
      std::cout << "   Number of RecHits "<<gemrhs.size()<<std::endl;
    }
    ++GE21_NumSegs_neg; GE21_chamber_neg = id.chamber();

    double deltaR_old = sqrt(pow(SIM_Pos_eta_neg-GE21_Pos_eta_neg,2)+pow(SIM_Pos_phi_neg-GE21_Pos_phi_neg,2));
    double deltaR_new = sqrt(pow(SIM_Pos_eta_neg-segGP.eta(),2)+pow(SIM_Pos_phi_neg-segGP.phi(),2));

    if(deltaR_new < deltaR_old){
      GE21_Pos_eta_neg = segGP.eta();  GE21_Pos_phi_neg = segGP.phi(); 
      GE21_Dir_eta_neg = segGD.eta();  GE21_Dir_phi_neg = segGD.phi(); 
      GE21_nhits_neg = gemrhs.size();
      GE21_BX_neg = (*it)->BunchX();
    }

    GE21_Pos_eta->Fill(segGP.eta()); GE21_Pos_phi->Fill(segGP.phi()); 
    GE21_Pos_eta_neg = segGP.eta();  GE21_Pos_phi_neg = segGP.phi();  
    /*GE21_Dir_eta->Fill(segGD.eta());*/ GE21_Dir_phi->Fill(segGD.phi()); 
    if(GE21_Dir_eta_neg < -2.50) GE21_Dir_eta->Fill(-2.495);
    else GE21_Dir_eta->Fill(GE21_Dir_eta_neg);

    if(gemrhs.size()==2)     {    
      GE21_Dir_eta_2hits->Fill(segGD.eta()); GE21_Dir_phi_2hits->Fill(segGD.phi());  
      GE21_Pos_eta_2hits->Fill(segGP.eta()); GE21_Pos_phi_2hits->Fill(segGP.phi());
    }
    else if(gemrhs.size()>2) {    
      GE21_Dir_eta_4hits->Fill(segGD.eta()); GE21_Dir_phi_4hits->Fill(segGD.phi());
      GE21_Pos_eta_4hits->Fill(segGP.eta()); GE21_Pos_phi_4hits->Fill(segGP.phi());
    }
    else {}

    GE21_LocPos_x->Fill(segLP.x());
    GE21_LocPos_y->Fill(segLP.y());  // std::cout<<"GE21 Segment Local Position y = "<<segLP.y()<<std::endl;
    GE21_GloPos_x->Fill(segGP.x());
    GE21_GloPos_y->Fill(segGP.y());
    GE21_GloPos_r->Fill(segGP.transverse());    // transverse = perp = sqrt (x*x+y*y)
    GE21_GloPos_p->Fill(segGP.phi().value());   // angle in radians, for angle in degrees take phi().degrees()
    GE21_GloPos_t->Fill(segGP.theta());         // theta

    for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){
      auto gemid = rh->gemId();
      auto roll = gemGeom->etaPartition(gemid);
      auto rhLP = rh->localPosition();
      auto rhGP = roll->toGlobal(rhLP);
      if(printEventOrder) {
	std::cout <<"      GEMRecHit in DetId "<<gemid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" bx = "<<rh->BunchX()<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<std::endl;
      }
    }
  }
  for(std::vector< std::unique_ptr<CSCSegment> >::const_iterator it = ME21_segs_neg.begin(); it!=ME21_segs_neg.end(); ++it) {
    CSCDetId id = (*it)->cscDetId();
    // Special ... skip CSC if segment is not in chamber with same number (so same phi)
    if(id.chamber() != GE21_chamber_neg) continue;

    auto chamb = cscGeom->chamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto cscrhs = (*it)->specificRecHits();
    if(printEventOrder) {
      std::cout <<"ME2/1 Segment:"<<std::endl;
      std::cout <<"   CSCSegmnt in DetId "<<id<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<std::endl;
      std::cout <<"        anqd dir eta = "<<std::setw(9)<<segGD.eta()<<" phi = "<<std::setw(9)<<segGD.phi()<<" with glob dir = "<<segGD<<std::endl;
      std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
      std::cout << "   Number of RecHits "<<cscrhs.size()<<std::endl;
    }
    ++ME21_NumSegs_neg; ME21_chamber_neg = id.chamber();

    double deltaR_old = sqrt(pow(SIM_Pos_eta_neg-ME21_Pos_eta_neg,2)+pow(SIM_Pos_phi_neg-ME21_Pos_phi_neg,2));
    double deltaR_new = sqrt(pow(SIM_Pos_eta_neg-segGP.eta(),2)+pow(SIM_Pos_phi_neg-segGP.phi(),2));

    if(deltaR_new < deltaR_old){
    // if(ME21_chamber_neg==GE21_chamber_neg) {
      ME21_Pos_eta_neg = segGP.eta();  ME21_Pos_phi_neg = segGP.phi(); 
      ME21_Dir_eta_neg = segGD.eta();  ME21_Dir_phi_neg = segGD.phi(); 
    }
    ME21_Pos_eta->Fill(segGP.eta()); ME21_Pos_phi->Fill(segGP.phi()); 
    ME21_Dir_eta->Fill(segGD.eta()); ME21_Dir_phi->Fill(segGD.phi()); 

    for (auto rh = cscrhs.begin(); rh!= cscrhs.end(); rh++){
      auto cscid = rh->cscDetId();
      auto roll = cscGeom->chamber(cscid);
      auto rhLP = rh->localPosition();
      auto rhGP = roll->toGlobal(rhLP);
      if(printEventOrder) {
	std::cout <<"      CSCRecHit in DetId "<<cscid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<std::endl;
      }
    }
  }
  // Outside Loop Fills
  if(fabs(SIM_Pos_eta_neg)<2.18 && fabs(SIM_Pos_eta_neg)>1.55) {NumSegs_GE11_neg->Fill(GE11_NumSegs_neg);}
  if(fabs(SIM_Pos_eta_neg)<2.40 && fabs(SIM_Pos_eta_neg)>1.50) {NumSegs_ME11_neg->Fill(ME11_NumSegs_neg);}
  if(fabs(SIM_Pos_eta_neg)<2.46 && fabs(SIM_Pos_eta_neg)>1.63) {NumSegs_GE21_neg->Fill(GE21_NumSegs_neg);}
  if(fabs(SIM_Pos_eta_neg)<2.40 && fabs(SIM_Pos_eta_neg)>1.63) {NumSegs_ME21_neg->Fill(ME21_NumSegs_neg);}

  if(GE11_NumSegs_neg == 1) {
    Delta_Pos_SIM_GE11_eta->Fill(fabs(SIM_Pos_eta_neg)-fabs(GE11_Pos_eta_neg));
    Delta_Pos_SIM_GE11_phi->Fill(reco::deltaPhi(SIM_Pos_phi_neg,GE11_Pos_phi_neg));
    // For SIM Matched segments fill BX distribution
    Delta_Pos_SIM_GE11_BX->Fill(GE11_BX_neg);
    // 2D scatter plot Pos vs Dir
    if(GE11_Dir_eta_neg < -2.50)      GE11_Pos_Dir_eta->Fill(GE11_Pos_eta_neg, -2.495);
    else if(GE11_Dir_eta_neg > +2.50) GE11_Pos_Dir_eta->Fill(GE11_Pos_eta_neg, +2.495);
    else                              GE11_Pos_Dir_eta->Fill(GE11_Pos_eta_neg, GE11_Dir_eta_neg);
    GE11_Pos_Dir_phi->Fill(GE11_Pos_phi_neg, GE11_Dir_phi_neg);
  }
  if(ME11_NumSegs_neg == 1) {
    // 2D scatter plot Pos vs Dir
    ME11_Pos_Dir_eta->Fill(ME11_Pos_eta_neg, ME11_Dir_eta_neg);
    ME11_Pos_Dir_phi->Fill(ME11_Pos_phi_neg, ME11_Dir_phi_neg);
  }
  if(GE11_NumSegs_neg == 1 && ME11_NumSegs_neg == 1 && ME11_chamber_neg == GE11_chamber_neg) {
    Delta_Pos_ME11_GE11_eta->Fill(fabs(ME11_Pos_eta_neg)-fabs(GE11_Pos_eta_neg));
    Delta_Pos_ME11_GE11_phi->Fill(reco::deltaPhi(ME11_Pos_phi_neg,GE11_Pos_phi_neg));
    double delta = ME11_Dir_eta_neg-GE11_Dir_eta_neg;
    if(delta < 0.5  && delta > -0.5) Delta_Dir_ME11_GE11_eta->Fill(delta);
    else if (delta > 0.5) Delta_Dir_ME11_GE11_eta->Fill(0.5);
    else if (delta < -0.5) Delta_Dir_ME11_GE11_eta->Fill(-0.5);
    else {}
    Delta_Dir_ME11_GE11_phi->Fill(reco::deltaPhi(ME11_Dir_phi_neg,GE11_Dir_phi_neg));
    if(printEventOrder) {
      std::cout<<"COMPARING GE11 and ME11 directions :: GE11.Dir.eta = "<<GE11_Dir_eta_neg<<" ME11.Dir.eta = "<<ME11_Dir_eta_neg<<" ME11-GE11 = "<<ME11_Dir_eta_neg-GE11_Dir_eta_neg<<std::endl;
      std::cout<<"COMPARING GE11 and ME11 directions :: GE11.Dir.phi = "<<GE11_Dir_phi_neg<<" ME11.Dir.phi = "<<ME11_Dir_phi_neg<<" ME11-GE11 = "<<reco::deltaPhi(ME11_Dir_phi_neg,GE11_Dir_phi_neg)<<std::endl;
    }
  }
  if(GE21_NumSegs_neg == 1) {
    Delta_Pos_SIM_GE21_eta->Fill(fabs(SIM_Pos_eta_neg)-fabs(GE21_Pos_eta_neg));
    Delta_Pos_SIM_GE21_phi->Fill(reco::deltaPhi(SIM_Pos_phi_neg,GE11_Pos_phi_neg));
    Delta_Pos_SIM_GE21_BX->Fill(GE21_BX_neg);
    // 2D scatter plot Pos vs Dir
    if(GE21_Dir_eta_neg < -2.50)      GE21_Pos_Dir_eta->Fill(GE21_Pos_eta_neg, -2.495);
    else if(GE21_Dir_eta_neg > +2.50) GE21_Pos_Dir_eta->Fill(GE21_Pos_eta_neg, +2.495);
    else                              GE21_Pos_Dir_eta->Fill(GE21_Pos_eta_neg, GE21_Dir_eta_neg);
    GE21_Pos_Dir_phi->Fill(GE21_Pos_phi_neg, GE21_Dir_phi_neg);
    if(GE21_nhits_neg==2) {
      if(GE21_Dir_eta_neg < -2.50)      GE21_Pos_Dir_eta_2hits->Fill(GE21_Pos_eta_neg, -2.495);
      else if(GE21_Dir_eta_neg > +2.50) GE21_Pos_Dir_eta_2hits->Fill(GE21_Pos_eta_neg, +2.495);
      else                              GE21_Pos_Dir_eta_2hits->Fill(GE21_Pos_eta_neg, GE21_Dir_eta_neg);
      GE21_Pos_Dir_phi_2hits->Fill(GE21_Pos_phi_neg, GE21_Dir_phi_neg);
    }
    else if(GE21_nhits_neg > 2) {
      if(GE21_Dir_eta_neg < -2.50)      GE21_Pos_Dir_eta_4hits->Fill(GE21_Pos_eta_neg, -2.495);
      else if(GE21_Dir_eta_neg > +2.50) GE21_Pos_Dir_eta_4hits->Fill(GE21_Pos_eta_neg, +2.495);
      else                              GE21_Pos_Dir_eta_4hits->Fill(GE21_Pos_eta_neg, GE21_Dir_eta_neg);
      GE21_Pos_Dir_phi_4hits->Fill(GE21_Pos_phi_neg, GE21_Dir_phi_neg);
    }
    else {}
  }
  if(ME21_NumSegs_neg == 1) {
    // 2D scatter plot Pos vs Dir
    ME21_Pos_Dir_eta->Fill(ME21_Pos_eta_neg, ME21_Dir_eta_neg);
    ME21_Pos_Dir_phi->Fill(ME21_Pos_phi_neg, ME21_Dir_phi_neg);
  }
  if(GE21_NumSegs_neg == 1 && ME21_NumSegs_neg == 1 && ME21_chamber_neg == GE21_chamber_neg) {
    Delta_Pos_ME21_GE21_eta->Fill(fabs(ME21_Pos_eta_neg)-fabs(GE21_Pos_eta_neg));
    Delta_Pos_ME21_GE21_phi->Fill(reco::deltaPhi(ME21_Pos_phi_neg,GE21_Pos_phi_neg));
    double delta = ME21_Dir_eta_neg-GE21_Dir_eta_neg;
    if(delta < 0.5  && delta > -0.5) Delta_Dir_ME21_GE21_eta->Fill(delta);
    else if (delta > 0.5) Delta_Dir_ME21_GE21_eta->Fill(0.5);
    else if (delta < -0.5) Delta_Dir_ME21_GE21_eta->Fill(-0.5);
    else {}
    Delta_Dir_ME21_GE21_phi->Fill(reco::deltaPhi(ME21_Dir_phi_neg,GE21_Dir_phi_neg));
    if(GE21_nhits_neg==2) {
      Delta_Dir_ME21_GE21_eta_2hits->Fill(fabs(ME21_Dir_eta_neg)-fabs(GE21_Dir_eta_neg));
      Delta_Dir_ME21_GE21_phi_2hits->Fill(reco::deltaPhi(ME21_Dir_phi_neg,GE21_Dir_phi_neg));
    }
    else if(GE21_nhits_neg > 2) {
      Delta_Dir_ME21_GE21_eta_4hits->Fill(fabs(ME21_Dir_eta_neg)-fabs(GE21_Dir_eta_neg));
      Delta_Dir_ME21_GE21_phi_4hits->Fill(reco::deltaPhi(ME21_Dir_phi_neg,GE21_Dir_phi_neg));
    }
    else{}
    if(printEventOrder) {
      std::cout<<"COMPARING GE21 and ME21 directions :: GE21.Dir.eta = "<<GE21_Dir_eta_neg<<" ME21.Dir.eta = "<<ME21_Dir_eta_neg<<" ME21-GE21 = "<<ME21_Dir_eta_neg-GE21_Dir_eta_neg<<std::endl;
      std::cout<<"COMPARING GE21 and ME21 directions :: GE21.Dir.phi = "<<GE21_Dir_phi_neg<<" ME21.Dir.phi = "<<ME21_Dir_phi_neg<<" ME21-GE21 = "<<reco::deltaPhi(ME21_Dir_phi_neg,GE21_Dir_phi_neg)<<std::endl;
    }
  }
  if(printEventOrder) { std::cout<<"\n"<<std::endl; }


  // ---------------
  // Positive Endcap
  // ---------------
  double SIM_Pos_eta_pos = 0.0,  SIM_Pos_phi_pos = 0.0;
  double GE11_Pos_eta_pos = 0.0, GE11_Pos_phi_pos = 0.0, GE11_Dir_eta_pos = 0.0, GE11_Dir_phi_pos = 0.0; int GE11_NumSegs_pos = 0;
  double GE21_Pos_eta_pos = 0.0, GE21_Pos_phi_pos = 0.0, GE21_Dir_eta_pos = 0.0, GE21_Dir_phi_pos = 0.0; int GE21_NumSegs_pos = 0;
  double ME11_Pos_eta_pos = 0.0, ME11_Pos_phi_pos = 0.0, ME11_Dir_eta_pos = 0.0, ME11_Dir_phi_pos = 0.0; int ME11_NumSegs_pos = 0;
  double ME21_Pos_eta_pos = 0.0, ME21_Pos_phi_pos = 0.0, ME21_Dir_eta_pos = 0.0, ME21_Dir_phi_pos = 0.0; int ME21_NumSegs_pos = 0;
  int GE11_chamber_pos = 0, GE21_chamber_pos = 0, ME11_chamber_pos = 0, ME21_chamber_pos = 0;
  int GE21_nhits_pos = 0;
  float GE11_BX_pos = -10.0, GE21_BX_pos = -10.0;

  if(printEventOrder) {
    std::cout<<" Overview along the path of the muon :: pos endcap "<<"\n"<<" ------------------------------------------------- "<<std::endl;
    // for(std::vector< std::unique_ptr<reco::GenParticle> >::const_iterator it = GEN_muons_pos.begin(); it!=GEN_muons_pos.end(); ++it) {
    for(std::vector< std::unique_ptr<HepMC::GenParticle> >::const_iterator it = GEN_muons_pos.begin(); it!=GEN_muons_pos.end(); ++it) {
      // std::cout<<"GEN Muon: id = "<<std::setw(2)<<(*it)->pdgId()/*<<" | index = "<<std::setw(9)<<(*it)->index()*/;
      // std::cout<<" | eta = "<<std::setw(9)<<(*it)->eta()<<" | phi = "<<std::setw(9)<<(*it)->phi();
      // std::cout<<" | pt = "<<std::setw(9)<<(*it)->pt()<<" | st = "<<std::setw(2)<<(*it)->status()<<std::endl;
      // std::cout<<"in the loop"<<std::endl;
      std::cout<<"GEN Muon: id = "<<std::setw(2)<<(*it)->pdg_id()/*<<" | index = "<<std::setw(9)<<(*it)->index()*/;
      std::cout<<" | eta = "<<std::setw(9)<<(*it)->momentum().eta()<<" | phi = "<<std::setw(9)<<(*it)->momentum().phi();
      std::cout<<" | pt = "<<std::setw(9)<<(*it)->momentum().perp()<<" | st = "<<std::setw(2)<<(*it)->status()<<std::endl;
    }
  }
  for(std::vector< std::unique_ptr<SimTrack> >::const_iterator it = SIM_muons_pos.begin(); it!=SIM_muons_pos.end(); ++it) {
    if(printEventOrder) {
      std::cout<<"SIM Muon: id = "<<std::setw(2)<<(*it)->type()/*<<" | index = "<<std::setw(9)<<(*it)->genpartIndex()*/;
      std::cout<<" | eta = "<<std::setw(9)<<(*it)->momentum().eta()<<" | phi = "<<std::setw(9)<<(*it)->momentum().phi();
      std::cout<<" | pt = "<<std::setw(9)<<(*it)->momentum().pt()<<std::endl;
    }
    SIM_Pos_eta_pos = (*it)->momentum().eta();
    SIM_Pos_phi_pos = (*it)->momentum().phi();
  }
  for(std::vector< std::unique_ptr<GEMSegment> >::const_iterator it = GE11_segs_pos.begin(); it!=GE11_segs_pos.end(); ++it) {
    GEMDetId id = (*it)->gemDetId();
    auto chamb = gemGeom->superChamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto gemrhs = (*it)->specificRecHits();
    if(printEventOrder) {
      std::cout <<"GE1/1 Segment:"<<std::endl;
      std::cout <<"   GEMSegmnt in DetId "<<id<<" bx = "<<(*it)->BunchX()<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<std::endl;
      std::cout <<"        and dir eta = "<<std::setw(9)<<segGD.eta()<<" phi = "<<std::setw(9)<<segGD.phi()<<" with glob dir = "<<segGD<<std::endl;
      std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
      std::cout << "   Number of RecHits "<<gemrhs.size()<<std::endl;
    }
    ++GE11_NumSegs_pos; GE11_chamber_pos = id.chamber();

    double deltaR_old = sqrt(pow(SIM_Pos_eta_pos-GE11_Pos_eta_pos,2)+pow(SIM_Pos_phi_pos-GE11_Pos_phi_pos,2));
    double deltaR_new = sqrt(pow(SIM_Pos_eta_pos-segGP.eta(),2)+pow(SIM_Pos_phi_pos-segGP.phi(),2));

    if(deltaR_new < deltaR_old){
      GE11_Pos_eta_pos = segGP.eta(); GE11_Pos_phi_pos = segGP.phi(); 
      GE11_Dir_eta_pos = segGD.eta(); GE11_Dir_phi_pos = segGD.phi(); 
      GE11_BX_pos = (*it)->BunchX();
    }
    GE11_Pos_eta->Fill(segGP.eta()); GE11_Pos_phi->Fill(segGP.phi()); 
    /* GE11_Dir_eta->Fill(segGD.eta()); */ GE11_Dir_phi->Fill(segGD.phi()); 
    if(GE11_Dir_eta_pos > 2.50) GE11_Dir_eta->Fill(2.495);
    else GE11_Dir_eta->Fill(GE11_Dir_eta_pos);
    if(gemrhs.size()==2) {    GE11_Dir_eta_2hits->Fill(segGD.eta()); GE11_Dir_phi_2hits->Fill(segGD.phi());  }

    GE11_LocPos_x->Fill(segLP.x());
    GE11_LocPos_y->Fill(segLP.y());
    GE11_GloPos_x->Fill(segGP.x());
    GE11_GloPos_y->Fill(segGP.y());
    GE11_GloPos_r->Fill(segGP.transverse()); // transverse = perp = sqrt (x*x+y*y)
    GE11_GloPos_p->Fill(segGP.phi().value());       // ang2 = phi().value() // angle in radians, for angle in degrees take phi().degrees()
    GE11_GloPos_t->Fill(segGP.theta());      // theta

    for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){
      auto gemid = rh->gemId();
      auto roll = gemGeom->etaPartition(gemid);
      auto rhLP = rh->localPosition();
      auto rhGP = roll->toGlobal(rhLP);
      if(printEventOrder) { 
	std::cout <<"      GEMRecHit in DetId "<<gemid<<" bx = "<<rh->BunchX()<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<" with glob pos = "<<rhGP<<std::endl; 
      }
    }
  }
  for(std::vector< std::unique_ptr<CSCSegment> >::const_iterator it = ME11_segs_pos.begin(); it!=ME11_segs_pos.end(); ++it) {
    CSCDetId id = (*it)->cscDetId();
    // Special ... skip CSC if segment is not in chamber with same number (so same phi)
    if(id.chamber() != GE11_chamber_pos) continue;

    auto chamb = cscGeom->chamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto cscrhs = (*it)->specificRecHits();
    if(printEventOrder) {
      std::cout <<"ME1/1 Segment:"<<std::endl;
      std::cout <<"   CSCSegmnt in DetId "<<id<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<std::endl;
      std::cout <<"        and dir eta = "<<std::setw(9)<<segGD.eta()<<" phi = "<<std::setw(9)<<segGD.phi()<<" with glob dir = "<<segGD<<std::endl;
      std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
      std::cout << "   Number of RecHits "<<cscrhs.size()<<std::endl;
    }
    ++ME11_NumSegs_pos; ME11_chamber_pos = id.chamber();

    double deltaR_old = sqrt(pow(SIM_Pos_eta_pos-ME11_Pos_eta_pos,2)+pow(SIM_Pos_phi_pos-ME11_Pos_phi_pos,2));
    double deltaR_new = sqrt(pow(SIM_Pos_eta_pos-segGP.eta(),2)+pow(SIM_Pos_phi_pos-segGP.phi(),2));

    if(deltaR_new < deltaR_old){
      // if(ME11_chamber_pos == GE11_chamber_pos) {
      ME11_Pos_eta_pos = segGP.eta();  ME11_Pos_phi_pos = segGP.phi();
      ME11_Dir_eta_pos = segGD.eta();  ME11_Dir_phi_pos = segGD.phi();
    }
    ME11_Pos_eta->Fill(segGP.eta()); ME11_Pos_phi->Fill(segGP.phi()); 
    ME11_Dir_eta->Fill(segGD.eta()); ME11_Dir_phi->Fill(segGD.phi()); 

    for (auto rh = cscrhs.begin(); rh!= cscrhs.end(); rh++){
      auto cscid = rh->cscDetId();
      auto roll = cscGeom->chamber(cscid);
      auto rhLP = rh->localPosition();
      auto rhGP = roll->toGlobal(rhLP);
      if(printEventOrder) {
	std::cout <<"      CSCRecHit in DetId "<<cscid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<std::endl;
      }
    }
  }
  for(std::vector< std::unique_ptr<GEMSegment> >::const_iterator it = GE21_segs_pos.begin(); it!=GE21_segs_pos.end(); ++it) {
    GEMDetId id = (*it)->gemDetId();
    auto chamb = gemGeom->superChamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto gemrhs = (*it)->specificRecHits();
    if(printEventOrder) {
      std::cout <<"GE2/1 Segment:"<<std::endl;
      std::cout <<"   GEMSegmnt in DetId "<<id<<" bx = "<<(*it)->BunchX()<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<std::endl;
      std::cout <<"        and dir eta = "<<std::setw(9)<<segGD.eta()<<" phi = "<<std::setw(9)<<segGD.phi()<<" with glob dir = "<<segGD<<std::endl;
      std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
      std::cout << "   Number of RecHits "<<gemrhs.size()<<std::endl;
    }
    ++GE21_NumSegs_pos; GE21_chamber_pos = id.chamber();

    double deltaR_old = sqrt(pow(SIM_Pos_eta_pos-GE21_Pos_eta_pos,2)+pow(SIM_Pos_phi_pos-GE21_Pos_phi_pos,2));
    double deltaR_new = sqrt(pow(SIM_Pos_eta_pos-segGP.eta(),2)+pow(SIM_Pos_phi_pos-segGP.phi(),2));

    if(deltaR_new < deltaR_old){
      GE21_Pos_eta_pos = segGP.eta();  GE21_Pos_phi_pos = segGP.phi(); 
      GE21_Dir_eta_pos = segGD.eta();  GE21_Dir_phi_pos = segGD.phi(); 
      GE21_nhits_pos = gemrhs.size();
      GE21_BX_pos = (*it)->BunchX();
    }

    GE21_Pos_eta->Fill(segGP.eta()); GE21_Pos_phi->Fill(segGP.phi());
    GE21_Pos_eta_pos = segGP.eta();  GE21_Pos_phi_pos = segGP.phi();
    /* GE21_Dir_eta->Fill(segGD.eta()); */ GE21_Dir_phi->Fill(segGD.phi()); 
    if(GE21_Dir_eta_pos > 2.50) GE21_Dir_eta->Fill(2.495);
    else GE21_Dir_eta->Fill(GE21_Dir_eta_pos);
 
    if(gemrhs.size()==2)     {    
      GE21_Pos_eta_2hits->Fill(segGP.eta()); GE21_Pos_phi_2hits->Fill(segGP.phi()); 
      GE21_Dir_eta_2hits->Fill(segGD.eta()); GE21_Dir_phi_2hits->Fill(segGD.phi());
    } 
    else if(gemrhs.size()>2) {
      GE21_Pos_eta_4hits->Fill(segGP.eta());  GE21_Pos_phi_4hits->Fill(segGP.phi());
      GE21_Dir_eta_4hits->Fill(segGD.eta()); GE21_Dir_phi_4hits->Fill(segGD.phi());  
    }
    else {}

    GE21_LocPos_x->Fill(segLP.x());
    GE21_LocPos_y->Fill(segLP.y()); // std::cout<<"GE21 Segment Local Position y = "<<segLP.y()<<std::endl;
    GE21_GloPos_x->Fill(segGP.x());
    GE21_GloPos_y->Fill(segGP.y());
    GE21_GloPos_r->Fill(segGP.transverse());  // transverse = perp = sqrt (x*x+y*y)
    GE21_GloPos_p->Fill(segGP.phi().value()); // angle in radians, for angle in degrees take phi().degrees()
    GE21_GloPos_t->Fill(segGP.theta());       // theta

    for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){
      auto gemid = rh->gemId();
      auto roll = gemGeom->etaPartition(gemid);
      auto rhLP = rh->localPosition();
      auto rhGP = roll->toGlobal(rhLP);
      if(printEventOrder) {
	std::cout <<"      GEMRecHit in DetId "<<gemid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" bx = "<<rh->BunchX()<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<std::endl;
      }
    }
  }
  for(std::vector< std::unique_ptr<CSCSegment> >::const_iterator it = ME21_segs_pos.begin(); it!=ME21_segs_pos.end(); ++it) {
    CSCDetId id = (*it)->cscDetId();
    // Special ... skip CSC if segment is not in chamber with same number (so same phi)
    if(id.chamber() != GE21_chamber_pos) continue;

    auto chamb = cscGeom->chamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto cscrhs = (*it)->specificRecHits();
    if(printEventOrder) {
      std::cout <<"ME2/1 Segment:"<<std::endl;
      std::cout <<"   CSCSegmnt in DetId "<<id<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<std::endl;
      std::cout <<"        and dir eta = "<<std::setw(9)<<segGD.eta()<<" phi = "<<std::setw(9)<<segGD.phi()<<" with glob dir = "<<segGD<<std::endl;
      std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
      std::cout << "   Number of RecHits "<<cscrhs.size()<<std::endl;
    }
    ++ME21_NumSegs_pos; ME21_chamber_pos = id.chamber();

    double deltaR_old = sqrt(pow(SIM_Pos_eta_pos-ME21_Pos_eta_pos,2)+pow(SIM_Pos_phi_pos-ME21_Pos_phi_pos,2));
    double deltaR_new = sqrt(pow(SIM_Pos_eta_pos-segGP.eta(),2)+pow(SIM_Pos_phi_pos-segGP.phi(),2));

    if(deltaR_new < deltaR_old){
      // if(ME21_chamber_pos == GE21_chamber_pos) {
      ME21_Pos_eta_pos = segGP.eta();  ME21_Pos_phi_pos = segGP.phi(); 
      ME21_Dir_eta_pos = segGD.eta();  ME21_Dir_phi_pos = segGD.phi(); 
    }
    ME21_Pos_eta->Fill(segGP.eta()); ME21_Pos_phi->Fill(segGP.phi()); 
    ME21_Dir_eta->Fill(segGD.eta()); ME21_Dir_phi->Fill(segGD.phi()); 

    for (auto rh = cscrhs.begin(); rh!= cscrhs.end(); rh++){
      auto cscid = rh->cscDetId();
      auto roll = cscGeom->chamber(cscid);
      auto rhLP = rh->localPosition();
      auto rhGP = roll->toGlobal(rhLP);
      if(printEventOrder) {
	std::cout <<"      CSCRecHit in DetId "<<cscid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<std::endl;
      }
    }
  }
  // Outside Loop Fills
  if(fabs(SIM_Pos_eta_pos)<2.18 && fabs(SIM_Pos_eta_pos)>1.55) {NumSegs_GE11_pos->Fill(GE11_NumSegs_pos);}
  if(fabs(SIM_Pos_eta_pos)<2.40 && fabs(SIM_Pos_eta_pos)>1.50) {NumSegs_ME11_pos->Fill(ME11_NumSegs_pos);}
  if(fabs(SIM_Pos_eta_pos)<2.46 && fabs(SIM_Pos_eta_pos)>1.63) {NumSegs_GE21_pos->Fill(GE21_NumSegs_pos);}
  if(fabs(SIM_Pos_eta_pos)<2.40 && fabs(SIM_Pos_eta_pos)>1.63) {NumSegs_ME21_pos->Fill(ME21_NumSegs_pos);}

  if(GE11_NumSegs_pos ==1) {
    Delta_Pos_SIM_GE11_eta->Fill(SIM_Pos_eta_pos-GE11_Pos_eta_pos);
    Delta_Pos_SIM_GE11_phi->Fill(reco::deltaPhi(SIM_Pos_phi_pos,GE11_Pos_phi_pos));
    Delta_Pos_SIM_GE11_BX->Fill(GE11_BX_pos);
    // 2D scatter plot Pos vs Dir
    if(GE11_Dir_eta_pos < -2.50)      GE11_Pos_Dir_eta->Fill(GE11_Pos_eta_pos, -2.495);
    else if(GE11_Dir_eta_pos > +2.495) GE11_Pos_Dir_eta->Fill(GE11_Pos_eta_pos, +2.495); 
    else                              GE11_Pos_Dir_eta->Fill(GE11_Pos_eta_pos, GE11_Dir_eta_pos);
    GE11_Pos_Dir_phi->Fill(GE11_Pos_phi_pos, GE11_Dir_phi_pos);
  }
  if(ME11_NumSegs_pos ==1) {
    // 2D scatter plot Pos vs Dir
    ME11_Pos_Dir_eta->Fill(ME11_Pos_eta_pos, ME11_Dir_eta_pos);
    ME11_Pos_Dir_phi->Fill(ME11_Pos_phi_pos, ME11_Dir_phi_pos);
  }
  if(GE11_NumSegs_pos ==1 && ME11_NumSegs_pos ==1 && ME11_chamber_pos == GE11_chamber_pos) {
    Delta_Pos_ME11_GE11_eta->Fill(ME11_Pos_eta_pos-GE11_Pos_eta_pos);
    Delta_Pos_ME11_GE11_phi->Fill(reco::deltaPhi(ME11_Pos_phi_pos,GE11_Pos_phi_pos));
    double delta = ME11_Dir_eta_pos-GE11_Dir_eta_pos;
    if(delta < 0.5  && delta > -0.5) Delta_Dir_ME11_GE11_eta->Fill(delta);
    else if (delta > 0.5) Delta_Dir_ME11_GE11_eta->Fill(0.5);
    else if (delta < -0.5) Delta_Dir_ME11_GE11_eta->Fill(-0.5);
    else {}
    Delta_Dir_ME11_GE11_eta->Fill(ME11_Dir_eta_pos-GE11_Dir_eta_pos);
    Delta_Dir_ME11_GE11_phi->Fill(reco::deltaPhi(ME11_Dir_phi_pos,GE11_Dir_phi_pos));
    if(printEventOrder) {
      std::cout<<"COMPARING GE11 and ME11 directions :: GE11.Dir.eta = "<<GE11_Dir_eta_pos<<" ME11.Dir.eta = "<<ME11_Dir_eta_pos<<" ME11-GE11 = "<<ME11_Dir_eta_pos-GE11_Dir_eta_pos<<std::endl;
      std::cout<<"COMPARING GE11 and ME11 directions :: GE11.Dir.phi = "<<GE11_Dir_phi_pos<<" ME11.Dir.phi = "<<ME11_Dir_phi_pos<<" ME11-GE11 = "<<reco::deltaPhi(ME11_Dir_phi_pos,GE11_Dir_phi_pos)<<std::endl;
    }
  }
  if(GE21_NumSegs_pos ==1) {
    Delta_Pos_SIM_GE21_eta->Fill(SIM_Pos_eta_pos-GE21_Pos_eta_pos);
    Delta_Pos_SIM_GE21_phi->Fill(reco::deltaPhi(SIM_Pos_phi_pos,GE11_Pos_phi_pos));
    Delta_Pos_SIM_GE21_BX->Fill(GE21_BX_pos);
    // 2D scatter plot Pos vs Dir
    if(GE21_Dir_eta_pos < -2.50)      GE21_Pos_Dir_eta->Fill(GE21_Pos_eta_pos, -2.495);
    else if(GE21_Dir_eta_pos > +2.50) GE21_Pos_Dir_eta->Fill(GE21_Pos_eta_pos, +2.495);
    else                              GE21_Pos_Dir_eta->Fill(GE21_Pos_eta_pos, GE21_Dir_eta_pos);
    GE21_Pos_Dir_phi->Fill(GE21_Pos_phi_pos, GE21_Dir_phi_pos);
    if(GE21_nhits_pos==2) {
      if(GE21_Dir_eta_pos < -2.50)      GE21_Pos_Dir_eta_2hits->Fill(GE21_Pos_eta_pos, -2.495);
      else if(GE21_Dir_eta_pos > +2.50) GE21_Pos_Dir_eta_2hits->Fill(GE21_Pos_eta_pos, +2.495);
      else                              GE21_Pos_Dir_eta_2hits->Fill(GE21_Pos_eta_pos, GE21_Dir_eta_pos);
      GE21_Pos_Dir_phi_2hits->Fill(GE21_Pos_phi_pos, GE21_Dir_phi_pos);
    }
    else if(GE21_nhits_pos > 2) {
      if(GE21_Dir_eta_pos < -2.50)      GE21_Pos_Dir_eta_4hits->Fill(GE21_Pos_eta_pos, -2.495);
      else if(GE21_Dir_eta_pos > +2.50) GE21_Pos_Dir_eta_4hits->Fill(GE21_Pos_eta_pos, +2.495);
      else                              GE21_Pos_Dir_eta_4hits->Fill(GE21_Pos_eta_pos, GE21_Dir_eta_pos);
      GE21_Pos_Dir_phi_4hits->Fill(GE21_Pos_phi_pos, GE21_Dir_phi_pos);
    }
    else {}
  }
  if(ME21_NumSegs_pos ==1) {
    // 2D scatter plot Pos vs Dir
    ME21_Pos_Dir_eta->Fill(ME21_Pos_eta_pos, ME21_Dir_eta_pos);
    ME21_Pos_Dir_phi->Fill(ME21_Pos_phi_pos, ME21_Dir_phi_pos);
  }
  if(GE21_NumSegs_pos ==1 && GE21_NumSegs_pos ==1 && ME21_chamber_pos == GE21_chamber_pos) {
    Delta_Pos_ME21_GE21_eta->Fill(ME21_Pos_eta_pos-GE21_Pos_eta_pos);
    Delta_Pos_ME21_GE21_phi->Fill(reco::deltaPhi(ME21_Pos_phi_pos,GE21_Pos_phi_pos));
    double delta = ME21_Dir_eta_pos-GE21_Dir_eta_pos;
    if(delta < 0.5  && delta > -0.5) Delta_Dir_ME21_GE21_eta->Fill(delta);
    else if (delta > 0.5) Delta_Dir_ME21_GE21_eta->Fill(0.5);
    else if (delta < -0.5) Delta_Dir_ME21_GE21_eta->Fill(-0.5);
    else {}
    Delta_Dir_ME21_GE21_phi->Fill(reco::deltaPhi(ME21_Dir_phi_pos,GE21_Dir_phi_pos));
    if(GE21_nhits_pos==2) {
      Delta_Dir_ME21_GE21_eta_2hits->Fill(fabs(ME21_Dir_eta_pos)-fabs(GE21_Dir_eta_pos));
      Delta_Dir_ME21_GE21_phi_2hits->Fill(reco::deltaPhi(ME21_Dir_phi_pos,GE21_Dir_phi_pos));
    }
    else if(GE21_nhits_pos > 2) {
      Delta_Dir_ME21_GE21_eta_4hits->Fill(fabs(ME21_Dir_eta_pos)-fabs(GE21_Dir_eta_pos));
      Delta_Dir_ME21_GE21_phi_4hits->Fill(reco::deltaPhi(ME21_Dir_phi_pos,GE21_Dir_phi_pos));
    }
    else{}
    if(printEventOrder) {
      std::cout<<"COMPARING GE21 and ME21 directions :: GE21.Dir.eta = "<<GE21_Dir_eta_pos<<" ME21.Dir.eta = "<<ME21_Dir_eta_pos<<" ME21-GE21 = "<<ME21_Dir_eta_pos-GE21_Dir_eta_pos<<std::endl;
      std::cout<<"COMPARING GE21 and ME21 directions :: GE21.Dir.phi = "<<GE21_Dir_phi_pos<<" ME21.Dir.phi = "<<ME21_Dir_phi_pos<<" ME21-GE21 = "<<reco::deltaPhi(ME21_Dir_phi_pos,GE21_Dir_phi_pos)<<std::endl;
    }
  }
  if(printEventOrder) { std::cout<<"\n"<<std::endl; }
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestGEMSegmentAnalyzer);
