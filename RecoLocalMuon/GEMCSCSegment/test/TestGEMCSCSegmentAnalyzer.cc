// -*- C++ -*-
//
// Package:    TestGEMCSCSegmentAnalyzer
// Class:      TestGEMCSCSegmentAnalyzer
// 
/**\class TestGEMCSCSegmentAnalyzer TestGEMCSCSegmentAnalyzer.cc MyAnalyzers/TestGEMCSCSegmentAnalyzer/src/TestGEMCSCSegmentAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Raffaella Radogna

// system include files
#include <memory>
#include <fstream>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <math.h>

// root include files
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <DataFormats/GEMRecHit/interface/GEMCSCSegmentCollection.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"

#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>//?
#include <Geometry/CSCGeometry/interface/CSCChamber.h>//?
#include <Geometry/CSCGeometry/interface/CSCLayer.h>//?
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/GEMDetId.h>

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
//
// class declaration
//

using namespace std;
using namespace edm;


class TestGEMCSCSegmentAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TestGEMCSCSegmentAnalyzer(const edm::ParameterSet&);
      ~TestGEMCSCSegmentAnalyzer();

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      //virtual void endRun(edm::Run const&, edm::EventSetup const&);
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

      edm::PSimHitContainer SimHitMatched(std::vector<GEMRecHit>::const_iterator, edm::ESHandle<GEMGeometry>, const edm::Event&);

      // ----------member data ---------------------------
    edm::ESHandle<GEMGeometry> gemGeom;
    edm::ESHandle<CSCGeometry> cscGeom;
    
  bool debug;
  std::string rootFileName;

  edm::EDGetTokenT<edm::SimTrackContainer>  SimTrack_Token;
  edm::EDGetTokenT<CSCSegmentCollection>    CSCSegment_Token;
  edm::EDGetTokenT<GEMCSCSegmentCollection> GEMCSCSegment_Token;
  edm::EDGetTokenT<edm::PSimHitContainer>   GEMSimHit_Token;


  std::unique_ptr<TFile> outputfile;

  std::unique_ptr<TH1F> CSC_fitchi2;
  std::unique_ptr<TH1F> GEMCSC_fitchi2;
  std::unique_ptr<TH1F> GEMCSC_fitchi2_odd;
  std::unique_ptr<TH1F> GEMCSC_fitchi2_even;
  std::unique_ptr<TH1F> GEMCSC_NumGEMRH;
  std::unique_ptr<TH1F> GEMCSC_NumCSCRH;
  std::unique_ptr<TH1F> GEMCSC_NumGEMCSCRH;
  std::unique_ptr<TH1F> GEMCSC_NumGEMCSCSeg;

  std::unique_ptr<TH1F>  GEMCSC_SSegm_LPx;
  std::unique_ptr<TH1F>  GEMCSC_SSegm_LPy;
  std::unique_ptr<TH1F>  GEMCSC_SSegm_LPEx;
  std::unique_ptr<TH1F>  GEMCSC_SSegm_LPEy;
  std::unique_ptr<TH1F>  GEMCSC_SSegm_LDx;
  std::unique_ptr<TH1F>  GEMCSC_SSegm_LDy;
  std::unique_ptr<TH1F>  GEMCSC_SSegm_LDEx;
  std::unique_ptr<TH1F>  GEMCSC_SSegm_LDEy;
  std::unique_ptr<TH2F>  GEMCSC_SSegm_LDEy_vs_ndof;
  std::unique_ptr<TH2F>  GEMCSC_SSegm_LPEy_vs_ndof;
  std::unique_ptr<TH1F>  GEMCSC_CSCSegm_LPx;
  std::unique_ptr<TH1F>  GEMCSC_CSCSegm_LPy;
  std::unique_ptr<TH1F>  GEMCSC_CSCSegm_LPEx;
  std::unique_ptr<TH1F>  GEMCSC_CSCSegm_LPEy;
  std::unique_ptr<TH1F>  GEMCSC_CSCSegm_LDx;
  std::unique_ptr<TH1F>  GEMCSC_CSCSegm_LDy;
  std::unique_ptr<TH1F>  GEMCSC_CSCSegm_LDEx;
  std::unique_ptr<TH1F>  GEMCSC_CSCSegm_LDEy;
  std::unique_ptr<TH2F>  GEMCSC_CSCSegm_LDEy_vs_ndof;
  std::unique_ptr<TH2F>  GEMCSC_CSCSegm_LPEy_vs_ndof;
    
  std::unique_ptr<TH1F>  SIMGEMCSC_SSegm_LDx;
  std::unique_ptr<TH1F>  SIMGEMCSC_SSegm_LDy;
  std::unique_ptr<TH1F>  SIMGEMCSC_SSegm_LDEx;
  std::unique_ptr<TH1F>  SIMGEMCSC_SSegm_LDEy;

    
  std::unique_ptr<TH1F> GEMCSC_Residuals_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_gem_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_csc_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_gem_odd_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_csc_odd_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_gem_even_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_csc_even_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_cscl1_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_cscl2_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_cscl3_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_cscl4_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_cscl5_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_cscl6_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_geml1_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_geml2_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_csc_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_odd_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_csc_odd_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_even_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_csc_even_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_cscl1_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_cscl2_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_cscl3_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_cscl4_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_cscl5_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_cscl6_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_geml1_x;
  std::unique_ptr<TH1F> GEMCSC_Pool_geml2_x;
  std::unique_ptr<TH1F> GEMCSC_Residuals_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_gem_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_csc_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_gem_odd_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_csc_odd_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_gem_even_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_csc_even_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_cscl1_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_cscl2_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_cscl3_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_cscl4_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_cscl5_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_cscl6_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_geml1_y;
  std::unique_ptr<TH1F> GEMCSC_Residuals_geml2_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_csc_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_odd_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_csc_odd_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_even_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_csc_even_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_cscl1_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_cscl2_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_cscl3_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_cscl4_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_cscl5_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_cscl6_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_geml1_y;
  std::unique_ptr<TH1F> GEMCSC_Pool_geml2_y;
    
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_x_newE;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_y_newE;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_odd_x_newE;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_odd_y_newE;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_x_newE_mp;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_y_newE_mp;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_odd_x_newE_mp;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_odd_y_newE_mp;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_x_newE_mm ;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_y_newE_mm ;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_odd_x_newE_mm;
  std::unique_ptr<TH1F> GEMCSC_Pool_gem_odd_y_newE_mm;
  
  std::unique_ptr<TH1F> GEMCSC_Dphi_min_afterCut;
  std::unique_ptr<TH1F> GEMCSC_Dtheta_min_afterCut;
  std::unique_ptr<TH1F> GEMCSC_DR_min_afterCut;
  std::unique_ptr<TH1F> GEMCSC_Dphi_min_afterCut_odd;
  std::unique_ptr<TH1F> GEMCSC_Dphi_min_afterCut_even;
  std::unique_ptr<TH1F> GEMCSC_Dphi_min_afterCut_l1;
  std::unique_ptr<TH1F> GEMCSC_Dphi_min_afterCut_l1_odd;
  std::unique_ptr<TH1F> GEMCSC_Dphi_min_afterCut_l1_even;
  std::unique_ptr<TH1F> GEMCSC_Dphi_min_afterCut_l2;
  std::unique_ptr<TH1F> GEMCSC_Dphi_min_afterCut_l2_odd;
  std::unique_ptr<TH1F> GEMCSC_Dphi_min_afterCut_l2_even;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut;
  std::unique_ptr<TH1F> GEMCSC_Dtheta_cscS_min_afterCut;
  std::unique_ptr<TH1F> GEMCSC_Dtheta_cscS_min_afterCut_odd;
  std::unique_ptr<TH1F> GEMCSC_Dtheta_cscS_min_afterCut_even;
  std::unique_ptr<TH1F> GEMCSC_DR_cscS_min_afterCut;

  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_l1;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_l1_odd;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_l1_even;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_l2;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_l2_odd;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_l2_even;
    
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_odd;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_even;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_odd_r1;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_odd_r2;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_odd_r3;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_odd_r4;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_odd_r5;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_odd_r6;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_odd_r7;
  std::unique_ptr<TH1F> GEMCSC_Dphi_cscS_min_afterCut_odd_r8;
  
  
  std::unique_ptr<TH1F> GEMCSC_SSegm_xe_l1;
  std::unique_ptr<TH1F> GEMCSC_SSegm_ye_l1;
  std::unique_ptr<TH1F> GEMCSC_SSegm_ze_l1;
  std::unique_ptr<TH1F> GEMCSC_SSegm_sigmaxe_l1;
  std::unique_ptr<TH1F> GEMCSC_SSegm_sigmaye_l1;
  std::unique_ptr<TH1F> GEMCSC_SSegm_xe_l1_odd;
  std::unique_ptr<TH1F> GEMCSC_SSegm_ye_l1_odd;
  std::unique_ptr<TH1F> GEMCSC_SSegm_ze_l1_odd;
  std::unique_ptr<TH1F> GEMCSC_SSegm_sigmaxe_l1_odd;
  std::unique_ptr<TH1F> GEMCSC_SSegm_sigmaye_l1_odd;
  std::unique_ptr<TH1F> GEMCSC_SSegm_xe_l1_even;
  std::unique_ptr<TH1F> GEMCSC_SSegm_ye_l1_even;
  std::unique_ptr<TH1F> GEMCSC_SSegm_ze_l1_even;
  std::unique_ptr<TH1F> GEMCSC_SSegm_sigmaxe_l1_even;
  std::unique_ptr<TH1F> GEMCSC_SSegm_sigmaye_l1_even;
  
  std::unique_ptr<TH1F> GEMCSC_SSegm_xe_l2;
  std::unique_ptr<TH1F> GEMCSC_SSegm_ye_l2;
  std::unique_ptr<TH1F> GEMCSC_SSegm_ze_l2;
  std::unique_ptr<TH1F> GEMCSC_SSegm_sigmaxe_l2;
  std::unique_ptr<TH1F> GEMCSC_SSegm_sigmaye_l2;
  std::unique_ptr<TH1F> GEMCSC_SSegm_xe_l2_odd;
  std::unique_ptr<TH1F> GEMCSC_SSegm_ye_l2_odd;
  std::unique_ptr<TH1F> GEMCSC_SSegm_ze_l2_odd;
  std::unique_ptr<TH1F> GEMCSC_SSegm_sigmaxe_l2_odd;
  std::unique_ptr<TH1F> GEMCSC_SSegm_sigmaye_l2_odd;
  std::unique_ptr<TH1F> GEMCSC_SSegm_xe_l2_even;
  std::unique_ptr<TH1F> GEMCSC_SSegm_ye_l2_even;
  std::unique_ptr<TH1F> GEMCSC_SSegm_ze_l2_even;
  std::unique_ptr<TH1F> GEMCSC_SSegm_sigmaxe_l2_even;
  std::unique_ptr<TH1F> GEMCSC_SSegm_sigmaye_l2_even;
    
    
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_xe_l1;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_ye_l1;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_ze_l1;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_sigmaxe_l1;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_sigmaye_l1;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_xe_l1_odd;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_ye_l1_odd;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_ze_l1_odd;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_sigmaxe_l1_odd;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_sigmaye_l1_odd;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_xe_l1_even;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_ye_l1_even;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_ze_l1_even;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_sigmaxe_l1_even;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_sigmaye_l1_even;
    
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_xe_l2;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_ye_l2;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_ze_l2;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_sigmaxe_l2;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_sigmaye_l2;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_xe_l2_odd;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_ye_l2_odd;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_ze_l2_odd;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_sigmaxe_l2_odd;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_sigmaye_l2_odd;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_xe_l2_even;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_ye_l2_even;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_ze_l2_even;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_sigmaxe_l2_even;
  std::unique_ptr<TH1F> GEMCSC_CSCSegm_sigmaye_l2_even;
    
  std::unique_ptr<TH2F> SIM_etaVScharge;
  std::unique_ptr<TH2F> SIM_etaVStype;
  std::unique_ptr<TH2F> SIMGEMCSC_theta_cscSsh_vs_ndof_odd;
  std::unique_ptr<TH2F> SIMGEMCSC_theta_cscSsh_vs_ndof_even;
  
  std::unique_ptr<TH1F> SIMGEMCSC_Dphi_SS_min_afterCut;
  std::unique_ptr<TH1F> SIMGEMCSC_Dtheta_SS_min_afterCut;
  std::unique_ptr<TH1F> SIMGEMCSC_Dtheta_SS_min_afterCut_odd;
  std::unique_ptr<TH1F> SIMGEMCSC_Dtheta_SS_min_afterCut_even;
  std::unique_ptr<TH1F> SIMGEMCSC_Dphi_SS_min_afterCut_odd;
  std::unique_ptr<TH1F> SIMGEMCSC_Dphi_SS_min_afterCut_even;
  
  std::unique_ptr<TH1F> SIMGEMCSC_Dphi_cscS_min_afterCut;
  std::unique_ptr<TH1F> SIMGEMCSC_Dtheta_cscS_min_afterCut;
  std::unique_ptr<TH1F> SIMGEMCSC_Dtheta_cscS_min_afterCut_odd;
  std::unique_ptr<TH1F> SIMGEMCSC_Dtheta_cscS_min_afterCut_even;
  std::unique_ptr<TH1F> SIMGEMCSC_Dphi_cscS_min_afterCut_odd;
  std::unique_ptr<TH1F> SIMGEMCSC_Dphi_cscS_min_afterCut_even;
  std::unique_ptr<TH1F> SIMGEMCSC_Residuals_gem_x;
  std::unique_ptr<TH1F> SIMGEMCSC_Residuals_gem_y;
  std::unique_ptr<TH1F> SIMGEMCSC_Pool_gem_x_newE;
  std::unique_ptr<TH1F> SIMGEMCSC_Pool_gem_y_newE;
  std::unique_ptr<TH1F> SIMGEMCSC_Residuals_gem_odd_x;
  std::unique_ptr<TH1F> SIMGEMCSC_Residuals_gem_odd_y;
  std::unique_ptr<TH1F> SIMGEMCSC_Pool_gem_x_newE_odd;
  std::unique_ptr<TH1F> SIMGEMCSC_Pool_gem_y_newE_odd;
  std::unique_ptr<TH1F> SIMGEMCSC_Residuals_gem_even_x;
  std::unique_ptr<TH1F> SIMGEMCSC_Residuals_gem_even_y;
  std::unique_ptr<TH1F> SIMGEMCSC_Pool_gem_x_newE_even;
  std::unique_ptr<TH1F> SIMGEMCSC_Pool_gem_y_newE_even;
  std::unique_ptr<TH1F> SIMGEMCSC_Residuals_gem_rhsh_x;
  std::unique_ptr<TH1F> SIMGEMCSC_Residuals_gem_rhsh_y;
  std::unique_ptr<TH1F> SIMGEMCSC_Pool_gem_rhsh_x_newE;
  std::unique_ptr<TH1F> SIMGEMCSC_Pool_gem_rhsh_y_newE;
};
//
// constants, enums and typedefs
//
// constructors and destructor
//
TestGEMCSCSegmentAnalyzer::TestGEMCSCSegmentAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  debug         = iConfig.getUntrackedParameter<bool>("Debug");
  rootFileName  = iConfig.getUntrackedParameter<std::string>("RootFileName");

  // outputfile = new TFile(rootFileName.c_str(), "RECREATE" );
  outputfile.reset(TFile::Open(rootFileName.c_str()));


  SimTrack_Token      = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  CSCSegment_Token    = consumes<CSCSegmentCollection>(edm::InputTag("cscSegments"));
  GEMCSCSegment_Token = consumes<GEMCSCSegmentCollection>(edm::InputTag("gemcscSegments"));
  GEMSimHit_Token     = consumes<edm::PSimHitContainer>(edm::InputTag("g4SimHits","MuonGEMHits"));
  
  SIM_etaVScharge     = std::unique_ptr<TH2F>(new TH2F("SimTrack_etaVScharge","SimTrack_etaVScharge",500,-2.5,2.5,6,-3,3));
  SIM_etaVStype       = std::unique_ptr<TH2F>(new TH2F("SimTrack_etaVStype","SimTrack_etaVStype",500,-2.5,2.5,30,-15,15));
  CSC_fitchi2         = std::unique_ptr<TH1F>(new TH1F("ReducedChi2_csc","ReducedChi2_csc",160,0.,4.));
    
  GEMCSC_fitchi2      = std::unique_ptr<TH1F>(new TH1F("ReducedChi2_gemcsc","ReducedChi2_gemcsc",160,0.,4.));
  GEMCSC_fitchi2_odd  = std::unique_ptr<TH1F>(new TH1F("ReducedChi2_odd_gemcsc","ReducedChi2_odd_gemcsc",160,0.,4.));
  GEMCSC_fitchi2_even = std::unique_ptr<TH1F>(new TH1F("ReducedChi2_even_gemcsc","ReducedChi2_even_gemcsc",160,0.,4.));
  GEMCSC_NumGEMRH     = std::unique_ptr<TH1F>(new TH1F("NumGEMRH","NumGEMRH",20,0.,20));
  GEMCSC_NumCSCRH     = std::unique_ptr<TH1F>(new TH1F("NumCSCRH","NumCSCRH",20,0.,20));
  GEMCSC_NumGEMCSCRH  = std::unique_ptr<TH1F>(new TH1F("NumGEMCSCRH","NumGEMCSCRH",20,0.,20));
  GEMCSC_NumGEMCSCSeg = std::unique_ptr<TH1F>(new TH1F("NumGMCSCSeg","NumGEMCSCSeg",20,0.,20));
  GEMCSC_SSegm_LPx    = std::unique_ptr<TH1F>(new TH1F("SuperS_LPx","SuperS_LPx",1200,-60.,60));
  GEMCSC_SSegm_LPy    = std::unique_ptr<TH1F>(new TH1F("SuperS_LPy","SuperS_LPy",4000,-200.,200));
  GEMCSC_SSegm_LPEx   = std::unique_ptr<TH1F>(new TH1F("SuperS_LPEx","SuperS_LPEx",10000,0.,0.5));
  GEMCSC_SSegm_LPEy   = std::unique_ptr<TH1F>(new TH1F("SuperS_LPEy","SuperS_LPEy",10000,0.,5));
  //GEMCSC_SSegm_LPEz = std::unique_ptr<TH1F>(new TH1F("SuperS_LPEz","SuperS_LPEz",1000,0.,0.5));
  GEMCSC_SSegm_LDx    = std::unique_ptr<TH1F>(new TH1F("SuperS_LDx","SuperS_LDx",1000,-2.,2));
  GEMCSC_SSegm_LDy    = std::unique_ptr<TH1F>(new TH1F("SuperS_LDy","SuperS_LDy",1000,-2.,2));
  GEMCSC_SSegm_LDEx   = std::unique_ptr<TH1F>(new TH1F("SuperS_LDEx","SuperS_LDEx",10000,0.,0.05));
  GEMCSC_SSegm_LDEy   = std::unique_ptr<TH1F>(new TH1F("SuperS_LDEy","SuperS_LDEy",10000,0.,0.5));
  GEMCSC_SSegm_LDEy_vs_ndof    = std::unique_ptr<TH2F>(new TH2F("SSegm_LDEy_vs_ndof","SSegm_LDEy vs ndof",1000,0.,0.05,15,-0.5,14.5));
  GEMCSC_SSegm_LPEy_vs_ndof    = std::unique_ptr<TH2F>(new TH2F("SSegm_LPEy_vs_ndof","SSegm_LPEy vs ndof",1000,0.,0.5,15,-0.5,14.5));
  //GEMCSC_SSegm_LDEz = std::unique_ptr<TH1F>(new TH1F("SuperS_LDEz","SuperS_LDEz",1000,0.,0.05));
  GEMCSC_CSCSegm_LPx  = std::unique_ptr<TH1F>(new TH1F("CSCSegm_LPx","CSCSegm_LPx",1200,-60.,60));
  GEMCSC_CSCSegm_LPy  = std::unique_ptr<TH1F>(new TH1F("CSCSegm_LPy","CSCSegm_LPy",4000,-200.,200));
  GEMCSC_CSCSegm_LPEx = std::unique_ptr<TH1F>(new TH1F("CSCSegm_LPEx","CSCSegm_LPEx",10000,0.,0.5));
  GEMCSC_CSCSegm_LPEy = std::unique_ptr<TH1F>(new TH1F("CSCSegm_LPEy","CSCSegm_LPEy",10000,0.,5));
  //GEMCSC_CSCSegm_LPEz = std::unique_ptr<TH1F>(new TH1F("CSCSegm_LPEz","CSCSegm_LPEz",1000,0.,0.5));
  GEMCSC_CSCSegm_LDx  = std::unique_ptr<TH1F>(new TH1F("CSCSegm_LDx","CSCSegm_LDx",1000,-2.,2));
  GEMCSC_CSCSegm_LDy  = std::unique_ptr<TH1F>(new TH1F("CSCSegm_LDy","CSCSegm_LDy",1000,-2.,2));
  GEMCSC_CSCSegm_LDEx = std::unique_ptr<TH1F>(new TH1F("CSCSegm_LDEx","CSCSegm_LDEx",10000,0.,0.05));
  GEMCSC_CSCSegm_LDEy = std::unique_ptr<TH1F>(new TH1F("CSCSegm_LDEy","CSCSegm_LDEy",10000,0.,0.5));
  //GEMCSC_CSCSegm_LDEz = std::unique_ptr<TH1F>(new TH1F("CSCSegm_LDEz","CSCSegm_LDEz",1000,0.,0.05));
  GEMCSC_CSCSegm_LDEy_vs_ndof    = std::unique_ptr<TH2F>(new TH2F("CSCSegm_LDEy_vs_ndof","CSCSegm_LDEy vs ndof",1000,0.,0.05,15,-0.5,14.5));
  GEMCSC_CSCSegm_LPEy_vs_ndof    = std::unique_ptr<TH2F>(new TH2F("CSCSegm_LPEy_vs_ndof","CSCSegm_LPEy vs ndof",1000,0.,0.5,15,-0.5,14.5));
  SIMGEMCSC_SSegm_LDx = std::unique_ptr<TH1F>(new TH1F("SSegm_LDx_expected","SSegm_LDx",1000,-2.,2));
  SIMGEMCSC_SSegm_LDy = std::unique_ptr<TH1F>(new TH1F("SSegm_LDy_expected","SSegm_LDy",1000,-2.,2));
  SIMGEMCSC_SSegm_LDEx    = std::unique_ptr<TH1F>(new TH1F("SuperS_LDEx_expected","SuperS_LDEx",10000,0.,0.005));
  SIMGEMCSC_SSegm_LDEy    = std::unique_ptr<TH1F>(new TH1F("SuperS_LDEy_expected","SuperS_LDEy",1000,0.,0.05));


  GEMCSC_SSegm_xe_l1        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_xe_l1","GEMCSC_SSegm_xe_l1",1200,-60.,60));
  GEMCSC_SSegm_ye_l1        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_ye_l1","GEMCSC_SSegm_ye_l1",4000,-200.,200));
  GEMCSC_SSegm_ze_l1        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_ze_l1","GEMCSC_SSegm_ze_l1",12000,-600.,600));
  GEMCSC_SSegm_xe_l1_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_xe_l1_odd","GEMCSC_SSegm_xe_l1_odd",1200,-60.,60));
  GEMCSC_SSegm_ye_l1_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_ye_l1_odd","GEMCSC_SSegm_ye_l1_odd",4000,-200.,200));
  GEMCSC_SSegm_ze_l1_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_ze_l1_odd","GEMCSC_SSegm_ze_l1_odd",12000,-600.,600));
  GEMCSC_SSegm_xe_l1_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_xe_l1_even","GEMCSC_SSegm_xe_l1_even",1200,-60.,60));
  GEMCSC_SSegm_ye_l1_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_ye_l1_even","GEMCSC_SSegm_ye_l1_even",4000,-200.,200));
  GEMCSC_SSegm_ze_l1_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_ze_l1_even","GEMCSC_SSegm_ze_l1_even",12000,-600.,600));
  GEMCSC_SSegm_xe_l2        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_xe_l2","GEMCSC_SSegm_xe_l2",1200,-60.,60));
  GEMCSC_SSegm_ye_l2        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_ye_l2","GEMCSC_SSegm_ye_l2",4000,-200.,200));
  GEMCSC_SSegm_ze_l2        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_ze_l2","GEMCSC_SSegm_ze_l2",12000,-600.,600));
  GEMCSC_SSegm_xe_l2_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_xe_l2_odd","GEMCSC_SSegm_xe_l2_odd",1200,-60.,60));
  GEMCSC_SSegm_ye_l2_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_ye_l2_odd","GEMCSC_SSegm_ye_l2_odd",4000,-200.,200));
  GEMCSC_SSegm_ze_l2_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_ze_l2_odd","GEMCSC_SSegm_ze_l2_odd",12000,-600.,600));
  GEMCSC_SSegm_xe_l2_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_xe_l2_even","GEMCSC_SSegm_xe_l2_even",1200,-60.,60));
  GEMCSC_SSegm_ye_l2_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_ye_l2_even","GEMCSC_SSegm_ye_l2_even",4000,-200.,200));
  GEMCSC_SSegm_ze_l2_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_ze_l2_even","GEMCSC_SSegm_ze_l2_even",12000,-600.,600));
    
  GEMCSC_CSCSegm_xe_l1        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_xe_l1","GEMCSC_CSCSegm_xe_l1",1200,-60.,60));
  GEMCSC_CSCSegm_ye_l1        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_ye_l1","GEMCSC_CSCSegm_ye_l1",4000,-200.,200));
  GEMCSC_CSCSegm_ze_l1        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_ze_l1","GEMCSC_CSCSegm_ze_l1",12000,-600.,600));
  GEMCSC_CSCSegm_xe_l1_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_xe_l1_odd","GEMCSC_CSCSegm_xe_l1_odd",1200,-60.,60));
  GEMCSC_CSCSegm_ye_l1_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_ye_l1_odd","GEMCSC_CSCSegm_ye_l1_odd",4000,-200.,200));
  GEMCSC_CSCSegm_ze_l1_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_ze_l1_odd","GEMCSC_CSCSegm_ze_l1_odd",12000,-600.,600));
  GEMCSC_CSCSegm_xe_l1_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_xe_l1_even","GEMCSC_CSCSegm_xe_l1_even",1200,-60.,60));
  GEMCSC_CSCSegm_ye_l1_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_ye_l1_even","GEMCSC_CSCSegm_ye_l1_even",4000,-200.,200));
  GEMCSC_CSCSegm_ze_l1_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_ze_l1_even","GEMCSC_CSCSegm_ze_l1_even",12000,-600.,600));
  GEMCSC_CSCSegm_xe_l2        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_xe_l2","GEMCSC_CSCSegm_xe_l2",1200,-60.,60));
  GEMCSC_CSCSegm_ye_l2        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_ye_l2","GEMCSC_CSCSegm_ye_l2",4000,-200.,200));
  GEMCSC_CSCSegm_ze_l2        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_ze_l2","GEMCSC_CSCSegm_ze_l2",12000,-600.,600));
  GEMCSC_CSCSegm_xe_l2_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_xe_l2_odd","GEMCSC_CSCSegm_xe_l2_odd",1200,-60.,60));
  GEMCSC_CSCSegm_ye_l2_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_ye_l2_odd","GEMCSC_CSCSegm_ye_l2_odd",4000,-200.,200));
  GEMCSC_CSCSegm_ze_l2_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_ze_l2_odd","GEMCSC_CSCSegm_ze_l2_odd",12000,-600.,600));
  GEMCSC_CSCSegm_xe_l2_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_xe_l2_even","GEMCSC_CSCSegm_xe_l2_even",1200,-60.,60));
  GEMCSC_CSCSegm_ye_l2_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_ye_l2_even","GEMCSC_CSCSegm_ye_l2_even",4000,-200.,200));
  GEMCSC_CSCSegm_ze_l2_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_ze_l2_even","GEMCSC_CSCSegm_ze_l2_even",12000,-600.,600));
  
  GEMCSC_SSegm_sigmaxe_l1        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_sigmaxe_l1","GEMCSC_SSegm_sigmaxe_l1",1000,0.,0.5));
  GEMCSC_SSegm_sigmaye_l1        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_sigmaye_l1","GEMCSC_SSegm_sigmaye_l1",1000,0.,10));
  GEMCSC_SSegm_sigmaxe_l1_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_sigmaxe_l1_odd","GEMCSC_SSegm_sigmaxe_l1_odd",1000,0.,0.5));
  GEMCSC_SSegm_sigmaye_l1_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_sigmaye_l1_odd","GEMCSC_SSegm_sigmaye_l1_odd",1000,0.,10));
  GEMCSC_SSegm_sigmaxe_l1_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_sigmaxe_l1_even","GEMCSC_SSegm_sigmaxe_l1_even",1000,0.,0.5));
  GEMCSC_SSegm_sigmaye_l1_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_sigmaye_l1_even","GEMCSC_SSegm_sigmaye_l1_even",1000,0.,10));
  GEMCSC_SSegm_sigmaxe_l2        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_sigmaxe_l2","GEMCSC_SSegm_sigmaxe_l2",1000,0.,0.5));
  GEMCSC_SSegm_sigmaye_l2        = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_sigmaye_l2","GEMCSC_SSegm_sigmaye_l2",1000,0.,10));
  GEMCSC_SSegm_sigmaxe_l2_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_sigmaxe_l2_odd","GEMCSC_SSegm_sigmaxe_l2_odd",1000,0.,0.5));
  GEMCSC_SSegm_sigmaye_l2_odd    = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_sigmaye_l2_odd","GEMCSC_SSegm_sigmaye_l2_odd",1000,0.,10));
  GEMCSC_SSegm_sigmaxe_l2_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_sigmaxe_l2_even","GEMCSC_SSegm_sigmaxe_l2_even",1000,0.,0.5));
  GEMCSC_SSegm_sigmaye_l2_even   = std::unique_ptr<TH1F>(new TH1F("GEMCSC_SSegm_sigmaye_l2_even","GEMCSC_SSegm_sigmaye_l2_even",1000,0.,10));
  GEMCSC_CSCSegm_sigmaxe_l1      = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_sigmaxe_l1","GEMCSC_CSCSegm_sigmaxe_l1",1000,0.,0.5));
  GEMCSC_CSCSegm_sigmaye_l1      = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_sigmaye_l1","GEMCSC_CSCSegm_sigmaye_l1",1000,0.,10));
  GEMCSC_CSCSegm_sigmaxe_l1_odd  = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_sigmaxe_l1_odd","GEMCSC_CSCSegm_sigmaxe_l1_odd",1000,0.,0.5));
  GEMCSC_CSCSegm_sigmaye_l1_odd  = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_sigmaye_l1_odd","GEMCSC_CSCSegm_sigmaye_l1_odd",1000,0.,10));
  GEMCSC_CSCSegm_sigmaxe_l1_even = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_sigmaxe_l1_even","GEMCSC_CSCSegm_sigmaxe_l1_even",1000,0.,0.5));
  GEMCSC_CSCSegm_sigmaye_l1_even = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_sigmaye_l1_even","GEMCSC_CSCSegm_sigmaye_l1_even",1000,0.,10));
  GEMCSC_CSCSegm_sigmaxe_l2      = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_sigmaxe_l2","GEMCSC_CSCSegm_sigmaxe_l2",1000,0.,0.5));
  GEMCSC_CSCSegm_sigmaye_l2      = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_sigmaye_l2","GEMCSC_CSCSegm_sigmaye_l2",1000,0.,10));
  GEMCSC_CSCSegm_sigmaxe_l2_odd  = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_sigmaxe_l2_odd","GEMCSC_CSCSegm_sigmaxe_l2_odd",1000,0.,0.5));
  GEMCSC_CSCSegm_sigmaye_l2_odd  = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_sigmaye_l2_odd","GEMCSC_CSCSegm_sigmaye_l2_odd",1000,0.,10));
  GEMCSC_CSCSegm_sigmaxe_l2_even = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_sigmaxe_l2_even","GEMCSC_CSCSegm_sigmaxe_l2_even",1000,0.,0.5));
  GEMCSC_CSCSegm_sigmaye_l2_even = std::unique_ptr<TH1F>(new TH1F("GEMCSC_CSCSegm_sigmaye_l2_even","GEMCSC_CSCSegm_sigmaye_l2_even",1000,0.,10));
  
  GEMCSC_Residuals_x            = std::unique_ptr<TH1F>(new TH1F("xGEMCSCRes","xGEMCSCRes",100,-0.5,0.5));
  GEMCSC_Residuals_gem_x        = std::unique_ptr<TH1F>(new TH1F("xGEMRes","xGEMRes",100,-0.5,0.5));
  GEMCSC_Residuals_csc_x        = std::unique_ptr<TH1F>(new TH1F("xCSCRes","xCSCRes",100,-0.5,0.5));
  GEMCSC_Residuals_gem_even_x   = std::unique_ptr<TH1F>(new TH1F("xGEMRes_even","xGEMRes even",100,-0.5,0.5));
  GEMCSC_Residuals_csc_even_x   = std::unique_ptr<TH1F>(new TH1F("xCSCRes_even","xCSCRes even",100,-0.5,0.5));
  GEMCSC_Residuals_gem_odd_x    = std::unique_ptr<TH1F>(new TH1F("xGEMRes_odd","xGEMRes odd",100,-0.5,0.5));
  GEMCSC_Residuals_csc_odd_x    = std::unique_ptr<TH1F>(new TH1F("xCSCRes_odd","xCSCRes odd",100,-0.5,0.5));
  GEMCSC_Residuals_cscl1_x      = std::unique_ptr<TH1F>(new TH1F("xGEMCSCRes_cscl1","xGEMCSCRes_cscl1",100,-0.5,0.5));
  GEMCSC_Residuals_cscl2_x      = std::unique_ptr<TH1F>(new TH1F("xGEMCSCRes_cscl2","xGEMCSCRes_cscl2",100,-0.5,0.5));
  GEMCSC_Residuals_cscl3_x      = std::unique_ptr<TH1F>(new TH1F("xGEMCSCRes_cscl3","xGEMCSCRes_cscl3",100,-0.5,0.5));
  GEMCSC_Residuals_cscl4_x      = std::unique_ptr<TH1F>(new TH1F("xGEMCSCRes_cscl4","xGEMCSCRes_cscl4",100,-0.5,0.5));
  GEMCSC_Residuals_cscl5_x      = std::unique_ptr<TH1F>(new TH1F("xGEMCSCRes_cscl5","xGEMCSCRes_cscl5",100,-0.5,0.5));
  GEMCSC_Residuals_cscl6_x      = std::unique_ptr<TH1F>(new TH1F("xGEMCSCRes_cscl6","xGEMCSCRes_cscl6",100,-0.5,0.5));
  GEMCSC_Residuals_geml1_x      = std::unique_ptr<TH1F>(new TH1F("xGEMCSCRes_geml1","xGEMCSCRes_geml1",100,-0.5,0.5));
  GEMCSC_Residuals_geml2_x      = std::unique_ptr<TH1F>(new TH1F("xGEMCSCRes_geml2","xGEMCSCRes_geml2",100,-0.5,0.5));
  GEMCSC_Pool_x                 = std::unique_ptr<TH1F>(new TH1F("xGEMCSCPool","xGEMCSCPool",100,-5.,5.));
  GEMCSC_Pool_gem_x             = std::unique_ptr<TH1F>(new TH1F("xGEMPool","xGEMPool",100,-5.,5.));
  GEMCSC_Pool_csc_x             = std::unique_ptr<TH1F>(new TH1F("xCSCPool","xCSCPool",100,-5.,5.));
  GEMCSC_Pool_gem_even_x        = std::unique_ptr<TH1F>(new TH1F("xGEMPool_even","xGEMPool even",100,-5.,5.));
  GEMCSC_Pool_csc_even_x        = std::unique_ptr<TH1F>(new TH1F("xCSCPool_even","xCSCPool even",100,-5.,5.));
  GEMCSC_Pool_gem_odd_x         = std::unique_ptr<TH1F>(new TH1F("xGEMPool_odd","xGEMPool odd",100,-5.,5.));
  GEMCSC_Pool_csc_odd_x         = std::unique_ptr<TH1F>(new TH1F("xCSCPool_odd","xCSCPool odd",100,-5.,5.));
  GEMCSC_Pool_cscl1_x = std::unique_ptr<TH1F>(new TH1F("xGEMCSCPool_cscl1","xGEMCSCPool_cscl1",100,-5.,5.));
  GEMCSC_Pool_cscl2_x = std::unique_ptr<TH1F>(new TH1F("xGEMCSCPool_cscl2","xGEMCSCPool_cscl2",100,-5.,5.));
  GEMCSC_Pool_cscl3_x = std::unique_ptr<TH1F>(new TH1F("xGEMCSCPool_cscl3","xGEMCSCPool_cscl3",100,-5.,5.));
  GEMCSC_Pool_cscl4_x = std::unique_ptr<TH1F>(new TH1F("xGEMCSCPool_cscl4","xGEMCSCPool_cscl4",100,-5.,5.));
  GEMCSC_Pool_cscl5_x = std::unique_ptr<TH1F>(new TH1F("xGEMCSCPool_cscl5","xGEMCSCPool_cscl5",100,-5.,5.));
  GEMCSC_Pool_cscl6_x = std::unique_ptr<TH1F>(new TH1F("xGEMCSCPool_cscl6","xGEMCSCPool_cscl6",100,-5.,5.));
  GEMCSC_Pool_geml1_x = std::unique_ptr<TH1F>(new TH1F("xGEMCSCPool_geml1","xGEMCSCPool_geml1",100,-5.,5.));
  GEMCSC_Pool_geml2_x = std::unique_ptr<TH1F>(new TH1F("xGEMCSCPool_geml2","xGEMCSCPool_geml2",100,-5.,5.));
  GEMCSC_Residuals_y        = std::unique_ptr<TH1F>(new TH1F("yGEMCSCRes","yGEMCSCRes",100,-10.,10.));
  GEMCSC_Residuals_gem_y    = std::unique_ptr<TH1F>(new TH1F("yGEMRes","yGEMRes",100,-10.,10.));
  GEMCSC_Residuals_csc_y    = std::unique_ptr<TH1F>(new TH1F("yCSCRes","yCSCRes",100,-5.,5.));
  GEMCSC_Residuals_gem_even_y   = std::unique_ptr<TH1F>(new TH1F("yGEMRes_even","yGEMRes even",100,-10.,10.));
  GEMCSC_Residuals_csc_even_y   = std::unique_ptr<TH1F>(new TH1F("yCSCRes_even","yCSCRes even",100,-5.,5.));
  GEMCSC_Residuals_gem_odd_y    = std::unique_ptr<TH1F>(new TH1F("yGEMRes_odd","yGEMRes odd",100,-10.,10.));
  GEMCSC_Residuals_csc_odd_y    = std::unique_ptr<TH1F>(new TH1F("yCSCRes_odd","yCSCRes odd",100,-5.,5.));
  GEMCSC_Residuals_cscl1_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCRes_cscl1","yGEMCSCRes_cscl1",100,-5.,5.));
  GEMCSC_Residuals_cscl2_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCRes_cscl2","yGEMCSCRes_cscl2",100,-5.,5.));
  GEMCSC_Residuals_cscl3_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCRes_cscl3","yGEMCSCRes_cscl3",100,-5.,5.));
  GEMCSC_Residuals_cscl4_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCRes_cscl4","yGEMCSCRes_cscl4",100,-5.,5.));
  GEMCSC_Residuals_cscl5_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCRes_cscl5","yGEMCSCRes_cscl5",100,-5.,5.));
  GEMCSC_Residuals_cscl6_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCRes_cscl6","yGEMCSCRes_cscl6",100,-5.,5.));
  GEMCSC_Residuals_geml1_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCRes_geml1","yGEMCSCRes_geml1",100,-5.,5.));
  GEMCSC_Residuals_geml2_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCRes_geml2","yGEMCSCRes_geml2",100,-5.,5.));
  GEMCSC_Pool_y            = std::unique_ptr<TH1F>(new TH1F("yGEMCSCPool","yGEMCSCPool",100,-5.,5.));
  GEMCSC_Pool_gem_y        = std::unique_ptr<TH1F>(new TH1F("yGEMPool","yGEMPool",100,-5.,5.));
  GEMCSC_Pool_csc_y        = std::unique_ptr<TH1F>(new TH1F("yCSCPool","yCSCPool",100,-5.,5.));
  GEMCSC_Pool_gem_even_y   = std::unique_ptr<TH1F>(new TH1F("yGEMPool_even","yGEMPool even",100,-5.,5.));
  GEMCSC_Pool_csc_even_y   = std::unique_ptr<TH1F>(new TH1F("yCSCPool_even","yCSCPool even",100,-5.,5.));
  GEMCSC_Pool_gem_odd_y    = std::unique_ptr<TH1F>(new TH1F("yGEMPool_odd","yGEMPool odd",100,-5.,5.));
  GEMCSC_Pool_csc_odd_y    = std::unique_ptr<TH1F>(new TH1F("yCSCPool_odd","yCSCPool odd",100,-5.,5.));
  GEMCSC_Pool_cscl1_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCPool_cscl1","yGEMCSCPool_cscl1",100,-5.,5.));
  GEMCSC_Pool_cscl2_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCPool_cscl2","yGEMCSCPool_cscl2",100,-5.,5.));
  GEMCSC_Pool_cscl3_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCPool_cscl3","yGEMCSCPool_cscl3",100,-5.,5.));
  GEMCSC_Pool_cscl4_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCPool_cscl4","yGEMCSCPool_cscl4",100,-5.,5.));
  GEMCSC_Pool_cscl5_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCPool_cscl5","yGEMCSCPool_cscl5",100,-5.,5.));
  GEMCSC_Pool_cscl6_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCPool_cscl6","yGEMCSCPool_cscl6",100,-5.,5.));
  GEMCSC_Pool_geml1_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCPool_geml1","yGEMCSCPool_geml1",100,-5.,5.));
  GEMCSC_Pool_geml2_y = std::unique_ptr<TH1F>(new TH1F("yGEMCSCPool_geml2","yGEMCSCPool_geml2",100,-5.,5.));
    

  GEMCSC_Pool_gem_x_newE           = std::unique_ptr<TH1F>(new TH1F("xGEMPool_newE","xGEMPool_newE",100,-5.,5.));
  GEMCSC_Pool_gem_y_newE           = std::unique_ptr<TH1F>(new TH1F("yGEMPool_newE","yGEMPool_newE",100,-5.,5.));
  GEMCSC_Pool_gem_odd_x_newE       = std::unique_ptr<TH1F>(new TH1F("xGEMPool_odd_newE","xGEMPool odd newE",100,-5.,5.));
  GEMCSC_Pool_gem_odd_y_newE       = std::unique_ptr<TH1F>(new TH1F("yGEMPool_odd_newE","yGEMPool odd newE",100,-5.,5.));
  GEMCSC_Pool_gem_x_newE_mp        = std::unique_ptr<TH1F>(new TH1F("xGEMPool_newE_mp","xGEMPool_newE muon+",100,-5.,5.));
  GEMCSC_Pool_gem_y_newE_mp        = std::unique_ptr<TH1F>(new TH1F("yGEMPool_newE_mp","yGEMPool_newE muon+",100,-5.,5.));
  GEMCSC_Pool_gem_odd_x_newE_mp    = std::unique_ptr<TH1F>(new TH1F("xGEMPool_odd_newE_mp","xGEMPool odd newE muon+",100,-5.,5.));
  GEMCSC_Pool_gem_odd_y_newE_mp    = std::unique_ptr<TH1F>(new TH1F("yGEMPool_odd_newE_mp","yGEMPool odd newE muon+",100,-5.,5.));
  GEMCSC_Pool_gem_x_newE_mm        = std::unique_ptr<TH1F>(new TH1F("xGEMPool_newE_mm","xGEMPool_newE muon-",100,-5.,5.));
  GEMCSC_Pool_gem_y_newE_mm        = std::unique_ptr<TH1F>(new TH1F("yGEMPool_newE_mm","yGEMPool_newE muon-",100,-5.,5.));
  GEMCSC_Pool_gem_odd_x_newE_mm    = std::unique_ptr<TH1F>(new TH1F("xGEMPool_odd_newE_mm","xGEMPool odd newE muon-",100,-5.,5.));
  GEMCSC_Pool_gem_odd_y_newE_mm    = std::unique_ptr<TH1F>(new TH1F("yGEMPool_odd_newE_mm","yGEMPool odd newE muon-",100,-5.,5.));
  
  GEMCSC_Dphi_min_afterCut              = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHgemcscS_min_afterCut","Dphi_gemRHgemcscS_min_afterCut",800000,-4.,4.));
  GEMCSC_Dtheta_min_afterCut            = std::unique_ptr<TH1F>(new TH1F("Dtheta_gemRHgemcscS_min_afterCut","Dtheta_gemRHgemcscS_min_afterCut",60000,-3.,3.));
  GEMCSC_DR_min_afterCut                = std::unique_ptr<TH1F>(new TH1F("DR_gemRHgemcscS_min_afterCut","DR_gemRHgemcscS_min_afterCut",60000,-3.,3.));
  GEMCSC_Dphi_min_afterCut_odd          = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHgemcscS_min_afterCut_odd","Dphi_gemRHgemcscS_min_afterCut_odd",800000,-4.,4.));
  GEMCSC_Dphi_min_afterCut_even         = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHgemcscS_min_afterCut_even","Dphi_gemRHgemcscS_min_afterCut_even",800000,-4.,4.));
  GEMCSC_Dphi_min_afterCut_l1           = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHgemcscS_min_afterCut_l1","Dphi_gemRHgemcscS_min_afterCut_l1",800000,-4.,4.));
  GEMCSC_Dphi_min_afterCut_l1_odd       = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHgemcscS_min_afterCut_l1_odd","Dphi_gemRHgemcscS_min_afterCut_l1_odd",800000,-4.,4.));
  GEMCSC_Dphi_min_afterCut_l1_even      = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHgemcscS_min_afterCut_l1_even","Dphi_gemRHgemcscS_min_afterCut_l1_even",800000,-4.,4.));
  GEMCSC_Dphi_min_afterCut_l2           = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHgemcscS_min_afterCut_l2","Dphi_gemRHgemcscS_min_afterCut_l2",800000,-4.,4.));
  GEMCSC_Dphi_min_afterCut_l2_odd       = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHgemcscS_min_afterCut_l2_odd","Dphi_gemRHgemcscS_min_afterCut_l2_odd",800000,-4.,4.));
  GEMCSC_Dphi_min_afterCut_l2_even      = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHgemcscS_min_afterCut_l2_even","Dphi_gemRHgemcscS_min_afterCut_l2_even",800000,-4.,4.));
  
  GEMCSC_Dphi_cscS_min_afterCut         = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut","Dphi_gemRHcscS_min_afterCut",800000,-4.,4.));
  GEMCSC_Dtheta_cscS_min_afterCut       = std::unique_ptr<TH1F>(new TH1F("Dtheta_gemRHcscS_min_afterCut","Dtheta_gemRHcscS_min_afterCut",60000,-3.,3.));
  GEMCSC_DR_cscS_min_afterCut           = std::unique_ptr<TH1F>(new TH1F("DR_gemRHcscS_min_afterCut","DR_gemRHcscS_min_afterCut",60000,-3.,3.));
  GEMCSC_Dphi_cscS_min_afterCut_l1      = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_l1","Dphi_gemRHcscS_min_afterCut_l1",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_l1_odd  = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_l1_odd","Dphi_gemRHcscS_min_afterCut_l1_odd",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_l1_even = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_l1_even","Dphi_gemRHcscS_min_afterCut_l1_even",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_l2      = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_l2","Dphi_gemRHcscS_min_afterCut_l2",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_l2_odd  = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_l2_odd","Dphi_gemRHcscS_min_afterCut_l2_odd",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_l2_even = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_l2_even","Dphi_gemRHcscS_min_afterCut_l2_even",800000,-4.,4.));
  
  GEMCSC_Dphi_cscS_min_afterCut_odd    = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_odd","Dphi_gemRHcscS_min_afterCut_odd",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_even   = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_even","Dphi_gemRHcscS_min_afterCut_even",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_odd_r1 = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_odd_r1","Dphi_gemRHcscS_min_afterCut_odd_r1",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_odd_r2 = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_odd_r2","Dphi_gemRHcscS_min_afterCut_odd_r2",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_odd_r3 = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_odd_r3","Dphi_gemRHcscS_min_afterCut_odd_r3",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_odd_r4 = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_odd_r4","Dphi_gemRHcscS_min_afterCut_odd_r4",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_odd_r5 = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_odd_r5","Dphi_gemRHcscS_min_afterCut_odd_r5",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_odd_r6 = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_odd_r6","Dphi_gemRHcscS_min_afterCut_odd_r6",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_odd_r7 = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_odd_r7","Dphi_gemRHcscS_min_afterCut_odd_r7",800000,-4.,4.));
  GEMCSC_Dphi_cscS_min_afterCut_odd_r8 = std::unique_ptr<TH1F>(new TH1F("Dphi_gemRHcscS_min_afterCut_odd_r8","Dphi_gemRHcscS_min_afterCut_odd_r8",800000,-4.,4.));
  GEMCSC_Dtheta_cscS_min_afterCut_odd  = std::unique_ptr<TH1F>(new TH1F("Dtheta_gemRHcscS_min_afterCut_odd","Dtheta_gemRHcscS_min_afterCut_odd",60000,-3.,3.));
  GEMCSC_Dtheta_cscS_min_afterCut_even = std::unique_ptr<TH1F>(new TH1F("Dtheta_gemRHcscS_min_afterCut_even","Dtheta_gemRHcscS_min_afterCut_even",60000,-3.,3.));
  
  
  SIMGEMCSC_Dphi_cscS_min_afterCut        = std::unique_ptr<TH1F>(new TH1F("Dphi_gemSHcscS_min_afterCut","Dphi_gemSHcscS_min_afterCut",800000,-4.,4.));
  SIMGEMCSC_Dtheta_cscS_min_afterCut      = std::unique_ptr<TH1F>(new TH1F("Dtheta_gemSHcscS_min_afterCut","Dtheta_gemSHcscS_min_afterCut",60000,-3.,3.));
  SIMGEMCSC_Dphi_cscS_min_afterCut_odd    = std::unique_ptr<TH1F>(new TH1F("Dphi_gemSHcscS_min_afterCut_odd","Dphi_gemSHcscS_min_afterCut_odd",800000,-4.,4.));
  SIMGEMCSC_Dphi_cscS_min_afterCut_even   = std::unique_ptr<TH1F>(new TH1F("Dphi_gemSHcscS_min_afterCut_even","Dphi_gemSHcscS_min_afterCut_even",800000,-4.,4.));
  SIMGEMCSC_Dtheta_cscS_min_afterCut_odd  = std::unique_ptr<TH1F>(new TH1F("Dtheta_gemSHcscS_min_afterCut_odd","Dtheta_gemSHcscS_min_afterCut_odd",60000,-3.,3.));
  SIMGEMCSC_Dtheta_cscS_min_afterCut_even = std::unique_ptr<TH1F>(new TH1F("Dtheta_gemSHcscS_min_afterCut_even","Dtheta_gemSHcscS_min_afterCut_even",60000,-3.,3.));
  
  SIMGEMCSC_Dphi_SS_min_afterCut        = std::unique_ptr<TH1F>(new TH1F("Dphi_gemSHSS_min_afterCut","Dphi_gemSHSS_min_afterCut",800000,-4.,4.));
  SIMGEMCSC_Dtheta_SS_min_afterCut      = std::unique_ptr<TH1F>(new TH1F("Dtheta_gemSHSS_min_afterCut","Dtheta_gemSHSS_min_afterCut",60000,-3.,3.));
  SIMGEMCSC_Dphi_SS_min_afterCut_odd    = std::unique_ptr<TH1F>(new TH1F("Dphi_gemSHSS_min_afterCut_odd","Dphi_gemSHSS_min_afterCut_odd",800000,-4.,4.));
  SIMGEMCSC_Dphi_SS_min_afterCut_even   = std::unique_ptr<TH1F>(new TH1F("Dphi_gemSHSS_min_afterCut_even","Dphi_gemSHSS_min_afterCut_even",800000,-4.,4.));
  SIMGEMCSC_Dtheta_SS_min_afterCut_odd  = std::unique_ptr<TH1F>(new TH1F("Dtheta_gemSHSS_min_afterCut_odd","Dtheta_gemSHSS_min_afterCut_odd",60000,-3.,3.));
  SIMGEMCSC_Dtheta_SS_min_afterCut_even = std::unique_ptr<TH1F>(new TH1F("Dtheta_gemSHSS_min_afterCut_even","Dtheta_gemSHSS_min_afterCut_even",60000,-3.,3.));
  
  SIMGEMCSC_theta_cscSsh_vs_ndof_odd    = std::unique_ptr<TH2F>(new TH2F("SIMGEMCSCSegm_theta_cscSsh_vs_ndof_odd","theta_cscSsh vs ndof odd",30000,0.,3.,15,-0.5,14.5));
  SIMGEMCSC_theta_cscSsh_vs_ndof_even   = std::unique_ptr<TH2F>(new TH2F("SIMGEMCSCSegm_theta_cscSsh_vs_ndof_even","theta_cscSsh vs ndof even",30000,0.,3.,15,-0.5,14.5));
  
  SIMGEMCSC_Residuals_gem_x      = std::unique_ptr<TH1F>(new TH1F("xGEMRes_simhit","xGEMRes",100,-0.5,0.5));
  SIMGEMCSC_Residuals_gem_y      = std::unique_ptr<TH1F>(new TH1F("yGEMRes_simhit","yGEMRes",100,-10.,10.));
  SIMGEMCSC_Pool_gem_x_newE      = std::unique_ptr<TH1F>(new TH1F("xGEMPool_newE_simhit","xGEMPool newE",100,-10.,10.));
  SIMGEMCSC_Pool_gem_y_newE      = std::unique_ptr<TH1F>(new TH1F("yGEMPool_newE_simhit","yGEMPool newE",100,-10.,10.));
  SIMGEMCSC_Residuals_gem_odd_x  = std::unique_ptr<TH1F>(new TH1F("xGEMRes_odd_simhit","xGEMRes",100,-0.5,0.5));
  SIMGEMCSC_Residuals_gem_odd_y  = std::unique_ptr<TH1F>(new TH1F("yGEMRes_odd_simhit","yGEMRes",100,-10.,10.));
  SIMGEMCSC_Pool_gem_x_newE_odd  = std::unique_ptr<TH1F>(new TH1F("xGEMPool_odd_newE_simhit","xGEMPool newE",100,-10.,10.));
  SIMGEMCSC_Pool_gem_y_newE_odd  = std::unique_ptr<TH1F>(new TH1F("yGEMPool_odd_newE_simhit","yGEMPool newE",100,-10.,10.));
  SIMGEMCSC_Residuals_gem_even_x = std::unique_ptr<TH1F>(new TH1F("xGEMRes_even_simhit","xGEMRes",100,-0.5,0.5));
  SIMGEMCSC_Residuals_gem_even_y = std::unique_ptr<TH1F>(new TH1F("yGEMRes_even_simhit","yGEMRes",100,-10.,10.));
  SIMGEMCSC_Pool_gem_x_newE_even = std::unique_ptr<TH1F>(new TH1F("xGEMPool_even_newE_simhit","xGEMPool newE",100,-10.,10.));
  SIMGEMCSC_Pool_gem_y_newE_even = std::unique_ptr<TH1F>(new TH1F("yGEMPool_even_newE_simhit","yGEMPool newE",100,-10.,10.));
  
  SIMGEMCSC_Residuals_gem_rhsh_x = std::unique_ptr<TH1F>(new TH1F("xGEMRes_shrh","xGEMRes",100,-0.5,0.5));
  SIMGEMCSC_Residuals_gem_rhsh_y = std::unique_ptr<TH1F>(new TH1F("yGEMRes_shrh","yGEMRes",100,-10.,10.));
  SIMGEMCSC_Pool_gem_rhsh_x_newE = std::unique_ptr<TH1F>(new TH1F("xGEMPool_shrh","xGEMPool sh rh",100,-5.,5.));
  SIMGEMCSC_Pool_gem_rhsh_y_newE = std::unique_ptr<TH1F>(new TH1F("yGEMPool_shrh","yGEMPool sh rh",100,-5.,5.));
  
}


TestGEMCSCSegmentAnalyzer::~TestGEMCSCSegmentAnalyzer()
{
  SIM_etaVScharge->Write();
  SIM_etaVStype->Write();
  
  CSC_fitchi2->Write();
  GEMCSC_fitchi2->Write();
  GEMCSC_fitchi2_odd->Write();
  GEMCSC_fitchi2_even->Write();
  GEMCSC_NumGEMRH->Write();
  GEMCSC_NumCSCRH->Write();
  GEMCSC_NumGEMCSCRH->Write();
  GEMCSC_NumGEMCSCSeg->Write();
  GEMCSC_SSegm_LPx->Write();
  GEMCSC_SSegm_LPy->Write();
  GEMCSC_SSegm_LPEx->Write();
  GEMCSC_SSegm_LPEy->Write();
  //GEMCSC_SSegm_LPEz->Write();
  GEMCSC_SSegm_LDx->Write();
  GEMCSC_SSegm_LDy->Write();
  GEMCSC_SSegm_LDEx->Write();
  GEMCSC_SSegm_LDEy->Write();
  //GEMCSC_SSegm_LDEz->Write();
  GEMCSC_CSCSegm_LPx->Write();
  GEMCSC_CSCSegm_LPy->Write();
  GEMCSC_CSCSegm_LPEx->Write();
  GEMCSC_CSCSegm_LPEy->Write();
  //GEMCSC_CSCSegm_LPEz->Write();
  GEMCSC_CSCSegm_LDx->Write();
  GEMCSC_CSCSegm_LDy->Write();
  GEMCSC_CSCSegm_LDEx->Write();
  GEMCSC_CSCSegm_LDEy->Write();
  //GEMCSC_CSCSegm_LDEz->Write();
  GEMCSC_SSegm_LDEy_vs_ndof->Write();
  GEMCSC_SSegm_LPEy_vs_ndof->Write();
  GEMCSC_CSCSegm_LDEy_vs_ndof->Write();
  GEMCSC_CSCSegm_LPEy_vs_ndof->Write();
  SIMGEMCSC_SSegm_LDx->Write();
  SIMGEMCSC_SSegm_LDy->Write();
  SIMGEMCSC_SSegm_LDEx->Write();
  SIMGEMCSC_SSegm_LDEy->Write();
  GEMCSC_Residuals_x->Write();
  GEMCSC_Residuals_gem_x->Write();
  GEMCSC_Residuals_csc_x->Write();
  GEMCSC_Residuals_gem_even_x->Write();
  GEMCSC_Residuals_csc_even_x->Write();
  GEMCSC_Residuals_gem_odd_x->Write();
  GEMCSC_Residuals_csc_odd_x->Write();
  GEMCSC_Residuals_cscl1_x->Write();
  GEMCSC_Residuals_cscl2_x->Write();
  GEMCSC_Residuals_cscl3_x->Write();
  GEMCSC_Residuals_cscl4_x->Write();
  GEMCSC_Residuals_cscl5_x->Write();
  GEMCSC_Residuals_cscl6_x->Write();
  GEMCSC_Residuals_geml1_x->Write();
  GEMCSC_Residuals_geml2_x->Write();
  GEMCSC_Pool_x->Write();
  GEMCSC_Pool_gem_x->Write();
  GEMCSC_Pool_csc_x->Write();
  GEMCSC_Pool_gem_even_x->Write();
  GEMCSC_Pool_csc_even_x->Write();
  GEMCSC_Pool_gem_odd_x->Write();
  GEMCSC_Pool_csc_odd_x->Write();
  GEMCSC_Pool_cscl1_x->Write();
  GEMCSC_Pool_cscl2_x->Write();
  GEMCSC_Pool_cscl3_x->Write();
  GEMCSC_Pool_cscl4_x->Write();
  GEMCSC_Pool_cscl5_x->Write();
  GEMCSC_Pool_cscl6_x->Write();
  GEMCSC_Pool_geml1_x->Write();
  GEMCSC_Pool_geml2_x->Write();
  GEMCSC_Residuals_y->Write();
  GEMCSC_Residuals_gem_y->Write();
  GEMCSC_Residuals_csc_y->Write();
  GEMCSC_Residuals_gem_even_y->Write();
  GEMCSC_Residuals_csc_even_y->Write();
  GEMCSC_Residuals_gem_odd_y->Write();
  GEMCSC_Residuals_csc_odd_y->Write();
  GEMCSC_Residuals_cscl1_y->Write();
  GEMCSC_Residuals_cscl2_y->Write();
  GEMCSC_Residuals_cscl3_y->Write();
  GEMCSC_Residuals_cscl4_y->Write();
  GEMCSC_Residuals_cscl5_y->Write();
  GEMCSC_Residuals_cscl6_y->Write();
  GEMCSC_Residuals_geml1_y->Write();
  GEMCSC_Residuals_geml2_y->Write();
  GEMCSC_Pool_y->Write();
  GEMCSC_Pool_gem_y->Write();
  GEMCSC_Pool_csc_y->Write();
  GEMCSC_Pool_gem_even_y->Write();
  GEMCSC_Pool_csc_even_y->Write();
  GEMCSC_Pool_gem_odd_y->Write();
  GEMCSC_Pool_csc_odd_y->Write();
  GEMCSC_Pool_cscl1_y->Write();
  GEMCSC_Pool_cscl2_y->Write();
  GEMCSC_Pool_cscl3_y->Write();
  GEMCSC_Pool_cscl4_y->Write();
  GEMCSC_Pool_cscl5_y->Write();
  GEMCSC_Pool_cscl6_y->Write();
  GEMCSC_Pool_geml1_y->Write();
  GEMCSC_Pool_geml2_y->Write();
  GEMCSC_Pool_gem_x_newE->Write();
  GEMCSC_Pool_gem_y_newE->Write();
  GEMCSC_Pool_gem_odd_x_newE->Write();
  GEMCSC_Pool_gem_odd_y_newE->Write();
  GEMCSC_Pool_gem_x_newE_mp->Write();
  GEMCSC_Pool_gem_y_newE_mp->Write();
  GEMCSC_Pool_gem_odd_x_newE_mp->Write();
  GEMCSC_Pool_gem_odd_y_newE_mp->Write();
  GEMCSC_Pool_gem_x_newE_mm->Write();
  GEMCSC_Pool_gem_y_newE_mm->Write();
  GEMCSC_Pool_gem_odd_x_newE_mm->Write();
  GEMCSC_Pool_gem_odd_y_newE_mm->Write();
  
  GEMCSC_SSegm_xe_l1->Write();
  GEMCSC_SSegm_ye_l1->Write();
  GEMCSC_SSegm_ze_l1->Write();
  GEMCSC_SSegm_sigmaxe_l1->Write();
  GEMCSC_SSegm_sigmaye_l1->Write();
  GEMCSC_SSegm_xe_l1_odd->Write();
  GEMCSC_SSegm_ye_l1_odd->Write();
  GEMCSC_SSegm_ze_l1_odd->Write();
  GEMCSC_SSegm_sigmaxe_l1_odd->Write();
  GEMCSC_SSegm_sigmaye_l1_odd->Write();
  GEMCSC_SSegm_xe_l1_even->Write();
  GEMCSC_SSegm_ye_l1_even->Write();
  GEMCSC_SSegm_ze_l1_even->Write();
  GEMCSC_SSegm_sigmaxe_l1_even->Write();
  GEMCSC_SSegm_sigmaye_l1_even->Write();
  
  GEMCSC_SSegm_xe_l2->Write();
  GEMCSC_SSegm_ye_l2->Write();
  GEMCSC_SSegm_ze_l2->Write();
  GEMCSC_SSegm_sigmaxe_l2->Write();
  GEMCSC_SSegm_sigmaye_l2->Write();
  GEMCSC_SSegm_xe_l2_odd->Write();
  GEMCSC_SSegm_ye_l2_odd->Write();
  GEMCSC_SSegm_ze_l2_odd->Write();
  GEMCSC_SSegm_sigmaxe_l2_odd->Write();
  GEMCSC_SSegm_sigmaye_l2_odd->Write();
  GEMCSC_SSegm_xe_l2_even->Write();
  GEMCSC_SSegm_ye_l2_even->Write();
  GEMCSC_SSegm_ze_l2_even->Write();
  GEMCSC_SSegm_sigmaxe_l2_even->Write();
  GEMCSC_SSegm_sigmaye_l2_even->Write();
  
    
  GEMCSC_CSCSegm_xe_l1->Write();
  GEMCSC_CSCSegm_ye_l1->Write();
  GEMCSC_CSCSegm_ze_l1->Write();
  GEMCSC_CSCSegm_sigmaxe_l1->Write();
  GEMCSC_CSCSegm_sigmaye_l1->Write();
  GEMCSC_CSCSegm_xe_l1_odd->Write();
  GEMCSC_CSCSegm_ye_l1_odd->Write();
  GEMCSC_CSCSegm_ze_l1_odd->Write();
  GEMCSC_CSCSegm_sigmaxe_l1_odd->Write();
  GEMCSC_CSCSegm_sigmaye_l1_odd->Write();
  GEMCSC_CSCSegm_xe_l1_even->Write();
  GEMCSC_CSCSegm_ye_l1_even->Write();
  GEMCSC_CSCSegm_ze_l1_even->Write();
  GEMCSC_CSCSegm_sigmaxe_l1_even->Write();
  GEMCSC_CSCSegm_sigmaye_l1_even->Write();
  
  GEMCSC_CSCSegm_xe_l2->Write();
  GEMCSC_CSCSegm_ye_l2->Write();
  GEMCSC_CSCSegm_ze_l2->Write();
  GEMCSC_CSCSegm_sigmaxe_l2->Write();
  GEMCSC_CSCSegm_sigmaye_l2->Write();
  GEMCSC_CSCSegm_xe_l2_odd->Write();
  GEMCSC_CSCSegm_ye_l2_odd->Write();
  GEMCSC_CSCSegm_ze_l2_odd->Write();
  GEMCSC_CSCSegm_sigmaxe_l2_odd->Write();
  GEMCSC_CSCSegm_sigmaye_l2_odd->Write();
  GEMCSC_CSCSegm_xe_l2_even->Write();
  GEMCSC_CSCSegm_ye_l2_even->Write();
  GEMCSC_CSCSegm_ze_l2_even->Write();
  GEMCSC_CSCSegm_sigmaxe_l2_even->Write();
  GEMCSC_CSCSegm_sigmaye_l2_even->Write();
  
  
  GEMCSC_Dphi_min_afterCut->Write();
  GEMCSC_Dtheta_min_afterCut->Write();
  GEMCSC_Dphi_min_afterCut_odd->Write();
  GEMCSC_Dphi_min_afterCut_even->Write();
  GEMCSC_DR_min_afterCut->Write();
  GEMCSC_Dphi_min_afterCut_l1->Write();
  GEMCSC_Dphi_min_afterCut_l1_odd->Write();
  GEMCSC_Dphi_min_afterCut_l1_even->Write();
  GEMCSC_Dphi_min_afterCut_l2->Write();
  GEMCSC_Dphi_min_afterCut_l2_odd->Write();
  GEMCSC_Dphi_min_afterCut_l2_even->Write();
  
  GEMCSC_Dphi_cscS_min_afterCut->Write();
  GEMCSC_Dtheta_cscS_min_afterCut->Write();
  GEMCSC_Dtheta_cscS_min_afterCut_odd->Write();
  GEMCSC_Dtheta_cscS_min_afterCut_even->Write();
  GEMCSC_DR_cscS_min_afterCut->Write();
  
  GEMCSC_Dphi_cscS_min_afterCut_l1->Write();
  GEMCSC_Dphi_cscS_min_afterCut_l1_odd->Write();
  GEMCSC_Dphi_cscS_min_afterCut_l1_even->Write();
  GEMCSC_Dphi_cscS_min_afterCut_l2->Write();
  GEMCSC_Dphi_cscS_min_afterCut_l2_odd->Write();
  GEMCSC_Dphi_cscS_min_afterCut_l2_even->Write();
  
  GEMCSC_Dphi_cscS_min_afterCut_odd->Write();
  GEMCSC_Dphi_cscS_min_afterCut_even->Write();
  GEMCSC_Dphi_cscS_min_afterCut_odd_r1->Write();
  GEMCSC_Dphi_cscS_min_afterCut_odd_r2->Write();
  GEMCSC_Dphi_cscS_min_afterCut_odd_r3->Write();
  GEMCSC_Dphi_cscS_min_afterCut_odd_r4->Write();
  GEMCSC_Dphi_cscS_min_afterCut_odd_r5->Write();
  GEMCSC_Dphi_cscS_min_afterCut_odd_r6->Write();
  GEMCSC_Dphi_cscS_min_afterCut_odd_r7->Write();
  GEMCSC_Dphi_cscS_min_afterCut_odd_r8->Write();
  
  
  SIMGEMCSC_Dphi_SS_min_afterCut->Write();
  SIMGEMCSC_Dtheta_SS_min_afterCut->Write();
  SIMGEMCSC_Dtheta_SS_min_afterCut_odd->Write();
  SIMGEMCSC_Dtheta_SS_min_afterCut_even->Write();
  SIMGEMCSC_Dphi_SS_min_afterCut_odd->Write();
  SIMGEMCSC_Dphi_SS_min_afterCut_even->Write();
  
  SIMGEMCSC_Dphi_cscS_min_afterCut->Write();
  SIMGEMCSC_Dtheta_cscS_min_afterCut->Write();
  SIMGEMCSC_Dtheta_cscS_min_afterCut_odd->Write();
  SIMGEMCSC_Dtheta_cscS_min_afterCut_even->Write();
  SIMGEMCSC_Dphi_cscS_min_afterCut_odd->Write();
  SIMGEMCSC_Dphi_cscS_min_afterCut_even->Write();
  SIMGEMCSC_Residuals_gem_x->Write();
  SIMGEMCSC_Residuals_gem_y->Write();
  SIMGEMCSC_Pool_gem_x_newE->Write();
  SIMGEMCSC_Pool_gem_y_newE->Write();
  SIMGEMCSC_Residuals_gem_odd_x->Write();
  SIMGEMCSC_Residuals_gem_odd_y->Write();
  SIMGEMCSC_Pool_gem_x_newE_odd->Write();
  SIMGEMCSC_Pool_gem_y_newE_odd->Write();
  SIMGEMCSC_Residuals_gem_even_x->Write();
  SIMGEMCSC_Residuals_gem_even_y->Write();
  SIMGEMCSC_Pool_gem_x_newE_even->Write();
  SIMGEMCSC_Pool_gem_y_newE_even->Write();
  
  SIMGEMCSC_Residuals_gem_rhsh_x->Write();
  SIMGEMCSC_Residuals_gem_rhsh_y->Write();
  SIMGEMCSC_Pool_gem_rhsh_x_newE->Write();
  SIMGEMCSC_Pool_gem_rhsh_y_newE->Write();
  
  SIMGEMCSC_theta_cscSsh_vs_ndof_odd->Write();
  SIMGEMCSC_theta_cscSsh_vs_ndof_even->Write();
    
}

// ------------ method called for each event  ------------
void
TestGEMCSCSegmentAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    iSetup.get<MuonGeometryRecord>().get(gemGeom);
    iSetup.get<MuonGeometryRecord>().get(cscGeom);
    const CSCGeometry* cscGeom_ = &*cscGeom;
 
 // ================
 // Sim Tracks
 // ================
    edm::Handle<edm::SimTrackContainer> simTracks;
    iEvent.getByToken(SimTrack_Token, simTracks);
    // edm::Handle<edm::SimTrackContainer> simTracks;
    // iEvent.getByLabel("g4SimHits",simTracks);



    edm::SimTrackContainer::const_iterator simTrack;

    for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack){
      if (!(abs((*simTrack).type()) == 13)) continue;
      double simEta = (*simTrack).momentum().eta();
      //double simPhi = (*simTrack).momentum().phi();
      int qGen = simTrack->charge();
      SIM_etaVScharge->Fill(simEta,qGen);
      SIM_etaVStype->Fill(simEta,(*simTrack).type());
    }
    
  // ================
  // CSC Segments
  // ================
  edm::Handle<CSCSegmentCollection> cscSegment;
  iEvent.getByToken(CSCSegment_Token, cscSegment);
  // edm::Handle<CSCSegmentCollection> cscSegment;
  // iEvent.getByLabel("cscSegments","",cscSegment);

  for (auto cscs = cscSegment->begin(); cscs != cscSegment->end(); cscs++) {
    CSCDetId CSCId = cscs->cscDetId();
    if(!(CSCId.station() == 1 && (CSCId.ring() == 1 || CSCId.ring() == 4))) continue;

    auto cscrhs = cscs->specificRecHits();
  }


  // ================
  // GEMCSC Segments
  // ================
  edm::Handle<GEMCSCSegmentCollection> gemcscSegment;
  iEvent.getByToken(GEMCSCSegment_Token, gemcscSegment);
  // edm::Handle<GEMCSCSegmentCollection> gemcscSegment;
  // iEvent.getByLabel("gemcscSegments","", gemcscSegment);
  // iEvent.getByLabel("gemcscSegments","", "RECO", gemcscSegment);
  // iEvent.getByLabel("gemcscSegments", "", "GEMCSCREC", gemcscSegment);

  if(gemcscSegment->size()!=0)GEMCSC_NumGEMCSCSeg->Fill(gemcscSegment->size());

    for (auto gemcscs = gemcscSegment->begin(); gemcscs != gemcscSegment->end(); gemcscs++) {

      auto gemrhs_if = gemcscs->gemRecHits();
      if(debug) 
	{ 	
	  std::cout<<"GEM-CSC Segment with "<<gemcscs->gemRecHits().size()<<" GEM rechits and "<<gemcscs->cscSegment().specificRecHits().size()<<" CSC rechits"<<std::endl;
	}
      if(gemrhs_if.size()==0) continue;
      // if(gemrhs_if.size()!=0) continue;

      // --- some printout for debug -----------------------------------------------------------------------------------------------------------------------------------
      if(debug) {
	std::cout<<"GEM-CSC Segment with "<<gemcscs->gemRecHits().size()<<" GEM rechits and "<<gemcscs->cscSegment().specificRecHits().size()<<" CSC rechits"<<std::endl;
	auto gemrhs = gemcscs->gemRecHits();
	for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){
	  GEMDetId gemId((*rh).geographicalId());
	  const GEMEtaPartition* id_etapart = gemGeom->etaPartition(gemId);
	  const BoundPlane & GEMSurface = id_etapart->surface();
	  GlobalPoint GEMGlobalPoint = GEMSurface.toGlobal((*rh).localPosition());
	  std::cout<<"GEM Rechit in "<<gemId<<" = "<<gemId.rawId()<<" at X = "<<GEMGlobalPoint.x()<<" Y = "<<GEMGlobalPoint.y()<<" Z = "<<GEMGlobalPoint.z()<<std::endl;
	}
	CSCSegment cscSeg = gemcscs->cscSegment();
	auto cscrhs = cscSeg.specificRecHits();
	for (auto rh = cscrhs.begin(); rh!= cscrhs.end(); rh++){
	  CSCDetId cscId = (CSCDetId)(*rh).cscDetId();
	  GlobalPoint CSCGlobalPoint = cscGeom->idToDet(cscId)->toGlobal((*rh).localPosition());
	  std::cout<<"CSC Rechit in "<<cscId<<" = "<<cscId.rawId()<<" at X = "<<CSCGlobalPoint.x()<<" Y = "<<CSCGlobalPoint.y()<<" Z = "<<CSCGlobalPoint.z()<<std::endl;
	}
      }
      // --- some printout for debug -----------------------------------------------------------------------------------------------------------------------------------



        
        ///////// GEMCSC seg //////////////////////////////////////
        CSCDetId gemcscId = gemcscs->cscDetId();
        const CSCChamber* cscChamber = cscGeom_->chamber(gemcscId);

        auto gemcscsegLP = gemcscs->localPosition();
        auto gemcscsegLD = gemcscs->localDirection();
        auto gemcscsegLEP = gemcscs->localPositionError();
        auto gemcscsegLED = gemcscs->localDirectionError();
        GEMCSC_SSegm_LPx->Fill(gemcscsegLP.x());
        GEMCSC_SSegm_LPy->Fill(gemcscsegLP.y());
        GEMCSC_SSegm_LPEx->Fill(sqrt(gemcscsegLEP.xx()));
        GEMCSC_SSegm_LPEy->Fill(sqrt(gemcscsegLEP.yy()));
        //GEMCSC_SSegm_LPEz->Fill(gemcscsegLEP.zz());
        GEMCSC_SSegm_LDx->Fill(gemcscsegLD.x());
        GEMCSC_SSegm_LDy->Fill(gemcscsegLD.y());
        GEMCSC_SSegm_LDEx->Fill(sqrt(gemcscsegLED.xx()));
        GEMCSC_SSegm_LDEy->Fill(sqrt(gemcscsegLED.yy()));
        //GEMCSC_SSegm_LDEz->Fill(gemcscsegLED.zz());
        GEMCSC_SSegm_LPEy_vs_ndof->Fill(gemcscsegLEP.yy(),gemcscs->degreesOfFreedom());
        GEMCSC_SSegm_LDEy_vs_ndof->Fill(gemcscsegLED.yy(),gemcscs->degreesOfFreedom());
        
        GEMCSC_fitchi2->Fill(gemcscs->chi2()/gemcscs->degreesOfFreedom());
        
        CSCDetId id((*gemcscs).geographicalId());
        int chamber = id.chamber();
        if(chamber%2!=0){
        GEMCSC_fitchi2_odd->Fill(gemcscs->chi2()/gemcscs->degreesOfFreedom());}
        else{GEMCSC_fitchi2_even->Fill(gemcscs->chi2()/gemcscs->degreesOfFreedom());}
      
        GEMCSC_NumGEMCSCRH->Fill(gemcscs->cscSegment().specificRecHits().size()+gemcscs->gemRecHits().size());
      
        ///////// CSC seg /////////////////////////////////////
        CSCSegment cscSeg = gemcscs->cscSegment();
        //CSCDetId CSCId_new = cscSeg.cscDetId();
        //if(!(CSCId_new.station() == 1 && CSCId_new.ring() == 1)) continue;
        auto cscrhs = cscSeg.specificRecHits();
        GEMCSC_NumCSCRH->Fill(cscrhs.size());
        
        auto cscsegLP = cscSeg.localPosition();
        auto cscsegLD = cscSeg.localDirection();
        auto cscsegLEP = cscSeg.localPositionError();
        auto cscsegLED = cscSeg.localDirectionError();
        
        GEMCSC_CSCSegm_LPx->Fill(cscsegLP.x());
        GEMCSC_CSCSegm_LPy->Fill(cscsegLP.y());
        GEMCSC_CSCSegm_LPEx->Fill(sqrt(cscsegLEP.xx()));
        GEMCSC_CSCSegm_LPEy->Fill(sqrt(cscsegLEP.yy()));
        //GEMCSC_CSCSegm_LPEz->Fill(cscsegLEP.zz());
        GEMCSC_CSCSegm_LDx->Fill(cscsegLD.x());
        GEMCSC_CSCSegm_LDy->Fill(cscsegLD.y());
        GEMCSC_CSCSegm_LDEx->Fill(sqrt(cscsegLED.xx()));
        GEMCSC_CSCSegm_LDEy->Fill(sqrt(cscsegLED.yy()));
        //GEMCSC_CSCSegm_LDEz->Fill(cscsegLED.zz());
        GEMCSC_CSCSegm_LPEy_vs_ndof->Fill(cscsegLEP.yy(),cscSeg.degreesOfFreedom());
        GEMCSC_CSCSegm_LDEy_vs_ndof->Fill(cscsegLED.yy(),cscSeg.degreesOfFreedom());
        
        CSC_fitchi2->Fill(cscSeg.chi2()/cscSeg.degreesOfFreedom());
      
        //////////////////////// CSC RH
        for (auto rh = cscrhs.begin(); rh!= cscrhs.end(); rh++){
            //CSCDetId cscrhId = rh.cscDetId();
            CSCDetId cscrhId = (CSCDetId)(*rh).cscDetId();
            const CSCLayer* cscrhRef = cscGeom_->layer( cscrhId );
          
            auto cscrhLP = rh->localPosition();
            auto cscrhLEP = rh->localPositionError();
            auto cscrhGP = cscrhRef->toGlobal(cscrhLP);
            auto cscrhLP_inSegmRef = cscChamber->toLocal(cscrhGP);
            float xe  = gemcscsegLP.x()+gemcscsegLD.x()*cscrhLP_inSegmRef.z()/gemcscsegLD.z();
            float ye  = gemcscsegLP.y()+gemcscsegLD.y()*cscrhLP_inSegmRef.z()/gemcscsegLD.z();
            float ze = cscrhLP_inSegmRef.z();
            LocalPoint extrPoint(xe,ye,ze); // in segment rest frame
            //float sigma_xe = gemcscsegLEP.xx()+gemcscsegLED.xx()*cscrhLP_inSegmRef.z()*cscrhLP_inSegmRef.z();
            //float sigma_ye = gemcscsegLEP.yy()+gemcscsegLED.yy()*cscrhLP_inSegmRef.z()*cscrhLP_inSegmRef.z();
            auto extSegm = cscrhRef->toLocal(cscChamber->toGlobal(extrPoint)); // in layer restframe
          
            GEMCSC_Residuals_x->Fill(cscrhLP.x()-extSegm.x());
            GEMCSC_Residuals_y->Fill(cscrhLP.y()-extSegm.y());
            GEMCSC_Pool_x->Fill((cscrhLP.x()-extSegm.x())/sqrt(cscrhLEP.xx()));
            GEMCSC_Pool_y->Fill((cscrhLP.y()-extSegm.y())/sqrt(cscrhLEP.yy()));
            GEMCSC_Residuals_csc_x->Fill(cscrhLP.x()-extSegm.x());
            GEMCSC_Residuals_csc_y->Fill(cscrhLP.y()-extSegm.y());
            GEMCSC_Pool_csc_x->Fill((cscrhLP.x()-extSegm.x())/sqrt(cscrhLEP.xx()));
            GEMCSC_Pool_csc_y->Fill((cscrhLP.y()-extSegm.y())/sqrt(cscrhLEP.yy()));

            CSCDetId id((*rh).geographicalId());
            int chamber = id.chamber();
            if(chamber%2!=0){
              //std::cout<<"camera dispari"<<chamber<<std::endl;
              GEMCSC_Residuals_csc_odd_x->Fill(cscrhLP.x()-extSegm.x());
              GEMCSC_Residuals_csc_odd_y->Fill(cscrhLP.y()-extSegm.y());
              GEMCSC_Pool_csc_odd_x->Fill((cscrhLP.x()-extSegm.x())/sqrt(cscrhLEP.xx()));
              GEMCSC_Pool_csc_odd_y->Fill((cscrhLP.y()-extSegm.y())/sqrt(cscrhLEP.yy()));
            }
            else {
              GEMCSC_Residuals_csc_even_x->Fill(cscrhLP.x()-extSegm.x());
              GEMCSC_Residuals_csc_even_y->Fill(cscrhLP.y()-extSegm.y());
              GEMCSC_Pool_csc_even_x->Fill((cscrhLP.x()-extSegm.x())/sqrt(cscrhLEP.xx()));
              GEMCSC_Pool_csc_even_y->Fill((cscrhLP.y()-extSegm.y())/sqrt(cscrhLEP.yy()));

            }
          
            switch (cscrhId.layer()){
              case 1:
                  GEMCSC_Residuals_cscl1_x->Fill(cscrhLP.x()-extSegm.x());
                  GEMCSC_Residuals_cscl1_y->Fill(cscrhLP.y()-extSegm.y());
                  GEMCSC_Pool_cscl1_x->Fill((cscrhLP.x()-extSegm.x())/sqrt(cscrhLEP.xx()));
                  GEMCSC_Pool_cscl1_y->Fill((cscrhLP.x()-extSegm.y())/sqrt(cscrhLEP.yy()));
                  break;
              case 2:
                  GEMCSC_Residuals_cscl2_x->Fill(cscrhLP.x()-extSegm.x());
                  GEMCSC_Residuals_cscl2_y->Fill(cscrhLP.y()-extSegm.y());
                  GEMCSC_Pool_cscl2_x->Fill((cscrhLP.x()-extSegm.x())/sqrt(cscrhLEP.xx()));
                  GEMCSC_Pool_cscl2_y->Fill((cscrhLP.x()-extSegm.y())/sqrt(cscrhLEP.yy()));
                  break;
                  
              case 3:
                  GEMCSC_Residuals_cscl3_x->Fill(cscrhLP.x()-extSegm.x());
                  GEMCSC_Residuals_cscl3_y->Fill(cscrhLP.y()-extSegm.y());
                  GEMCSC_Pool_cscl3_x->Fill((cscrhLP.x()-extSegm.x())/sqrt(cscrhLEP.xx()));
                  GEMCSC_Pool_cscl3_y->Fill((cscrhLP.x()-extSegm.y())/sqrt(cscrhLEP.yy()));
                  break;
              case 4:
                  GEMCSC_Residuals_cscl4_x->Fill(cscrhLP.x()-extSegm.x());
                  GEMCSC_Residuals_cscl4_y->Fill(cscrhLP.y()-extSegm.y());
                  GEMCSC_Pool_cscl4_x->Fill((cscrhLP.x()-extSegm.x())/sqrt(cscrhLEP.xx()));
                  GEMCSC_Pool_cscl4_y->Fill((cscrhLP.x()-extSegm.y())/sqrt(cscrhLEP.yy()));
                  break;
              case 5:
                  GEMCSC_Residuals_cscl5_x->Fill(cscrhLP.x()-extSegm.x());
                  GEMCSC_Residuals_cscl5_y->Fill(cscrhLP.y()-extSegm.y());
                  GEMCSC_Pool_cscl5_x->Fill((cscrhLP.x()-extSegm.x())/sqrt(cscrhLEP.xx()));
                  GEMCSC_Pool_cscl5_y->Fill((cscrhLP.x()-extSegm.y())/sqrt(cscrhLEP.yy()));
                  break;
              case 6:
                  GEMCSC_Residuals_cscl6_x->Fill(cscrhLP.x()-extSegm.x());
                  GEMCSC_Residuals_cscl6_y->Fill(cscrhLP.y()-extSegm.y());
                  GEMCSC_Pool_cscl6_x->Fill((cscrhLP.x()-extSegm.x())/sqrt(cscrhLEP.xx()));
                  GEMCSC_Pool_cscl6_y->Fill((cscrhLP.x()-extSegm.y())/sqrt(cscrhLEP.yy()));
                  break;
              default:
                  std::cout <<" Unphysical GEMCSC layer "<<cscrhId<<std::endl;
          }
      }
        //////////////////
      
        //////// GEM recHits ////////////////////
        auto gemrhs = gemcscs->gemRecHits();
        GEMCSC_NumGEMRH->Fill(gemrhs.size());
        for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){
            GEMDetId id((*rh).geographicalId());
            int chamber = id.chamber();
            
            //GEMDetId gemid(rh->geographicalId());
            auto gemrhId = rh->gemId();
            const GEMEtaPartition* gemrhRef  = gemGeom->etaPartition(gemrhId);
            auto gemrhLP = rh->localPosition();
            auto gemrhLEP = rh->localPositionError();
            auto gemrhGP = gemrhRef->toGlobal(gemrhLP);
            auto gemrhLP_inSegmRef = cscChamber->toLocal(gemrhGP);
            float phi_rh = gemrhGP.phi();
            float theta_rh = gemrhGP.theta();
            float eta_rh = gemrhGP.eta();
            float xe  = gemcscsegLP.x()+gemcscsegLD.x()*gemrhLP_inSegmRef.z()/gemcscsegLD.z();
            float ye  = gemcscsegLP.y()+gemcscsegLD.y()*gemrhLP_inSegmRef.z()/gemcscsegLD.z();
            float ze = gemrhLP_inSegmRef.z();
            LocalPoint extrPoint(xe,ye,ze); // in segment rest frame
            float sigma_xe = sqrt (gemcscsegLEP.xx()+gemcscsegLED.xx()*gemrhLP_inSegmRef.z()*gemrhLP_inSegmRef.z());
            float sigma_ye = sqrt (gemcscsegLEP.yy()+gemcscsegLED.yy()*gemrhLP_inSegmRef.z()*gemrhLP_inSegmRef.z());
            auto extrPoinGP_fromSegmRef = cscChamber->toGlobal(extrPoint);
            float phi_ext = extrPoinGP_fromSegmRef.phi();
            float theta_ext = extrPoinGP_fromSegmRef.theta();
            
            auto extSegm = gemrhRef->toLocal(cscChamber->toGlobal(extrPoint)); // in layer restframe

            GEMCSC_Dphi_min_afterCut->Fill(phi_ext-phi_rh);
            GEMCSC_Dtheta_min_afterCut->Fill((theta_ext-theta_rh));
            GEMCSC_DR_min_afterCut->Fill((theta_ext-theta_rh)*(theta_ext-theta_rh)+(phi_ext-phi_rh)*(phi_ext-phi_rh));
            
            switch (gemrhId.layer()){
                case 1:
                    GEMCSC_SSegm_xe_l1->Fill(xe);
                    GEMCSC_SSegm_ye_l1->Fill(ye);
                    GEMCSC_SSegm_ze_l1->Fill(ze);
                    GEMCSC_SSegm_sigmaxe_l1->Fill(sigma_xe);
                    GEMCSC_SSegm_sigmaye_l1->Fill(sigma_ye);
                    GEMCSC_Dphi_min_afterCut_l1->Fill((phi_ext-phi_rh));
                    if(chamber%2!=0){GEMCSC_Dphi_min_afterCut_l1_odd->Fill((phi_ext-phi_rh));
                        GEMCSC_Dphi_min_afterCut_odd->Fill((phi_ext-phi_rh));
                        GEMCSC_SSegm_xe_l1_odd->Fill(xe);
                        GEMCSC_SSegm_ye_l1_odd->Fill(ye);
                        GEMCSC_SSegm_ze_l1_odd->Fill(ze);
                        GEMCSC_SSegm_sigmaxe_l1_odd->Fill(sigma_xe);
                        GEMCSC_SSegm_sigmaye_l1_odd->Fill(sigma_ye);
                    }
                    else{GEMCSC_Dphi_min_afterCut_l1_even->Fill((phi_ext-phi_rh));
                        GEMCSC_Dphi_min_afterCut_even->Fill((phi_ext-phi_rh));
                        GEMCSC_SSegm_xe_l1_even->Fill(xe);
                        GEMCSC_SSegm_ye_l1_even->Fill(ye);
                        GEMCSC_SSegm_ze_l1_even->Fill(ze);
                        GEMCSC_SSegm_sigmaxe_l1_even->Fill(sigma_xe);
                        GEMCSC_SSegm_sigmaye_l1_even->Fill(sigma_ye);
                    }
                    
                    break;
                 case 2:
                    GEMCSC_SSegm_xe_l2->Fill(xe);
                    GEMCSC_SSegm_ye_l2->Fill(ye);
                    GEMCSC_SSegm_ze_l2->Fill(ze);
                    GEMCSC_SSegm_sigmaxe_l2->Fill(sigma_xe);
                    GEMCSC_SSegm_sigmaye_l2->Fill(sigma_ye);
                    GEMCSC_Dphi_min_afterCut_l2->Fill((phi_ext-phi_rh));
                    if(chamber%2!=0){GEMCSC_Dphi_min_afterCut_l2_odd->Fill((phi_ext-phi_rh));
                        GEMCSC_Dphi_min_afterCut_odd->Fill((phi_ext-phi_rh));
                        GEMCSC_SSegm_xe_l2_odd->Fill(xe);
                        GEMCSC_SSegm_ye_l2_odd->Fill(ye);
                        GEMCSC_SSegm_ze_l2_odd->Fill(ze);
                        GEMCSC_SSegm_sigmaxe_l2_odd->Fill(sigma_xe);
                        GEMCSC_SSegm_sigmaye_l2_odd->Fill(sigma_ye);
                    }
                    else{GEMCSC_Dphi_min_afterCut_l2_even->Fill((phi_ext-phi_rh));
                        GEMCSC_Dphi_min_afterCut_even->Fill((phi_ext-phi_rh));
                        GEMCSC_SSegm_xe_l2_even->Fill(xe);
                        GEMCSC_SSegm_ye_l2_even->Fill(ye);
                        GEMCSC_SSegm_ze_l2_even->Fill(ze);
                        GEMCSC_SSegm_sigmaxe_l2_even->Fill(sigma_xe);
                        GEMCSC_SSegm_sigmaye_l2_even->Fill(sigma_ye);
                    }
                    
                    break;
                default:
                    std::cout <<" Unphysical GEMCSC layer "<<gemrhId<<std::endl;
            }

            GEMCSC_Residuals_x->Fill(gemrhLP.x()-extSegm.x());
            GEMCSC_Residuals_y->Fill(gemrhLP.y()-extSegm.y());
            GEMCSC_Pool_x->Fill((gemrhLP.x()-extSegm.x())/sqrt(gemrhLEP.xx()));
            GEMCSC_Pool_y->Fill((gemrhLP.y()-extSegm.y())/sqrt(gemrhLEP.yy()));
            GEMCSC_Residuals_gem_x->Fill(gemrhLP.x()-extSegm.x());
            GEMCSC_Residuals_gem_y->Fill(gemrhLP.y()-extSegm.y());
            GEMCSC_Pool_gem_x->Fill((gemrhLP.x()-extSegm.x())/sqrt(gemrhLEP.xx()));
            GEMCSC_Pool_gem_y->Fill((gemrhLP.y()-extSegm.y())/sqrt(gemrhLEP.yy()));
            GEMCSC_Pool_gem_x_newE->Fill((gemrhLP.x()-extSegm.x())/sqrt(gemrhLEP.xx()+ sigma_xe*sigma_xe ));
            GEMCSC_Pool_gem_y_newE->Fill((gemrhLP.y()-extSegm.y())/sqrt(gemrhLEP.yy()+ sigma_ye*sigma_ye));
            
            if (eta_rh>0){
                GEMCSC_Pool_gem_x_newE_mp->Fill((gemrhLP.x()-extSegm.x())/sqrt(gemrhLEP.xx()+ sigma_xe*sigma_xe ));
                GEMCSC_Pool_gem_y_newE_mp->Fill((gemrhLP.y()-extSegm.y())/sqrt(gemrhLEP.yy()+ sigma_ye*sigma_ye));
            }
            if (eta_rh<0){
                GEMCSC_Pool_gem_x_newE_mm->Fill((gemrhLP.x()-extSegm.x())/sqrt(gemrhLEP.xx()+ sigma_xe*sigma_xe ));
                GEMCSC_Pool_gem_y_newE_mm->Fill((gemrhLP.y()-extSegm.y())/sqrt(gemrhLEP.yy()+ sigma_ye*sigma_ye));
            }

 
            
            if(chamber%2!=0){
              GEMCSC_Residuals_gem_odd_x->Fill(gemrhLP.x()-extSegm.x());
              GEMCSC_Residuals_gem_odd_y->Fill(gemrhLP.y()-extSegm.y());
              GEMCSC_Pool_gem_odd_x->Fill((gemrhLP.x()-extSegm.x())/sqrt(gemrhLEP.xx()));
              GEMCSC_Pool_gem_odd_y->Fill((gemrhLP.y()-extSegm.y())/sqrt(gemrhLEP.yy()));
                GEMCSC_Pool_gem_odd_x_newE->Fill((gemrhLP.x()-extSegm.x())/sqrt(gemrhLEP.xx()+ sigma_xe*sigma_xe ));
                GEMCSC_Pool_gem_odd_y_newE->Fill((gemrhLP.y()-extSegm.y())/sqrt(gemrhLEP.yy()+ sigma_ye*sigma_ye));
                if (eta_rh>0){
                    GEMCSC_Pool_gem_odd_x_newE_mp->Fill((gemrhLP.x()-extSegm.x())/sqrt(gemrhLEP.xx()+ sigma_xe*sigma_xe ));
                    GEMCSC_Pool_gem_odd_y_newE_mp->Fill((gemrhLP.y()-extSegm.y())/sqrt(gemrhLEP.yy()+ sigma_ye*sigma_ye));
                }
                if (eta_rh<0){
                    GEMCSC_Pool_gem_odd_x_newE_mm->Fill((gemrhLP.x()-extSegm.x())/sqrt(gemrhLEP.xx()+ sigma_xe*sigma_xe ));
                    GEMCSC_Pool_gem_odd_y_newE_mm->Fill((gemrhLP.y()-extSegm.y())/sqrt(gemrhLEP.yy()+ sigma_ye*sigma_ye));
                }
            }
            else {
              GEMCSC_Residuals_gem_even_x->Fill(gemrhLP.x()-extSegm.x());
              GEMCSC_Residuals_gem_even_y->Fill(gemrhLP.y()-extSegm.y());
              GEMCSC_Pool_gem_even_x->Fill((gemrhLP.x()-extSegm.x())/sqrt(gemrhLEP.xx()));
              GEMCSC_Pool_gem_even_y->Fill((gemrhLP.y()-extSegm.y())/sqrt(gemrhLEP.yy()));
              
            }

            switch (gemrhId.layer()){
              case 1:
                  GEMCSC_Residuals_geml1_x->Fill(gemrhLP.x()-extSegm.x());
                  GEMCSC_Residuals_geml1_y->Fill(gemrhLP.y()-extSegm.y());
                  GEMCSC_Pool_geml1_x->Fill((gemrhLP.x()-extSegm.x())/sqrt(gemrhLEP.xx()));
                  GEMCSC_Pool_geml1_y->Fill((gemrhLP.x()-extSegm.y())/sqrt(gemrhLEP.yy()));
                  break;
              case 2:
                  GEMCSC_Residuals_geml2_x->Fill(gemrhLP.x()-extSegm.x());
                  GEMCSC_Residuals_geml2_y->Fill(gemrhLP.y()-extSegm.y());
                  GEMCSC_Pool_geml2_x->Fill((gemrhLP.x()-extSegm.x())/sqrt(gemrhLEP.xx()));
                  GEMCSC_Pool_geml2_y->Fill((gemrhLP.x()-extSegm.y())/sqrt(gemrhLEP.yy()));
                  break;
              default:
                  std::cout <<" Unphysical GEMCSC layer "<<gemrhId<<std::endl;
                }
            
                    float xe_csc  = cscsegLP.x()+cscsegLD.x()*gemrhLP_inSegmRef.z()/cscsegLD.z();
                    float ye_csc  = cscsegLP.y()+cscsegLD.y()*gemrhLP_inSegmRef.z()/cscsegLD.z();
                    float ze_csc = gemrhLP_inSegmRef.z();
                    LocalPoint extrPoint_csc(xe_csc,ye_csc,ze_csc); // in segment rest frame
                    float sigma_xe_csc = sqrt(cscsegLEP.xx()+cscsegLED.xx()*gemrhLP_inSegmRef.z()*gemrhLP_inSegmRef.z());
                    float sigma_ye_csc = sqrt(cscsegLEP.yy()+cscsegLED.yy()*gemrhLP_inSegmRef.z()*gemrhLP_inSegmRef.z());
                    auto extrPoinGP_csc_fromSegmRef = cscChamber->toGlobal(extrPoint_csc);
                    float phi_ext_csc = extrPoinGP_csc_fromSegmRef.phi();
                    float theta_ext_csc = extrPoinGP_csc_fromSegmRef.theta();
            
                    //auto extSegm_csc = gemrhRef->toLocal(cscChamber->toGlobal(extrPoint_csc)); // in layer restframe
                    
                    GEMCSC_Dphi_cscS_min_afterCut->Fill((phi_ext_csc-phi_rh));
                    GEMCSC_Dtheta_cscS_min_afterCut->Fill((theta_ext_csc-theta_rh));
                    GEMCSC_DR_cscS_min_afterCut->Fill((theta_ext_csc-theta_rh)*(theta_ext_csc-theta_rh)+(phi_ext_csc-phi_rh)*(phi_ext_csc-phi_rh));
            if(chamber%2!=0){GEMCSC_Dphi_cscS_min_afterCut_odd->Fill((phi_ext_csc-phi_rh));
                GEMCSC_Dtheta_cscS_min_afterCut_odd->Fill((theta_ext_csc-theta_rh));
                switch (gemrhId.roll()){
                    case 1: GEMCSC_Dphi_cscS_min_afterCut_odd_r1->Fill((phi_ext_csc-phi_rh)); break;
                    case 2: GEMCSC_Dphi_cscS_min_afterCut_odd_r2->Fill((phi_ext_csc-phi_rh)); break;
                    case 3: GEMCSC_Dphi_cscS_min_afterCut_odd_r3->Fill((phi_ext_csc-phi_rh)); break;
                    case 4: GEMCSC_Dphi_cscS_min_afterCut_odd_r4->Fill((phi_ext_csc-phi_rh)); break;
                    case 5: GEMCSC_Dphi_cscS_min_afterCut_odd_r5->Fill((phi_ext_csc-phi_rh)); break;
                    case 6: GEMCSC_Dphi_cscS_min_afterCut_odd_r6->Fill((phi_ext_csc-phi_rh)); break;
                    case 7: GEMCSC_Dphi_cscS_min_afterCut_odd_r7->Fill((phi_ext_csc-phi_rh)); break;
                    case 8: GEMCSC_Dphi_cscS_min_afterCut_odd_r8->Fill((phi_ext_csc-phi_rh)); break;
                    default:
                    std::cout <<" Unphysical GEM roll "<<gemrhId<<std::endl;
                }
            }
            else {
                GEMCSC_Dtheta_cscS_min_afterCut_even->Fill((theta_ext_csc-theta_rh));
                GEMCSC_Dphi_cscS_min_afterCut_even->Fill((phi_ext_csc-phi_rh));}
            
                    switch (gemrhId.layer()){
                        case 1:
                            GEMCSC_CSCSegm_xe_l1->Fill(xe_csc);
                            GEMCSC_CSCSegm_ye_l1->Fill(ye_csc);
                            GEMCSC_CSCSegm_ze_l1->Fill(ze_csc);
                            GEMCSC_CSCSegm_sigmaxe_l1->Fill(sigma_xe_csc);
                            GEMCSC_CSCSegm_sigmaye_l1->Fill(sigma_ye_csc);
                            GEMCSC_Dphi_cscS_min_afterCut_l1->Fill((phi_ext_csc-phi_rh));
                            if(chamber%2!=0){GEMCSC_Dphi_cscS_min_afterCut_l1_odd->Fill((phi_ext_csc-phi_rh));
                                GEMCSC_CSCSegm_xe_l1_odd->Fill(xe_csc);
                                GEMCSC_CSCSegm_ye_l1_odd->Fill(ye_csc);
                                GEMCSC_CSCSegm_ze_l1_odd->Fill(ze_csc);
                                GEMCSC_CSCSegm_sigmaxe_l1_odd->Fill(sigma_xe_csc);
                                GEMCSC_CSCSegm_sigmaye_l1_odd->Fill(sigma_ye_csc);
                            }
                            else{GEMCSC_Dphi_cscS_min_afterCut_l1_even->Fill((phi_ext_csc-phi_rh));
                                GEMCSC_CSCSegm_xe_l1_even->Fill(xe_csc);
                                GEMCSC_CSCSegm_ye_l1_even->Fill(ye_csc);
                                GEMCSC_CSCSegm_ze_l1_even->Fill(ze_csc);
                                GEMCSC_CSCSegm_sigmaxe_l1_even->Fill(sigma_xe_csc);
                                GEMCSC_CSCSegm_sigmaye_l1_even->Fill(sigma_ye_csc);
                            }
                            
                            break;
                        case 2:
                            GEMCSC_CSCSegm_xe_l2->Fill(xe_csc);
                            GEMCSC_CSCSegm_ye_l2->Fill(ye_csc);
                            GEMCSC_CSCSegm_ze_l2->Fill(ze_csc);
                            GEMCSC_CSCSegm_sigmaxe_l2->Fill(sigma_xe_csc);
                            GEMCSC_CSCSegm_sigmaye_l2->Fill(sigma_ye_csc);

                            GEMCSC_Dphi_cscS_min_afterCut_l2->Fill((phi_ext_csc-phi_rh));
                            if(chamber%2!=0){GEMCSC_Dphi_cscS_min_afterCut_l2_odd->Fill((phi_ext_csc-phi_rh));
                                GEMCSC_CSCSegm_xe_l2_odd->Fill(xe_csc);
                                GEMCSC_CSCSegm_ye_l2_odd->Fill(ye_csc);
                                GEMCSC_CSCSegm_ze_l2_odd->Fill(ze_csc);
                                GEMCSC_CSCSegm_sigmaxe_l2_odd->Fill(sigma_xe_csc);
                                GEMCSC_CSCSegm_sigmaye_l2_odd->Fill(sigma_ye_csc);
                            }
                            else{GEMCSC_Dphi_cscS_min_afterCut_l2_even->Fill((phi_ext_csc-phi_rh));
                                GEMCSC_CSCSegm_xe_l2_even->Fill(xe_csc);
                                GEMCSC_CSCSegm_ye_l2_even->Fill(ye_csc);
                                GEMCSC_CSCSegm_ze_l2_even->Fill(ze_csc);
                                GEMCSC_CSCSegm_sigmaxe_l2_even->Fill(sigma_xe_csc);
                                GEMCSC_CSCSegm_sigmaye_l2_even->Fill(sigma_ye_csc);
                            }

                            break;
                        default:
                            std::cout <<" Unphysical GEMCSC layer "<<gemrhId<<std::endl;
                    }

            int Nsimhit=0;
            edm::PSimHitContainer selGEMSimHit = SimHitMatched(rh, gemGeom, iEvent);
            
            for(edm::PSimHitContainer::const_iterator itHit = selGEMSimHit.begin(); itHit != selGEMSimHit.end(); ++itHit){
                    Nsimhit++;
                    if(Nsimhit>1)continue;
                
                
                
                //const GEMEtaPartition* gemshRef  = gemGeom->etaPartition(gemshId);
                LocalPoint gemshLP = itHit->localPosition();
                GlobalPoint gemshGP(gemGeom->idToDet(itHit->detUnitId())->surface().toGlobal(gemshLP));
                //auto gemrhGP = gemrhRef->toGlobal(gemrhLP);
                auto gemshLP_inSegmRef = cscChamber->toLocal(gemshGP);
                //auto gemshLEP = itHit->localPositionError();
                //LocalPoint lp = itHit->entryPoint();
                //std::cout <<" entry Position x = "<<lp.x()<<" y= "<<lp.y()<<" z= "<<lp.z()<<std::endl;
                //auto gemshLP_inSegmRef = cscChamber->toLocal(gemshGP);
                //GlobalPoint hitGP_sim(gemGeom->idToDet(itHit->detUnitId())->surface().toGlobal(lp));
                float phi_sh = gemshGP.phi();
                float theta_sh = gemshGP.theta();
                //float eta_sh = gemshGP.eta();
                
                float xe_sh  = gemcscsegLP.x()+gemcscsegLD.x()*gemshLP_inSegmRef.z()/gemcscsegLD.z();
                float ye_sh  = gemcscsegLP.y()+gemcscsegLD.y()*gemshLP_inSegmRef.z()/gemcscsegLD.z();
                float ze_sh = gemshLP_inSegmRef.z();
                LocalPoint extrPoint_sh(xe_sh,ye_sh,ze_sh); // in segment rest frame
                //float sigma_xe_sh = sqrt (gemcscsegLEP.xx());
                //float sigma_ye_sh = sqrt (gemcscsegLEP.yy());
                float sigma_xe_sh = sqrt (gemcscsegLEP.xx()+gemcscsegLED.xx()*gemshLP_inSegmRef.z()*gemshLP_inSegmRef.z());
                float sigma_ye_sh = sqrt (gemcscsegLEP.yy()+gemcscsegLED.yy()*gemshLP_inSegmRef.z()*gemshLP_inSegmRef.z());
                auto extSegm_sh = gemrhRef->toLocal(cscChamber->toGlobal(extrPoint_sh));
                auto extrPoinGP_fromSegmRef_sh = cscChamber->toGlobal(extrPoint_sh);
                float phi_ext_sh = extrPoinGP_fromSegmRef_sh.phi();
                float theta_ext_sh = extrPoinGP_fromSegmRef_sh.theta();
                
                float dxdz=(gemshLP_inSegmRef.x()-cscsegLP.x())/gemshLP_inSegmRef.z();
                float dydz=(gemshLP_inSegmRef.y()-cscsegLP.y())/gemshLP_inSegmRef.z();
                float sigma_dxdz=cscsegLEP.xx()/(gemshLP_inSegmRef.z()*gemshLP_inSegmRef.z());
                float sigma_dydz=cscsegLEP.yy()/(gemshLP_inSegmRef.z()*gemshLP_inSegmRef.z());
                SIMGEMCSC_SSegm_LDx->Fill(dxdz);
                SIMGEMCSC_SSegm_LDy->Fill(dydz);
                SIMGEMCSC_SSegm_LDEx->Fill(sigma_dxdz);
                SIMGEMCSC_SSegm_LDEy->Fill(sigma_dydz);
                
                float xe_csc_sh  = cscsegLP.x()+cscsegLD.x()*gemshLP_inSegmRef.z()/cscsegLD.z();
                float ye_csc_sh  = cscsegLP.y()+cscsegLD.y()*gemshLP_inSegmRef.z()/cscsegLD.z();
                float ze_csc_sh = gemshLP_inSegmRef.z();
                LocalPoint extrPoint_csc_sh(xe_csc_sh,ye_csc_sh,ze_csc_sh); // in segment rest frame
                //float sigma_xe_csc_sh = sqrt(cscsegLEP.xx()+cscsegLED.xx()*gemshLP_inSegmRef.z()*gemshLP_inSegmRef.z());
                //float sigma_ye_csc_sh = sqrt(cscsegLEP.yy()+cscsegLED.yy()*gemshLP_inSegmRef.z()*gemshLP_inSegmRef.z());
                auto extrPoinGP_csc_fromSegmRef_sh = cscChamber->toGlobal(extrPoint_csc_sh);
                float phi_ext_csc_sh = extrPoinGP_csc_fromSegmRef_sh.phi();
                float theta_ext_csc_sh = extrPoinGP_csc_fromSegmRef_sh.theta();

                SIMGEMCSC_Dphi_cscS_min_afterCut->Fill((phi_ext_csc_sh-phi_sh));
                SIMGEMCSC_Dtheta_cscS_min_afterCut->Fill((theta_ext_csc_sh-theta_sh));
                SIMGEMCSC_Dphi_SS_min_afterCut->Fill((phi_ext_sh-phi_sh));
                SIMGEMCSC_Dtheta_SS_min_afterCut->Fill((theta_ext_sh-theta_sh));
                
                SIMGEMCSC_Residuals_gem_x->Fill(gemshLP.x()-extSegm_sh.x());
                SIMGEMCSC_Residuals_gem_y->Fill(gemshLP.y()-extSegm_sh.y());
                SIMGEMCSC_Pool_gem_x_newE->Fill((gemshLP.x()-extSegm_sh.x())/sqrt(sigma_xe_sh*sigma_xe_sh ));
                SIMGEMCSC_Pool_gem_y_newE->Fill((gemshLP.y()-extSegm_sh.y())/sqrt(sigma_ye_sh*sigma_ye_sh));
                
                SIMGEMCSC_Residuals_gem_rhsh_x->Fill(gemshLP.x()-gemrhLP.x());
                SIMGEMCSC_Residuals_gem_rhsh_y->Fill(gemshLP.y()-gemrhLP.y());
                SIMGEMCSC_Pool_gem_rhsh_x_newE->Fill((gemshLP.x()-gemrhLP.x())/sqrt(gemrhLEP.xx()));
                SIMGEMCSC_Pool_gem_rhsh_y_newE->Fill((gemshLP.y()-gemrhLP.y())/sqrt(gemrhLEP.yy()));
                
                if(chamber%2!=0){
                    SIMGEMCSC_Residuals_gem_odd_x->Fill(gemshLP.x()-extSegm_sh.x());
                    SIMGEMCSC_Residuals_gem_odd_y->Fill(gemshLP.y()-extSegm_sh.y());
                    SIMGEMCSC_Pool_gem_x_newE_odd->Fill((gemshLP.x()-extSegm_sh.x())/sqrt(sigma_xe_sh*sigma_xe_sh ));
                    SIMGEMCSC_Pool_gem_y_newE_odd->Fill((gemshLP.y()-extSegm_sh.y())/sqrt(sigma_ye_sh*sigma_ye_sh));
                    SIMGEMCSC_Dphi_cscS_min_afterCut_odd->Fill((phi_ext_csc_sh-phi_sh));
                    SIMGEMCSC_Dtheta_cscS_min_afterCut_odd->Fill((theta_ext_csc_sh-theta_sh));
                    SIMGEMCSC_Dphi_SS_min_afterCut_odd->Fill((phi_ext_sh-phi_sh));
                    SIMGEMCSC_Dtheta_SS_min_afterCut_odd->Fill((theta_ext_sh-theta_sh));
                    SIMGEMCSC_theta_cscSsh_vs_ndof_odd->Fill((theta_ext_csc_sh-theta_sh),cscSeg.degreesOfFreedom());
                    
                }
                else{
                    SIMGEMCSC_Residuals_gem_even_x->Fill(gemshLP.x()-extSegm_sh.x());
                    SIMGEMCSC_Residuals_gem_even_y->Fill(gemshLP.y()-extSegm_sh.y());
                    SIMGEMCSC_Pool_gem_x_newE_even->Fill((gemshLP.x()-extSegm_sh.x())/sqrt(sigma_xe_sh*sigma_xe_sh ));
                    SIMGEMCSC_Pool_gem_y_newE_even->Fill((gemshLP.y()-extSegm_sh.y())/sqrt(sigma_ye_sh*sigma_ye_sh));
                    SIMGEMCSC_Dphi_cscS_min_afterCut_even->Fill((phi_ext_csc_sh-phi_sh));
                    SIMGEMCSC_Dtheta_cscS_min_afterCut_even->Fill((theta_ext_csc_sh-theta_sh));
                    SIMGEMCSC_Dphi_SS_min_afterCut_even->Fill((phi_ext_sh-phi_sh));
                    SIMGEMCSC_Dtheta_SS_min_afterCut_even->Fill((theta_ext_sh-theta_sh));
                    SIMGEMCSC_theta_cscSsh_vs_ndof_even->Fill((theta_ext_csc_sh-theta_sh),cscSeg.degreesOfFreedom());
                }
                
            }// fine loop sim hit compatibili

        }// fine loop gem rh
        
    
        
        
        
        
        
        
        
        
        /////////////////////////////////////////////
        auto gemcscrhs = gemcscs->recHits();
        for (auto rh = gemcscrhs.begin(); rh!= gemcscrhs.end(); rh++){
	  
	  // if (rh->geographicalId().subdetId() == MuonSubdetId::CSC){
	  DetId d = DetId((*rh)->rawId());
	  if (d.subdetId() == MuonSubdetId::CSC) {
	    
	    std::cout<<"CSC found"<<std::endl;
	    
	    // CSCDetId id(rh->geographicalId());
	    CSCDetId id(d);
	    
	    //int layer = id.layer();
	    int station = id.station();
	    int ring = id.ring();
	    //int chamber = id.chamber();
	    //int roll = id.roll();
	    
	    std::cout<<"CSC Region"<<" Station "<<station<<" ring "<<ring<<std::endl;
	    
	  }
	  // else if (rh->geographicalId().subdetId() == MuonSubdetId::GEM){
	  else if (d.subdetId() == MuonSubdetId::GEM) {
	    // std::cout<<"GEM found"<<std::endl;
	    
	    // GEMDetId id(rh->geographicalId());
	    GEMDetId id(d);
	    int region = id.region();
	    int layer = id.layer();
	    int station = id.station();
	    int ring = id.ring();
	    int chamber = id.chamber();
	    int roll = id.roll();
	    
	    std::cout<<"GEM Region"<<region<<" Station "<<station<<" ring "<<ring<<" layer "<<layer<<" chamber "<<chamber<<" roll "<<roll<<std::endl;
	  }
	  
	  
        }

      
        auto gemcscsegGD = cscChamber->toGlobal(gemcscs->localPosition());
        for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack){
            double simEta = (*simTrack).momentum().eta();
            double simPhi = (*simTrack).momentum().phi();
            double dR = sqrt(pow((simEta-gemcscsegGD.eta()),2) + pow((simPhi-gemcscsegGD.phi()),2));
            if(dR > 0.1) continue;
  
        
        }
  
  
  }// loop gemcsc segments

  std::cout <<"------------------------------------------------------------------------------"<<std::endl;
  std::cout <<"------------------------------------------------------------------------------"<<std::endl;
       
}

//
// member functions
//


edm::PSimHitContainer TestGEMCSCSegmentAnalyzer::SimHitMatched(std::vector<GEMRecHit>::const_iterator recHit, edm::ESHandle<GEMGeometry> gemGeom, const Event & iEvent)
{
    

    edm::PSimHitContainer selectedGEMHits;

    GEMDetId id((*recHit).geographicalId());

    
    int region = id.region();
    int layer = id.layer();
    int station = id.station();
    int chamber = id.chamber();
    int roll = id.roll();

    int cls = recHit->clusterSize();
    int firstStrip = recHit->firstClusterStrip();

    edm::Handle<edm::PSimHitContainer> GEMHits;
    iEvent.getByToken(GEMSimHit_Token, GEMHits);
    // edm::Handle<edm::PSimHitContainer> GEMHits;
    // iEvent.getByLabel(edm::InputTag("g4SimHits","MuonGEMHits"), GEMHits);

    for(edm::PSimHitContainer::const_iterator itHit = GEMHits->begin(); itHit != GEMHits->end(); ++itHit){
        if (!(abs(itHit->particleType()) == 13)) continue;
      	GEMDetId idGem = GEMDetId(itHit->detUnitId());
      	int region_sim = idGem.region();
      	int layer_sim = idGem.layer();
      	int station_sim = idGem.station();
      	int chamber_sim = idGem.chamber();
      	int roll_sim = idGem.roll();
        
      	LocalPoint lp = itHit->entryPoint();
        // GlobalPoint hitGP_sim(gemGeom->idToDet(itHit->detUnitId())->surface().toGlobal(lp));
      	float strip_sim = gemGeom->etaPartition(idGem)->strip(lp);
      	if(region != region_sim) continue;
      	if(layer != layer_sim) continue;
      	if(station != station_sim) continue;
      	if(chamber != chamber_sim) continue;
      	if(roll != roll_sim) continue;
        for(int i = firstStrip; i < (firstStrip + cls); i++ ){
            if(abs(strip_sim-i)<1) {selectedGEMHits.push_back(*itHit);}
        }

        
    }
 

    return selectedGEMHits;
    
}


// ------------ method called once each job just before starting event loop  ------------
void 
TestGEMCSCSegmentAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestGEMCSCSegmentAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
TestGEMCSCSegmentAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  //iSetup.get<MuonGeometryRecord>().get(gemGeom);
  //iSetup.get<MuonGeometryRecord>().get(cscGeom);



}

//define this as a plug-in
DEFINE_FWK_MODULE(TestGEMCSCSegmentAnalyzer);
