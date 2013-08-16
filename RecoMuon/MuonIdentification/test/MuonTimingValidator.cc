// -*- C++ -*-
//
// Package:    MuonTimingValidator
// Class:      MuonTimingValidator
// 
/**\class MuonTimingValidator MuonTimingValidator.cc 

 Description: An example analyzer that fills muon timing information histograms 

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Piotr Traczyk
//         Created:  Wed Sep 27 14:54:28 EDT 2006
//
//

#include "RecoMuon/MuonIdentification/test/MuonTimingValidator.h"
#include "FWCore/Utilities/interface/isFinite.h"

// system include files
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <iostream>
#include <iomanip>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCleaner.h"
#include "RecoLocalMuon/DTSegment/src/DTHitPairForFit.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"

#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCRangeMapAccessor.h>

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/MuonReco/interface/MuonCocktails.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TFrame.h>

//
// constructors and destructor
//
MuonTimingValidator::MuonTimingValidator(const edm::ParameterSet& iConfig) 
  :
  TKtrackTags_(iConfig.getUntrackedParameter<edm::InputTag>("TKtracks")),
  MuonTags_(iConfig.getUntrackedParameter<edm::InputTag>("Muons")),
  TimeTags_(iConfig.getUntrackedParameter<edm::InputTag>("Timing")),
  out(iConfig.getParameter<std::string>("out")),
  open(iConfig.getParameter<std::string>("open")),
  theMinEta(iConfig.getParameter<double>("etaMin")),
  theMaxEta(iConfig.getParameter<double>("etaMax")),
  theMinPt(iConfig.getParameter<double>("simPtMin")),
  thePtCut(iConfig.getParameter<double>("PtCut")),
  theMinPtres(iConfig.getParameter<double>("PtresMin")),
  theMaxPtres(iConfig.getParameter<double>("PtresMax")),
  theScale(iConfig.getParameter<double>("PlotScale")),
  theDtCut(iConfig.getParameter<int>("DTcut")),
  theCscCut(iConfig.getParameter<int>("CSCcut")),
  theNBins(iConfig.getParameter<int>("nbins"))
{
  //now do what ever initialization is needed
}


MuonTimingValidator::~MuonTimingValidator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  if (hFile!=0) {
    hFile->Close();
    delete hFile;
  }
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
MuonTimingValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
//  std::cout << "*** Begin Muon Timing Validatior " << std::endl;
//  std::cout << " Event: " << iEvent.id() << "  Orbit: " << iEvent.orbitNumber() << "  BX: " << iEvent.bunchCrossing() << std::endl;

  iEvent.getByLabel( TKtrackTags_, TKTrackCollection);
  reco::TrackCollection tkTC;
  const reco::TrackCollection tkTC1 = *(TKTrackCollection.product());

  iEvent.getByLabel(MuonTags_,MuCollection);
  const reco::MuonCollection muonC = *(MuCollection.product());
  if (!muonC.size()) return;

  iEvent.getByLabel(TimeTags_.label(),"combined",timeMap1);
  const reco::MuonTimeExtraMap & timeMapCmb = *timeMap1;
  iEvent.getByLabel(TimeTags_.label(),"dt",timeMap2);
  const reco::MuonTimeExtraMap & timeMapDT = *timeMap2;
  iEvent.getByLabel(TimeTags_.label(),"csc",timeMap3);
  const reco::MuonTimeExtraMap & timeMapCSC = *timeMap3;

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  reco::MuonCollection::const_iterator imuon;
  
  bool debug=false;

  int imucount=0;
  for(imuon = muonC.begin(); imuon != muonC.end(); ++imuon){
    
    debug=false;
    // if (imuon->pt()>100 && imuon->isGlobalMuon()) debug=true;
    
    if (debug)
      std::cout << " Event: " << iEvent.id() << " Found muon. Pt: " << imuon->p() << std::endl;
    
    if ((fabs(imuon->eta())<theMinEta) || (fabs(imuon->eta())>theMaxEta)) continue;
    
    reco::TrackRef trkTrack = imuon->track();
    if (trkTrack.isNonnull()) { 
      hi_tk_pt->Fill(((*trkTrack).pt()));
      hi_tk_phi->Fill(((*trkTrack).phi()));
      hi_tk_eta->Fill(((*trkTrack).eta()));
    }  

    reco::TrackRef staTrack = imuon->standAloneMuon();
    if (staTrack.isNonnull()) {
      hi_sta_pt->Fill((*staTrack).pt());
      hi_sta_phi->Fill((*staTrack).phi());
      hi_sta_eta->Fill(((*staTrack).eta()));
    }

    reco::TrackRef glbTrack = imuon->combinedMuon();
    if (glbTrack.isNonnull()) {
      hi_glb_pt->Fill((*glbTrack).pt());
      hi_glb_phi->Fill((*glbTrack).phi());
      hi_glb_eta->Fill((*glbTrack).eta()); 
      
      if (debug)
        std::cout << " Global Pt: " << (*glbTrack).pt() << std::endl;
    }

    // Analyze the short info stored directly in reco::Muon
    
    reco::MuonTime muonTime;
    if (imuon->isTimeValid()) { 
      muonTime = imuon->time();
//      std::cout << "    Time points: " << muonTime.nDof << "  time: " << muonTime.timeAtIpInOut << std::endl;
      if (muonTime.nDof) {
        hi_mutime_vtx->Fill(muonTime.timeAtIpInOut);
        hi_mutime_vtx_err->Fill(muonTime.timeAtIpInOutErr);
      }
    }
    
    reco::MuonEnergy muonE;
//    if (imuon->isEnergyValid() && fabs(imuon->eta())<1.5) { 
    if (imuon->isEnergyValid()) { 
      muonE = imuon->calEnergy();
      if (muonE.emMax>0.25) {
        hi_ecal_time->Fill(muonE.ecal_time);
        if (muonE.emMax>1.) hi_ecal_time_ecut->Fill(muonE.ecal_time);
        hi_ecal_energy->Fill(muonE.emMax);
        if (muonE.hadMax>0.25) hi_hcalecal_vtx->Fill(muonE.hcal_time-muonE.ecal_time);
        
        double emErr = 1.5/muonE.emMax;
        
        if (emErr>0.) {
          hi_ecal_time_err->Fill(emErr);
          hi_ecal_time_pull->Fill((muonE.ecal_time-1.)/emErr);
          if (debug)
            std::cout << "     ECAL time: " << muonE.ecal_time << " +/- " << emErr << std::endl;
        }
      }

      if (muonE.hadMax>0.25) {
        hi_hcal_time->Fill(muonE.hcal_time);
        if (muonE.hadMax>1.) hi_hcal_time_ecut->Fill(muonE.hcal_time);
        hi_hcal_energy->Fill(muonE.hadMax);
        
        double hadErr = 1.; // DUMMY!!!
        
        hi_hcal_time_err->Fill(hadErr);
        hi_hcal_time_pull->Fill((muonE.hcal_time-1.)/hadErr);
        if (debug)
          std::cout << "     HCAL time: " << muonE.hcal_time << " +/- " << hadErr << std::endl;
      }
    }
    
    // Analyze the MuonTimeExtra information
    reco::MuonRef muonR(MuCollection,imucount);
    reco::MuonTimeExtra timec = timeMapCmb[muonR];
    reco::MuonTimeExtra timedt = timeMapDT[muonR];
    reco::MuonTimeExtra timecsc = timeMapCSC[muonR];

    hi_cmbtime_ndof->Fill(timec.nDof());
    hi_dttime_ndof->Fill(timedt.nDof());
    hi_csctime_ndof->Fill(timecsc.nDof());

    if (timedt.nDof()>theDtCut) {
      if (debug) {
        std::cout << "          DT nDof: " << timedt.nDof() << std::endl;
        std::cout << "          DT Time: " << timedt.timeAtIpInOut() << " +/- " << timedt.inverseBetaErr() << std::endl;
        std::cout << "         DT FreeB: " << timedt.freeInverseBeta() << " +/- " << timedt.freeInverseBetaErr() << std::endl;
      }
      hi_dttime_ibt->Fill(timedt.inverseBeta());
      hi_dttime_ibt_pt->Fill(imuon->pt(),timedt.inverseBeta());
      hi_dttime_ibt_err->Fill(timedt.inverseBetaErr());
      hi_dttime_fib->Fill(timedt.freeInverseBeta());
      hi_dttime_fib_err->Fill(timedt.freeInverseBetaErr());
      hi_dttime_vtx->Fill(timedt.timeAtIpInOut());
      hi_dttime_vtx_err->Fill(timedt.timeAtIpInOutErr());
      hi_dttime_vtxr->Fill(timedt.timeAtIpOutIn());
      hi_dttime_vtxr_err->Fill(timedt.timeAtIpOutInErr());
      hi_dttime_errdiff->Fill(timedt.timeAtIpInOutErr()-timedt.timeAtIpOutInErr());

      if (timedt.inverseBetaErr()>0.)
        hi_dttime_ibt_pull->Fill((timedt.inverseBeta()-1.)/timedt.inverseBetaErr());
      if (timedt.freeInverseBetaErr()>0.)    
        hi_dttime_fib_pull->Fill((timedt.freeInverseBeta()-1.)/timedt.freeInverseBetaErr());
      if (timedt.timeAtIpInOutErr()>0.)
        hi_dttime_vtx_pull->Fill(timedt.timeAtIpInOut()/timedt.timeAtIpInOutErr());
      if (timedt.timeAtIpOutInErr()>0.)
        hi_dttime_vtxr_pull->Fill(timedt.timeAtIpOutIn()/timedt.timeAtIpOutInErr());

      if (timecsc.nDof()>theCscCut)
        hi_dtcsc_vtx->Fill(timedt.timeAtIpInOut()-timecsc.timeAtIpInOut());
      if (imuon->isEnergyValid()) {
        if (muonE.emMax>0.25) hi_dtecal_vtx->Fill(timedt.timeAtIpInOut()-muonE.ecal_time);
        if (muonE.hadMax>0.25) hi_dthcal_vtx->Fill(timedt.timeAtIpInOut()-muonE.hcal_time);
      }    

    }

    if (timecsc.nDof()>theCscCut) {
      if (debug) {
        std::cout << "         CSC nDof: " << timecsc.nDof() << std::endl;
        std::cout << "         CSC Time: " << timecsc.timeAtIpInOut() << " +/- " << timecsc.inverseBetaErr() << std::endl;
        std::cout << "        CSC FreeB: " << timecsc.freeInverseBeta() << " +/- " << timecsc.freeInverseBetaErr() << std::endl;
      }
      hi_csctime_ibt->Fill(timecsc.inverseBeta());
      hi_csctime_ibt_pt->Fill(imuon->pt(),timecsc.inverseBeta());
      hi_csctime_ibt_err->Fill(timecsc.inverseBetaErr());
      hi_csctime_fib->Fill(timecsc.freeInverseBeta());
      hi_csctime_fib_err->Fill(timecsc.freeInverseBetaErr());
      hi_csctime_vtx->Fill(timecsc.timeAtIpInOut());
      hi_csctime_vtx_err->Fill(timecsc.timeAtIpInOutErr());
      hi_csctime_vtxr->Fill(timecsc.timeAtIpOutIn());
      hi_csctime_vtxr_err->Fill(timecsc.timeAtIpOutInErr());

      if (timec.inverseBetaErr()>0.)
        hi_csctime_ibt_pull->Fill((timecsc.inverseBeta()-1.)/timecsc.inverseBetaErr());
      if (timecsc.freeInverseBetaErr()>0.)    
        hi_csctime_fib_pull->Fill((timecsc.freeInverseBeta()-1.)/timecsc.freeInverseBetaErr());
      if (timecsc.timeAtIpInOutErr()>0.)
        hi_csctime_vtx_pull->Fill(timecsc.timeAtIpInOut()/timecsc.timeAtIpInOutErr());
      if (timecsc.timeAtIpOutInErr()>0.)
        hi_csctime_vtxr_pull->Fill(timecsc.timeAtIpOutIn()/timecsc.timeAtIpOutInErr());

      if (imuon->isEnergyValid()) {
        if (muonE.emMax>0.25) hi_ecalcsc_vtx->Fill(muonE.ecal_time-timecsc.timeAtIpInOut());
        if (muonE.hadMax>0.25) hi_hcalcsc_vtx->Fill(muonE.hcal_time-timecsc.timeAtIpInOut());
      }
    }
    
    if (timec.nDof()>0) {
      if (debug) {
        std::cout << "    Combined nDof: " << timec.nDof() << std::endl;
        std::cout << "    Combined Time: " << timec.timeAtIpInOut() << " +/- " << timec.inverseBetaErr() << std::endl;
        std::cout << "   Combined FreeB: " << timec.freeInverseBeta() << " +/- " << timec.freeInverseBetaErr() << std::endl;
      }
      hi_cmbtime_ibt->Fill(timec.inverseBeta());
      hi_cmbtime_ibt_pt->Fill(imuon->pt(),timec.inverseBeta());
      hi_cmbtime_ibt_err->Fill(timec.inverseBetaErr());
      hi_cmbtime_fib->Fill(timec.freeInverseBeta());
      hi_cmbtime_fib_err->Fill(timec.freeInverseBetaErr());
      hi_cmbtime_vtx->Fill(timec.timeAtIpInOut());
      hi_cmbtime_vtx_err->Fill(timec.timeAtIpInOutErr());
      hi_cmbtime_vtxr->Fill(timec.timeAtIpOutIn());
      hi_cmbtime_vtxr_err->Fill(timec.timeAtIpOutInErr());

      if (timec.inverseBetaErr()>0.)
        hi_cmbtime_ibt_pull->Fill((timec.inverseBeta()-1.)/timec.inverseBetaErr());
      if (timec.freeInverseBetaErr()>0.)    
        hi_cmbtime_fib_pull->Fill((timec.freeInverseBeta()-1.)/timec.freeInverseBetaErr());
      if (timec.timeAtIpInOutErr()>0.)
        hi_cmbtime_vtx_pull->Fill(timec.timeAtIpInOut()/timec.timeAtIpInOutErr());
      if (timec.timeAtIpOutInErr()>0.)
        hi_cmbtime_vtxr_pull->Fill(timec.timeAtIpOutIn()/timec.timeAtIpOutInErr());
    }

    imucount++;    
  }  
}


// ------------ method called once each job just before starting event loop  ------------
void 
MuonTimingValidator::beginJob()
{
   hFile = new TFile( out.c_str(), open.c_str() );
   hFile->cd();

   effStyle = new TStyle("effStyle","Efficiency Study Style");   
   effStyle->SetCanvasBorderMode(0);
   effStyle->SetPadBorderMode(1);
   effStyle->SetOptTitle(0);
   effStyle->SetStatFont(42);
   effStyle->SetTitleFont(22);
   effStyle->SetCanvasColor(10);
   effStyle->SetPadColor(0);
   effStyle->SetLabelFont(42,"x");
   effStyle->SetLabelFont(42,"y");
   effStyle->SetHistFillStyle(1001);
   effStyle->SetHistFillColor(0);
   effStyle->SetOptStat(0);
   effStyle->SetOptFit(0111);
   effStyle->SetStatH(0.05);

   hi_sta_pt   = new TH1F("hi_sta_pt","P_{T}^{STA}",theNBins,0.0,theMaxPtres);
   hi_tk_pt   = new TH1F("hi_tk_pt","P_{T}^{TK}",theNBins,0.0,theMaxPtres);
   hi_glb_pt   = new TH1F("hi_glb_pt","P_{T}^{GLB}",theNBins,0.0,theMaxPtres);

   hi_sta_phi = new TH1F("hi_sta_phi","#phi^{STA}",theNBins,-3.0,3.);
   hi_tk_phi  = new TH1F("hi_tk_phi","#phi^{TK}",theNBins,-3.0,3.);
   hi_glb_phi = new TH1F("hi_glb_phi","#phi^{GLB}",theNBins,-3.0,3.);

   hi_mutime_vtx = new TH1F("hi_mutime_vtx","Time at Vertex (inout)",theNBins,-25.*theScale,25.*theScale);
   hi_mutime_vtx_err = new TH1F("hi_mutime_vtx_err","Time at Vertex Error (inout)",theNBins,0.,25.0);

   hi_dtcsc_vtx = new TH1F("hi_dtcsc_vtx","Time at Vertex (DT-CSC)",theNBins,-25.*theScale,25.*theScale);
   hi_dtecal_vtx = new TH1F("hi_dtecal_vtx","Time at Vertex (DT-ECAL)",theNBins,-25.*theScale,25.*theScale);
   hi_ecalcsc_vtx = new TH1F("hi_ecalcsc_vtx","Time at Vertex (ECAL-CSC)",theNBins,-25.*theScale,25.*theScale);
   hi_dthcal_vtx = new TH1F("hi_dthcal_vtx","Time at Vertex (DT-HCAL)",theNBins,-25.*theScale,25.*theScale);
   hi_hcalcsc_vtx = new TH1F("hi_hcalcsc_vtx","Time at Vertex (HCAL-CSC)",theNBins,-25.*theScale,25.*theScale);
   hi_hcalecal_vtx = new TH1F("hi_hcalecal_vtx","Time at Vertex (HCAL-ECAL)",theNBins,-25.*theScale,25.*theScale);

   hi_cmbtime_ibt = new TH1F("hi_cmbtime_ibt","Inverse Beta",theNBins,0.,1.6);
   hi_cmbtime_ibt_pt = new TH2F("hi_cmbtime_ibt_pt","P{T} vs Inverse Beta",theNBins,0.0,theMaxPtres,theNBins,0.7,2.0);
   hi_cmbtime_ibt_err = new TH1F("hi_cmbtime_ibt_err","Inverse Beta Error",theNBins,0.,1.0);
   hi_cmbtime_fib = new TH1F("hi_cmbtime_fib","Free Inverse Beta",theNBins,-5.*theScale,7.*theScale);
   hi_cmbtime_fib_err = new TH1F("hi_cmbtime_fib_err","Free Inverse Beta Error",theNBins,0,5.);
   hi_cmbtime_vtx = new TH1F("hi_cmbtime_vtx","Time at Vertex (inout)",theNBins,-25.*theScale,25.*theScale);
   hi_cmbtime_vtx_err = new TH1F("hi_cmbtime_vtx_err","Time at Vertex Error (inout)",theNBins,0.,25.0);
   hi_cmbtime_vtxr = new TH1F("hi_cmbtime_vtxR","Time at Vertex (inout)",theNBins,0.,75.*theScale);
   hi_cmbtime_vtxr_err = new TH1F("hi_cmbtime_vtxR_err","Time at Vertex Error (inout)",theNBins,0.,25.0);
   hi_cmbtime_ibt_pull = new TH1F("hi_cmbtime_ibt_pull","Inverse Beta Pull",theNBins,-5.,5.0);
   hi_cmbtime_fib_pull = new TH1F("hi_cmbtime_fib_pull","Free Inverse Beta Pull",theNBins,-5.,5.0);
   hi_cmbtime_vtx_pull = new TH1F("hi_cmbtime_vtx_pull","Time at Vertex Pull (inout)",theNBins,-5.,5.0);
   hi_cmbtime_vtxr_pull = new TH1F("hi_cmbtime_vtxR_pull","Time at Vertex Pull (inout)",theNBins,-5.,5.0);

   hi_cmbtime_ndof = new TH1F("hi_cmbtime_ndof","Number of timing measurements",48,0.,48.0);

   hi_dttime_ibt = new TH1F("hi_dttime_ibt","DT Inverse Beta",theNBins,0.,1.6);
   hi_dttime_ibt_pt = new TH2F("hi_dttime_ibt_pt","P{T} vs DT Inverse Beta",theNBins,0.0,theMaxPtres,theNBins,0.7,2.0);
   hi_dttime_ibt_err = new TH1F("hi_dttime_ibt_err","DT Inverse Beta Error",theNBins,0.,0.3);
   hi_dttime_fib = new TH1F("hi_dttime_fib","DT Free Inverse Beta",theNBins,-5.*theScale,7.*theScale);
   hi_dttime_fib_err = new TH1F("hi_dttime_fib_err","DT Free Inverse Beta Error",theNBins,0,5.);
   hi_dttime_vtx = new TH1F("hi_dttime_vtx","DT Time at Vertex (inout)",theNBins,-25.*theScale,25.*theScale);
   hi_dttime_vtx_err = new TH1F("hi_dttime_vtx_err","DT Time at Vertex Error (inout)",theNBins,0.,10.0);
   hi_dttime_vtxr = new TH1F("hi_dttime_vtxR","DT Time at Vertex (inout)",theNBins,0.,75.*theScale);
   hi_dttime_vtxr_err = new TH1F("hi_dttime_vtxR_err","DT Time at Vertex Error (inout)",theNBins,0.,10.0);
   hi_dttime_ibt_pull = new TH1F("hi_dttime_ibt_pull","DT Inverse Beta Pull",theNBins,-5.,5.0);
   hi_dttime_fib_pull = new TH1F("hi_dttime_fib_pull","DT Free Inverse Beta Pull",theNBins,-5.,5.0);
   hi_dttime_vtx_pull = new TH1F("hi_dttime_vtx_pull","DT Time at Vertex Pull (inout)",theNBins,-5.,5.0);
   hi_dttime_vtxr_pull = new TH1F("hi_dttime_vtxR_pull","DT Time at Vertex Pull (inout)",theNBins,-5.,5.0);
   hi_dttime_errdiff = new TH1F("hi_dttime_errdiff","DT Time at Vertex inout-outin error difference",theNBins,-2.*theScale,2.*theScale);

   hi_dttime_ndof = new TH1F("hi_dttime_ndof","Number of DT timing measurements",48,0.,48.0);

   hi_csctime_ibt = new TH1F("hi_csctime_ibt","CSC Inverse Beta",theNBins,0.,1.6);
   hi_csctime_ibt_pt = new TH2F("hi_csctime_ibt_pt","P{T} vs CSC Inverse Beta",theNBins,0.0,theMaxPtres,theNBins,0.7,2.0);
   hi_csctime_ibt_err = new TH1F("hi_csctime_ibt_err","CSC Inverse Beta Error",theNBins,0.,1.0);
   hi_csctime_fib = new TH1F("hi_csctime_fib","CSC Free Inverse Beta",theNBins,-5.*theScale,7.*theScale);
   hi_csctime_fib_err = new TH1F("hi_csctime_fib_err","CSC Free Inverse Beta Error",theNBins,0,5.);
   hi_csctime_vtx = new TH1F("hi_csctime_vtx","CSC Time at Vertex (inout)",theNBins,-25.*theScale,25.*theScale);
   hi_csctime_vtx_err = new TH1F("hi_csctime_vtx_err","CSC Time at Vertex Error (inout)",theNBins,0.,25.0);
   hi_csctime_vtxr = new TH1F("hi_csctime_vtxR","CSC Time at Vertex (inout)",theNBins,0.,75.*theScale);
   hi_csctime_vtxr_err = new TH1F("hi_csctime_vtxR_err","CSC Time at Vertex Error (inout)",theNBins,0.,25.0);
   hi_csctime_ibt_pull = new TH1F("hi_csctime_ibt_pull","CSC Inverse Beta Pull",theNBins,-5.,5.0);
   hi_csctime_fib_pull = new TH1F("hi_csctime_fib_pull","CSC Free Inverse Beta Pull",theNBins,-5.,5.0);
   hi_csctime_vtx_pull = new TH1F("hi_csctime_vtx_pull","CSC Time at Vertex Pull (inout)",theNBins,-5.,5.0);
   hi_csctime_vtxr_pull = new TH1F("hi_csctime_vtxR_pull","CSC Time at Vertex Pull (inout)",theNBins,-5.,5.0);

   hi_csctime_ndof = new TH1F("hi_csctime_ndof","Number of CSC timing measurements",48,0.,48.0);

   hi_ecal_time = new TH1F("hi_ecal_time","ECAL Time at Vertex (inout)",theNBins,-40.*theScale,40.*theScale);
   hi_ecal_time_err = new TH1F("hi_ecal_time_err","ECAL Time at Vertex Error",theNBins,0.,20.0);
   hi_ecal_time_pull = new TH1F("hi_ecal_time_pull","ECAL Time at Vertex Pull",theNBins,-7.0,7.0);
   hi_ecal_time_ecut = new TH1F("hi_ecal_time_ecut","ECAL Time at Vertex (inout) after energy cut",theNBins,-20.*theScale,20.*theScale);
   hi_ecal_energy = new TH1F("hi_ecal_energy","ECAL max energy in 5x5 crystals",theNBins,.0,5.0);

   hi_hcal_time = new TH1F("hi_hcal_time","HCAL Time at Vertex (inout)",theNBins,-40.*theScale,40.*theScale);
   hi_hcal_time_err = new TH1F("hi_hcal_time_err","HCAL Time at Vertex Error",theNBins,0.,20.0);
   hi_hcal_time_pull = new TH1F("hi_hcal_time_pull","HCAL Time at Vertex Pull",theNBins,-7.0,7.0);
   hi_hcal_time_ecut = new TH1F("hi_hcal_time_ecut","HCAL Time at Vertex (inout) after energy cut",theNBins,-20.*theScale,20.*theScale);
   hi_hcal_energy = new TH1F("hi_hcal_energy","HCAL max energy in 5x5 crystals",theNBins,.0,5.0);

   hi_sta_eta = new TH1F("hi_sta_eta","#eta^{STA}",theNBins/2,theMinEta,theMaxEta);
   hi_tk_eta  = new TH1F("hi_tk_eta","#eta^{TK}",theNBins/2,theMinEta,theMaxEta);
   hi_glb_eta = new TH1F("hi_glb_eta","#eta^{GLB}",theNBins/2,theMinEta,theMaxEta);

}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonTimingValidator::endJob() {

  hFile->cd();

  gROOT->SetStyle("effStyle");

  hi_sta_pt->Write();
  hi_tk_pt->Write();
  hi_glb_pt->Write();

  hi_sta_phi->Write();
  hi_tk_phi->Write();
  hi_glb_phi->Write();

  hi_sta_eta->Write();
  hi_tk_eta->Write();
  hi_glb_eta->Write();

  hi_mutime_vtx->Write();
  hi_mutime_vtx_err->Write();

  hFile->mkdir("differences");
  hFile->cd("differences");

  hi_dtcsc_vtx->Write();
  hi_dtecal_vtx->Write();
  hi_ecalcsc_vtx->Write();
  hi_dthcal_vtx->Write();
  hi_hcalcsc_vtx->Write();
  hi_hcalecal_vtx->Write();

  hFile->cd();
  hFile->mkdir("combined");
  hFile->cd("combined");

  hi_cmbtime_ibt->Write();
  hi_cmbtime_ibt_pt->Write();
  hi_cmbtime_ibt_err->Write();
  hi_cmbtime_fib->Write();
  hi_cmbtime_fib_err->Write();
  hi_cmbtime_vtx->Write();
  hi_cmbtime_vtx_err->Write();
  hi_cmbtime_vtxr->Write();
  hi_cmbtime_vtxr_err->Write();
  hi_cmbtime_ibt_pull->Write();
  hi_cmbtime_fib_pull->Write();
  hi_cmbtime_vtx_pull->Write();
  hi_cmbtime_vtxr_pull->Write();
  hi_cmbtime_ndof->Write();

  hFile->cd();
  hFile->mkdir("dt");
  hFile->cd("dt");

  hi_dttime_ibt->Write();
  hi_dttime_ibt_pt->Write();
  hi_dttime_ibt_err->Write();
  hi_dttime_fib->Write();
  hi_dttime_fib_err->Write();
  hi_dttime_vtx->Write();
  hi_dttime_vtx_err->Write();
  hi_dttime_vtxr->Write();
  hi_dttime_vtxr_err->Write();
  hi_dttime_ibt_pull->Write();
  hi_dttime_fib_pull->Write();
  hi_dttime_vtx_pull->Write();
  hi_dttime_vtxr_pull->Write();
  hi_dttime_errdiff->Write();
  hi_dttime_ndof->Write();

  hFile->cd();
  hFile->mkdir("csc");
  hFile->cd("csc");

  hi_csctime_ibt->Write();
  hi_csctime_ibt_pt->Write();
  hi_csctime_ibt_err->Write();
  hi_csctime_fib->Write();
  hi_csctime_fib_err->Write();
  hi_csctime_vtx->Write();
  hi_csctime_vtx_err->Write();
  hi_csctime_vtxr->Write();
  hi_csctime_vtxr_err->Write();
  hi_csctime_ibt_pull->Write();
  hi_csctime_fib_pull->Write();
  hi_csctime_vtx_pull->Write();
  hi_csctime_vtxr_pull->Write();
  hi_csctime_ndof->Write();

  hFile->cd();
  hFile->mkdir("ecal");
  hFile->cd("ecal");

  hi_ecal_time->Write();
  hi_ecal_time_ecut->Write();
  hi_ecal_time_err->Write();
  hi_ecal_time_pull->Write();
  hi_ecal_energy->Write();

  hFile->cd();
  hFile->mkdir("hcal");
  hFile->cd("hcal");

  hi_hcal_time->Write();
  hi_hcal_time_ecut->Write();
  hi_hcal_time_err->Write();
  hi_hcal_time_pull->Write();
  hi_hcal_energy->Write();

  hFile->Write();
}

float 
MuonTimingValidator::calculateDistance(const math::XYZVector& vect1, const math::XYZVector& vect2) {
  float dEta = vect1.eta() - vect2.eta();
  float dPhi = fabs(Geom::Phi<float>(vect1.phi()) - Geom::Phi<float>(vect2.phi()));
  float distance = sqrt(pow(dEta,2) + pow(dPhi,2) );

  return distance;
}

//
// return h1/h2 with recalculated errors
//
TH1F* MuonTimingValidator::divideErr(TH1F* h1, TH1F* h2, TH1F* hout) {

  hout->Reset();
  hout->Divide(h1,h2,1.,1.,"B");

  for (int i = 0; i <= hout->GetNbinsX()+1; i++ ) {
    Float_t tot   = h2->GetBinContent(i) ;
    Float_t tot_e = h2->GetBinError(i);
    Float_t eff = hout->GetBinContent(i) ;
    Float_t Err = 0.;
    if (tot > 0) Err = tot_e / tot * sqrt( eff* (1-eff) );
    if (eff == 1. || edm::isNotFinite(Err)) Err=1.e-3;
    hout->SetBinError(i, Err);
  }
  return hout;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonTimingValidator);
