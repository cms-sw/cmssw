// -*- C++ -*-
//
// Package:    RecoEgamma/Examples
// Class:      GsfElectronDataAnalyzer
//
/**\class GsfElectronDataAnalyzer RecoEgamma/Examples/src/GsfElectronDataAnalyzer.cc

 Description: GsfElectrons analyzer using reco data

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//
//

// user include files
#include "RecoEgamma/Examples/plugins/DQMAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include <iostream>
#include "TMath.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"
#include <iostream>
#include <cassert>

using namespace reco;

DQMAnalyzer::DQMAnalyzer(const edm::ParameterSet &conf) : beamSpot_(conf.getParameter<edm::InputTag>("beamSpot")) {
  outputFile_ = conf.getParameter<std::string>("outputFile");
  electronCollection_ = conf.getParameter<edm::InputTag>("electronCollection");
  matchingObjectCollection_ = conf.getParameter<edm::InputTag>("matchingObjectCollection");
  matchingCondition_ = conf.getParameter<std::string>("matchingCondition");
  readAOD_ = conf.getParameter<bool>("readAOD");

  histfile_ = new TFile(outputFile_.c_str(), "RECREATE");

  // currently only one possible matching conditions
  assert(matchingCondition_ == "Cone");
  maxPtMatchingObject_ = conf.getParameter<double>("MaxPtMatchingObject");
  maxAbsEtaMatchingObject_ = conf.getParameter<double>("MaxAbsEtaMatchingObject");
  deltaR_ = conf.getParameter<double>("DeltaR");

  Selection_ = conf.getParameter<int>("Selection");
  massLow_ = conf.getParameter<double>("MassLow");
  massHigh_ = conf.getParameter<double>("MassHigh");
  TPchecksign_ = conf.getParameter<bool>("TPchecksign");
  TAGcheckclass_ = conf.getParameter<bool>("TAGcheckclass");
  PROBEetcut_ = conf.getParameter<bool>("PROBEetcut");
  PROBEcheckclass_ = conf.getParameter<bool>("PROBEcheckclass");

  minEt_ = conf.getParameter<double>("MinEt");
  minPt_ = conf.getParameter<double>("MinPt");
  maxAbsEta_ = conf.getParameter<double>("MaxAbsEta");
  isEB_ = conf.getParameter<bool>("SelectEB");
  isEE_ = conf.getParameter<bool>("SelectEE");
  isNotEBEEGap_ = conf.getParameter<bool>("SelectNotEBEEGap");
  isEcalDriven_ = conf.getParameter<bool>("SelectEcalDriven");
  isTrackerDriven_ = conf.getParameter<bool>("SelectTrackerDriven");
  eOverPMinBarrel_ = conf.getParameter<double>("MinEOverPBarrel");
  eOverPMaxBarrel_ = conf.getParameter<double>("MaxEOverPBarrel");
  eOverPMinEndcaps_ = conf.getParameter<double>("MinEOverPEndcaps");
  eOverPMaxEndcaps_ = conf.getParameter<double>("MaxEOverPEndcaps");
  dEtaMinBarrel_ = conf.getParameter<double>("MinDetaBarrel");
  dEtaMaxBarrel_ = conf.getParameter<double>("MaxDetaBarrel");
  dEtaMinEndcaps_ = conf.getParameter<double>("MinDetaEndcaps");
  dEtaMaxEndcaps_ = conf.getParameter<double>("MaxDetaEndcaps");
  dPhiMinBarrel_ = conf.getParameter<double>("MinDphiBarrel");
  dPhiMaxBarrel_ = conf.getParameter<double>("MaxDphiBarrel");
  dPhiMinEndcaps_ = conf.getParameter<double>("MinDphiEndcaps");
  dPhiMaxEndcaps_ = conf.getParameter<double>("MaxDphiEndcaps");
  sigIetaIetaMinBarrel_ = conf.getParameter<double>("MinSigIetaIetaBarrel");
  sigIetaIetaMaxBarrel_ = conf.getParameter<double>("MaxSigIetaIetaBarrel");
  sigIetaIetaMinEndcaps_ = conf.getParameter<double>("MinSigIetaIetaEndcaps");
  sigIetaIetaMaxEndcaps_ = conf.getParameter<double>("MaxSigIetaIetaEndcaps");
  hadronicOverEmMaxBarrel_ = conf.getParameter<double>("MaxHoEBarrel");
  hadronicOverEmMaxEndcaps_ = conf.getParameter<double>("MaxHoEEndcaps");
  mvaMin_ = conf.getParameter<double>("MinMVA");
  tipMaxBarrel_ = conf.getParameter<double>("MaxTipBarrel");
  tipMaxEndcaps_ = conf.getParameter<double>("MaxTipEndcaps");
  tkIso03Max_ = conf.getParameter<double>("MaxTkIso03");
  hcalIso03Depth1MaxBarrel_ = conf.getParameter<double>("MaxHcalIso03Depth1Barrel");
  hcalIso03Depth1MaxEndcaps_ = conf.getParameter<double>("MaxHcalIso03Depth1Endcaps");
  hcalIso03Depth2MaxEndcaps_ = conf.getParameter<double>("MaxHcalIso03Depth2Endcaps");
  ecalIso03MaxBarrel_ = conf.getParameter<double>("MaxEcalIso03Barrel");
  ecalIso03MaxEndcaps_ = conf.getParameter<double>("MaxEcalIso03Endcaps");

  triggerResults_ = conf.getParameter<edm::InputTag>("triggerResults");
  HLTPathsByName_ = conf.getParameter<std::vector<std::string> >("hltPaths");
  HLTPathsByIndex_.resize(HLTPathsByName_.size());

  edm::ParameterSet pset = conf.getParameter<edm::ParameterSet>("HistosConfigurationData");

  etamin = pset.getParameter<double>("Etamin");
  etamax = pset.getParameter<double>("Etamax");
  phimin = pset.getParameter<double>("Phimin");
  phimax = pset.getParameter<double>("Phimax");
  ptmax = pset.getParameter<double>("Ptmax");
  pmax = pset.getParameter<double>("Pmax");
  eopmax = pset.getParameter<double>("Eopmax");
  eopmaxsht = pset.getParameter<double>("Eopmaxsht");
  detamin = pset.getParameter<double>("Detamin");
  detamax = pset.getParameter<double>("Detamax");
  dphimin = pset.getParameter<double>("Dphimin");
  dphimax = pset.getParameter<double>("Dphimax");
  detamatchmin = pset.getParameter<double>("Detamatchmin");
  detamatchmax = pset.getParameter<double>("Detamatchmax");
  dphimatchmin = pset.getParameter<double>("Dphimatchmin");
  dphimatchmax = pset.getParameter<double>("Dphimatchmax");
  fhitsmax = pset.getParameter<double>("Fhitsmax");
  lhitsmax = pset.getParameter<double>("Lhitsmax");
  nbineta = pset.getParameter<int>("Nbineta");
  nbineta2D = pset.getParameter<int>("Nbineta2D");
  nbinp = pset.getParameter<int>("Nbinp");
  nbinpt = pset.getParameter<int>("Nbinpt");
  nbinp2D = pset.getParameter<int>("Nbinp2D");
  nbinpt2D = pset.getParameter<int>("Nbinpt2D");
  nbinpteff = pset.getParameter<int>("Nbinpteff");
  nbinphi = pset.getParameter<int>("Nbinphi");
  nbinphi2D = pset.getParameter<int>("Nbinphi2D");
  nbineop = pset.getParameter<int>("Nbineop");
  nbineop2D = pset.getParameter<int>("Nbineop2D");
  nbinfhits = pset.getParameter<int>("Nbinfhits");
  nbinlhits = pset.getParameter<int>("Nbinlhits");
  nbinxyz = pset.getParameter<int>("Nbinxyz");
  nbindeta = pset.getParameter<int>("Nbindeta");
  nbindphi = pset.getParameter<int>("Nbindphi");
  nbindetamatch = pset.getParameter<int>("Nbindetamatch");
  nbindphimatch = pset.getParameter<int>("Nbindphimatch");
  nbindetamatch2D = pset.getParameter<int>("Nbindetamatch2D");
  nbindphimatch2D = pset.getParameter<int>("Nbindphimatch2D");
  nbinpoptrue = pset.getParameter<int>("Nbinpoptrue");
  poptruemin = pset.getParameter<double>("Poptruemin");
  poptruemax = pset.getParameter<double>("Poptruemax");
  nbinmee = pset.getParameter<int>("Nbinmee");
  meemin = pset.getParameter<double>("Meemin");
  meemax = pset.getParameter<double>("Meemax");
  nbinhoe = pset.getParameter<int>("Nbinhoe");
  hoemin = pset.getParameter<double>("Hoemin");
  hoemax = pset.getParameter<double>("Hoemax");
}

DQMAnalyzer::~DQMAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  histfile_->Write();
  histfile_->Close();
}

void DQMAnalyzer::beginJob() {
  histfile_->cd();

  nEvents_ = 0;
  nAfterTrigger_ = 0;

  // matching object
  std::string::size_type locSC = matchingObjectCollection_.label().find("SuperCluster", 0);
  std::string type;
  if (locSC != std::string::npos) {
    std::cout << "Matching objects are SuperClusters " << std::endl;
    type = "SC";
  } else {
    std::cout << "Didn't recognize input matching objects!! " << std::endl;
  }

  //==================================================
  // matching object distributions
  //==================================================

  std::string htitle, hlabel;
  hlabel = "h_" + type + "Num";
  htitle = "# " + type + "s";
  h_matchingObjectNum = new TH1F(hlabel.c_str(), htitle.c_str(), nbinfhits, 0., fhitsmax);
  hlabel = "h_" + type + "_eta";
  htitle = type + " #eta";
  h_matchingObjectEta = new TH1F(hlabel.c_str(), htitle.c_str(), nbineta, etamin, etamax);
  hlabel = "h_" + type + "_abseta";
  htitle = type + " |#eta|";
  h_matchingObjectAbsEta = new TH1F(hlabel.c_str(), htitle.c_str(), nbineta / 2, 0., etamax);
  hlabel = "h_" + type + "_P";
  htitle = type + " p";
  h_matchingObjectP = new TH1F(hlabel.c_str(), htitle.c_str(), nbinp, 0., pmax);
  hlabel = "h_" + type + "_Pt";
  htitle = type + " pt";
  h_matchingObjectPt = new TH1F(hlabel.c_str(), htitle.c_str(), nbinpteff, 5., ptmax);
  hlabel = "h_" + type + "_phi";
  htitle = type + " phi";
  h_matchingObjectPhi = new TH1F(hlabel.c_str(), htitle.c_str(), nbinphi, phimin, phimax);
  hlabel = "h_" + type + "_z";
  htitle = type + " z";
  h_matchingObjectZ = new TH1F(hlabel.c_str(), htitle.c_str(), nbinxyz, -25, 25);

  h_matchingObjectNum->GetXaxis()->SetTitle("N_{SC}");
  h_matchingObjectNum->GetYaxis()->SetTitle("Events");
  h_matchingObjectEta->GetXaxis()->SetTitle("#eta_{SC}");
  h_matchingObjectEta->GetYaxis()->SetTitle("Events");
  h_matchingObjectP->GetXaxis()->SetTitle("E_{SC} (GeV)");
  h_matchingObjectP->GetYaxis()->SetTitle("Events");

  h_ele_matchingObjectEta_matched =
      new TH1F("h_ele_matchingObjectEta_matched", "Efficiency vs matching SC #eta", nbineta, etamin, etamax);
  h_ele_matchingObjectEta_matched->Sumw2();
  h_ele_matchingObjectAbsEta_matched =
      new TH1F("h_ele_matchingObjectAbsEta_matched", "Efficiency vs matching SC |#eta|", nbineta / 2, 0., 2.5);
  h_ele_matchingObjectAbsEta_matched->Sumw2();
  h_ele_matchingObjectPt_matched =
      new TH1F("h_ele_matchingObjectPt_matched", "Efficiency vs matching SC E_{T}", nbinpteff, 5., ptmax);
  h_ele_matchingObjectPt_matched->Sumw2();
  h_ele_matchingObjectPhi_matched =
      new TH1F("h_ele_matchingObjectPhi_matched", "Efficiency vs matching SC phi", nbinphi, phimin, phimax);
  h_ele_matchingObjectPhi_matched->Sumw2();
  h_ele_matchingObjectZ_matched =
      new TH1F("h_ele_matchingObjectZ_matched", "Efficiency vs matching SC z", nbinxyz, -25, 25);
  h_ele_matchingObjectZ_matched->Sumw2();

  //==================================================
  // caractéristique particule
  //==================================================

  h_ele_vertexPt = new TH1F("h_ele_vertexPt", "ele transverse momentum", nbinpt, 0., ptmax);
  h_ele_Et = new TH1F("h_ele_Et", "ele SC transverse energy", nbinpt, 0., ptmax);
  h_ele_vertexEta = new TH1F("h_ele_vertexEta", "ele momentum eta", nbineta, etamin, etamax);
  h_ele_vertexPhi = new TH1F("h_ele_vertexPhi", "ele  momentum #phi", nbinphi, phimin, phimax);
  h_ele_vertexX = new TH1F("h_ele_vertexX", "ele vertex x", nbinxyz, -0.1, 0.1);
  h_ele_vertexY = new TH1F("h_ele_vertexY", "ele vertex y", nbinxyz, -0.1, 0.1);
  h_ele_vertexZ = new TH1F("h_ele_vertexZ", "ele vertex z", nbinxyz, -25, 25);
  h_ele_vertexTIP = new TH1F("h_ele_vertexTIP", "ele transverse impact parameter (wrt bs)", 90, 0., 0.15);
  h_ele_charge = new TH1F("h_ele_charge", "ele charge", 5, -2., 2.);

  h_ele_charge->GetXaxis()->SetTitle("charge");
  h_ele_charge->GetYaxis()->SetTitle("Events");

  h_ele_vertexPt->GetXaxis()->SetTitle("p_{T vertex} (GeV/c)");
  h_ele_vertexPt->GetYaxis()->SetTitle("Events");

  h_ele_Et->GetXaxis()->SetTitle("E_{T} (GeV)");
  h_ele_Et->GetYaxis()->SetTitle("Events");

  h_ele_vertexEta->GetXaxis()->SetTitle("#eta");
  h_ele_vertexEta->GetYaxis()->SetTitle("Events");
  h_ele_vertexPhi->GetXaxis()->SetTitle("#phi (rad)");
  h_ele_vertexPhi->GetYaxis()->SetTitle("Events");

  h_ele_vertexX->GetXaxis()->SetTitle("x (cm)");
  h_ele_vertexX->GetYaxis()->SetTitle("Events");
  h_ele_vertexY->GetXaxis()->SetTitle("y (cm)");
  h_ele_vertexY->GetYaxis()->SetTitle("Events");
  h_ele_vertexZ->GetXaxis()->SetTitle("z (cm)");
  h_ele_vertexZ->GetYaxis()->SetTitle("Events");

  h_ele_vertexTIP->GetXaxis()->SetTitle("TIP (cm)");
  h_ele_vertexTIP->GetYaxis()->SetTitle("Events");

  //==================================================
  // # rec electrons
  //==================================================

  histNum_ = new TH1F("h_recEleNum", "# rec electrons", 20, 0., 20.);

  histNum_->GetXaxis()->SetTitle("N_{ele}");
  histNum_->GetYaxis()->SetTitle("Events");

  //==================================================
  // SuperClusters
  //==================================================

  histSclEn_ = new TH1F("h_scl_energy", "ele supercluster energy", nbinp, 0., pmax);
  histSclEt_ = new TH1F("h_scl_et", "ele supercluster transverse energy", nbinpt, 0., ptmax);
  histSclEta_ = new TH1F("h_scl_eta", "ele supercluster eta", nbineta, etamin, etamax);
  histSclPhi_ = new TH1F("h_scl_phi", "ele supercluster phi", nbinphi, phimin, phimax);
  histSclSigEtaEta_ = new TH1F("h_scl_sigetaeta", "ele supercluster sigma eta eta", 100, 0., 0.05);

  //==================================================
  // electron track
  //==================================================

  h_ele_ambiguousTracks = new TH1F("h_ele_ambiguousTracks", "ele # ambiguous tracks", 5, 0., 5.);
  h_ele_ambiguousTracksVsEta =
      new TH2F("h_ele_ambiguousTracksVsEta", "ele # ambiguous tracks  vs eta", nbineta2D, etamin, etamax, 5, 0., 5.);
  h_ele_ambiguousTracksVsPhi =
      new TH2F("h_ele_ambiguousTracksVsPhi", "ele # ambiguous tracks  vs phi", nbinphi2D, phimin, phimax, 5, 0., 5.);
  h_ele_ambiguousTracksVsPt =
      new TH2F("h_ele_ambiguousTracksVsPt", "ele # ambiguous tracks vs pt", nbinpt2D, 0., ptmax, 5, 0., 5.);
  h_ele_foundHits = new TH1F("h_ele_foundHits", "ele track # found hits", nbinfhits, 0., fhitsmax);
  h_ele_foundHitsVsEta = new TH2F(
      "h_ele_foundHitsVsEta", "ele track # found hits vs eta", nbineta2D, etamin, etamax, nbinfhits, 0., fhitsmax);
  h_ele_foundHitsVsPhi = new TH2F(
      "h_ele_foundHitsVsPhi", "ele track # found hits vs phi", nbinphi2D, phimin, phimax, nbinfhits, 0., fhitsmax);
  h_ele_foundHitsVsPt =
      new TH2F("h_ele_foundHitsVsPt", "ele track # found hits vs pt", nbinpt2D, 0., ptmax, nbinfhits, 0., fhitsmax);
  h_ele_lostHits = new TH1F("h_ele_lostHits", "ele track # lost hits", 5, 0., 5.);
  h_ele_lostHitsVsEta = new TH2F(
      "h_ele_lostHitsVsEta", "ele track # lost hits vs eta", nbineta2D, etamin, etamax, nbinlhits, 0., lhitsmax);
  h_ele_lostHitsVsPhi = new TH2F(
      "h_ele_lostHitsVsPhi", "ele track # lost hits vs eta", nbinphi2D, phimin, phimax, nbinlhits, 0., lhitsmax);
  h_ele_lostHitsVsPt =
      new TH2F("h_ele_lostHitsVsPt", "ele track # lost hits vs eta", nbinpt2D, 0., ptmax, nbinlhits, 0., lhitsmax);
  h_ele_chi2 = new TH1F("h_ele_chi2", "ele track #chi^{2}", 100, 0., 15.);
  h_ele_chi2VsEta = new TH2F("h_ele_chi2VsEta", "ele track #chi^{2} vs eta", nbineta2D, etamin, etamax, 50, 0., 15.);
  h_ele_chi2VsPhi = new TH2F("h_ele_chi2VsPhi", "ele track #chi^{2} vs phi", nbinphi2D, phimin, phimax, 50, 0., 15.);
  h_ele_chi2VsPt = new TH2F("h_ele_chi2VsPt", "ele track #chi^{2} vs pt", nbinpt2D, 0., ptmax, 50, 0., 15.);

  h_ele_foundHits->GetXaxis()->SetTitle("N_{hits}");
  h_ele_foundHits->GetYaxis()->SetTitle("Events");
  h_ele_lostHits->GetXaxis()->SetTitle("N_{lost hits}");
  h_ele_lostHits->GetYaxis()->SetTitle("Events");
  h_ele_chi2->GetXaxis()->SetTitle("#Chi^{2}");
  h_ele_chi2->GetYaxis()->SetTitle("Events");

  //==================================================
  // electron matching and ID
  //==================================================

  h_ele_EoP = new TH1F("h_ele_EoP", "ele E/P_{vertex}", nbineop, 0., eopmax);
  //  h_ele_EoPout         = new TH1F( "h_ele_EoPout",         "ele E/P_{out}",           nbineop,0.,eopmax);
  h_ele_EeleOPout = new TH1F("h_ele_EeleOPout", "ele E_{ele}/P_{out}", nbineop, 0., eopmax);
  h_ele_dEtaSc_propVtx = new TH1F(
      "h_ele_dEtaSc_propVtx", "ele #eta_{sc} - #eta_{tr}, prop from vertex", nbindetamatch, detamatchmin, detamatchmax);
  h_ele_dPhiSc_propVtx = new TH1F(
      "h_ele_dPhiSc_propVtx", "ele #phi_{sc} - #phi_{tr}, prop from vertex", nbindphimatch, dphimatchmin, dphimatchmax);
  h_ele_dEtaCl_propOut = new TH1F("h_ele_dEtaCl_propOut",
                                  "ele #eta_{cl} - #eta_{tr}, prop from outermost",
                                  nbindetamatch,
                                  detamatchmin,
                                  detamatchmax);
  h_ele_dPhiCl_propOut = new TH1F("h_ele_dPhiCl_propOut",
                                  "ele #phi_{cl} - #phi_{tr}, prop from outermost",
                                  nbindphimatch,
                                  dphimatchmin,
                                  dphimatchmax);
  h_ele_dEtaEleCl_propOut = new TH1F("h_ele_dEtaEleCl_propOut",
                                     "ele #eta_{EleCl} - #eta_{tr}, prop from outermost",
                                     nbindetamatch,
                                     detamatchmin,
                                     detamatchmax);
  h_ele_dPhiEleCl_propOut = new TH1F("h_ele_dPhiEleCl_propOut",
                                     "ele #phi_{EleCl} - #phi_{tr}, prop from outermost",
                                     nbindphimatch,
                                     dphimatchmin,
                                     dphimatchmax);
  h_ele_HoE = new TH1F("h_ele_HoE", "ele hadronic energy / em energy", nbinhoe, hoemin, hoemax);
  h_ele_outerP = new TH1F("h_ele_outerP", "ele track outer p, mean", nbinp, 0., pmax);
  h_ele_outerP_mode = new TH1F("h_ele_outerP_mode", "ele track outer p, mode", nbinp, 0., pmax);
  h_ele_outerPt = new TH1F("h_ele_outerPt", "ele track outer p_{T}, mean", nbinpt, 0., ptmax);
  h_ele_outerPt_mode = new TH1F("h_ele_outerPt_mode", "ele track outer p_{T}, mode", nbinpt, 0., ptmax);

  h_ele_PinMnPout = new TH1F("h_ele_PinMnPout", "ele track inner p - outer p, mean", nbinp, 0., 200.);
  h_ele_PinMnPout_mode = new TH1F("h_ele_PinMnPout_mode", "ele track inner p - outer p, mode", nbinp, 0., 100.);

  h_ele_mva = new TH1F("h_ele_mva", "ele identification mva", 100, -1., 1.);
  h_ele_provenance = new TH1F("h_ele_provenance", "ele provenance", 5, -2., 3.);

  h_ele_PinMnPout->GetXaxis()->SetTitle("P_{vertex} - P_{out} (GeV/c)");
  h_ele_PinMnPout->GetYaxis()->SetTitle("Events");
  h_ele_PinMnPout_mode->GetXaxis()->SetTitle("P_{vertex} - P_{out}, mode (GeV/c)");
  h_ele_PinMnPout_mode->GetYaxis()->SetTitle("Events");

  h_ele_outerP->GetXaxis()->SetTitle("P_{out} (GeV/c)");
  h_ele_outerP->GetYaxis()->SetTitle("Events");
  h_ele_outerP_mode->GetXaxis()->SetTitle("P_{out} (GeV/c)");
  h_ele_outerP_mode->GetYaxis()->SetTitle("Events");

  h_ele_outerPt->GetXaxis()->SetTitle("P_{T out} (GeV/c)");
  h_ele_outerPt->GetYaxis()->SetTitle("Events");
  h_ele_outerPt_mode->GetXaxis()->SetTitle("P_{T out} (GeV/c)");
  h_ele_outerPt_mode->GetYaxis()->SetTitle("Events");

  h_ele_EoP->GetXaxis()->SetTitle("E/P_{vertex}");
  h_ele_EoP->GetYaxis()->SetTitle("Events");

  //  h_ele_EoPout->GetXaxis()-> SetTitle("E_{seed}/P_{out}");
  //  h_ele_EoPout->GetYaxis()-> SetTitle("Events");
  h_ele_EeleOPout->GetXaxis()->SetTitle("E_{ele}/P_{out}");
  h_ele_EeleOPout->GetYaxis()->SetTitle("Events");

  h_ele_dEtaSc_propVtx->GetXaxis()->SetTitle("#eta_{sc} - #eta_{tr}");
  h_ele_dEtaSc_propVtx->GetYaxis()->SetTitle("Events");
  h_ele_dEtaCl_propOut->GetXaxis()->SetTitle("#eta_{seedcl} - #eta_{tr}");
  h_ele_dEtaCl_propOut->GetYaxis()->SetTitle("Events");
  h_ele_dEtaEleCl_propOut->GetXaxis()->SetTitle("#eta_{elecl} - #eta_{tr}");
  h_ele_dEtaEleCl_propOut->GetYaxis()->SetTitle("Events");
  h_ele_dPhiSc_propVtx->GetXaxis()->SetTitle("#phi_{sc} - #phi_{tr} (rad)");
  h_ele_dPhiSc_propVtx->GetYaxis()->SetTitle("Events");
  h_ele_dPhiCl_propOut->GetXaxis()->SetTitle("#phi_{seedcl} - #phi_{tr} (rad)");
  h_ele_dPhiCl_propOut->GetYaxis()->SetTitle("Events");
  h_ele_dPhiEleCl_propOut->GetXaxis()->SetTitle("#phi_{elecl} - #phi_{tr} (rad)");
  h_ele_dPhiEleCl_propOut->GetYaxis()->SetTitle("Events");
  h_ele_HoE->GetXaxis()->SetTitle("H/E");
  h_ele_HoE->GetYaxis()->SetTitle("Events");

  //==================================================
  // isolation
  //==================================================

  h_ele_tkSumPt_dr03 = new TH1F("h_ele_tkSumPt_dr03", "tk isolation sum, dR=0.3", 100, 0.0, 20.);
  h_ele_ecalRecHitSumEt_dr03 = new TH1F("h_ele_ecalRecHitSumEt_dr03", "ecal isolation sum, dR=0.3", 100, 0.0, 20.);
  h_ele_hcalDepth1TowerSumEt_dr03 =
      new TH1F("h_ele_hcalDepth1TowerSumEt_dr03", "hcal depth1 isolation sum, dR=0.3", 100, 0.0, 20.);
  h_ele_hcalDepth2TowerSumEt_dr03 =
      new TH1F("h_ele_hcalDepth2TowerSumEt_dr03", "hcal depth2 isolation sum, dR=0.3", 100, 0.0, 20.);
  h_ele_tkSumPt_dr04 = new TH1F("h_ele_tkSumPt_dr04", "hcal isolation sum", 100, 0.0, 20.);
  h_ele_ecalRecHitSumEt_dr04 = new TH1F("h_ele_ecalRecHitSumEt_dr04", "ecal isolation sum, dR=0.4", 100, 0.0, 20.);
  h_ele_hcalDepth1TowerSumEt_dr04 =
      new TH1F("h_ele_hcalDepth1TowerSumEt_dr04", "hcal depth1 isolation sum, dR=0.4", 100, 0.0, 20.);
  h_ele_hcalDepth2TowerSumEt_dr04 =
      new TH1F("h_ele_hcalDepth2TowerSumEt_dr04", "hcal depth2 isolation sum, dR=0.4", 100, 0.0, 20.);

  //==================================================
  // T&P
  //==================================================
  h_ele_mee_os = new TH1F("h_ele_mee_os", "ele pairs invariant mass, opposite sign", nbinmee, meemin, meemax);

  //==================================================
  // OBSOLETE
  //==================================================

  //  h_ele_PtoPtmatchingObject_matched        = new TH1F( "h_ele_PtoPtmatchingObject_matched",        "ele trans momentum / matching SC trans energy", nbinpoptrue,poptruemin,poptruemax);
  //  h_ele_PtoPtmatchingObject_barrel_matched         = new TH1F( "h_ele_PtoPmatchingObject_barrel_matched",        "ele trans momentum / matching SC trans energy, barrel",nbinpoptrue,poptruemin,poptruemax);
  //  h_ele_PtoPtmatchingObject_endcaps_matched        = new TH1F( "h_ele_PtoPmatchingObject_endcaps_matched",        "ele trans momentum / matching SC trans energy, endcaps",nbinpoptrue,poptruemin,poptruemax);
  //  h_ele_PoPmatchingObject_matched        = new TH1F( "h_ele_PoPmatchingObject_matched",        "ele momentum / matching SC energy", nbinpoptrue,poptruemin,poptruemax);
  //  h_ele_PoPmatchingObject_barrel_matched         = new TH1F( "h_ele_PoPmatchingObject_barrel_matched",        "ele momentum / matching SC energy, barrel",nbinpoptrue,poptruemin,poptruemax);
  //  h_ele_PoPmatchingObject_endcaps_matched        = new TH1F( "h_ele_PoPmatchingObject_endcaps_matched",        "ele momentum / matching SC energy, endcaps",nbinpoptrue,poptruemin,poptruemax);
  //  // h_ele_PtoPtmatchingObject_matched        = new TH1F( "h_ele_PtoPtmatchingObject_matched",        "ele trans momentum / matching SC trans energy", nbinpoptrue,poptruemin,poptruemax);
  //  h_ele_EtaMnEtamatchingObject_matched   = new TH1F( "h_ele_EtaMnEtamatchingObject_matched",   "ele momentum eta - matching SC eta",nbindeta,detamin,detamax);
  //  h_ele_PhiMnPhimatchingObject_matched   = new TH1F( "h_ele_PhiMnPhimatchingObject_matched",   "ele momentum phi - matching SC phi",nbindphi,dphimin,dphimax);
  //  h_ele_PhiMnPhimatchingObject2_matched   = new TH1F( "h_ele_PhiMnPhimatchingObject2_matched",   "ele momentum phi - matching SC phi",nbindphimatch2D,dphimatchmin,dphimatchmax);

  //  h_ele_PoPmatchingObject_matched->GetXaxis()-> SetTitle("P/E_{SC}");
  //  h_ele_PoPmatchingObject_matched->GetYaxis()-> SetTitle("Events");
  //  h_ele_PoPmatchingObject_barrel_matched->GetXaxis()-> SetTitle("P/E_{SC}");
  //  h_ele_PoPmatchingObject_barrel_matched->GetYaxis()-> SetTitle("Events");
  //  h_ele_PoPmatchingObject_endcaps_matched->GetXaxis()-> SetTitle("P/E_{SC}");
  //  h_ele_PoPmatchingObject_endcaps_matched->GetYaxis()-> SetTitle("Events");
  //  h_ele_PtoPtmatchingObject_matched->GetXaxis()-> SetTitle("P_{T}/E_{T}^{SC}");
  //  h_ele_PtoPtmatchingObject_matched->GetYaxis()-> SetTitle("Events");
  //  h_ele_PtoPtmatchingObject_barrel_matched->GetXaxis()-> SetTitle("P_{T}/E_{T}^{SC}");
  //  h_ele_PtoPtmatchingObject_barrel_matched->GetYaxis()-> SetTitle("Events");
  //  h_ele_PtoPtmatchingObject_endcaps_matched->GetXaxis()-> SetTitle("P_{T}/E_{T}^{SC}");
  //  h_ele_PtoPtmatchingObject_endcaps_matched->GetYaxis()-> SetTitle("Events");
  //
  //  h_ele_EtaMnEtamatchingObject_matched->GetXaxis()-> SetTitle("#eta_{rec} - #eta_{SC}");
  //  h_ele_EtaMnEtamatchingObject_matched->GetYaxis()-> SetTitle("Events");
  //  h_ele_PhiMnPhimatchingObject_matched->GetXaxis()-> SetTitle("#phi_{rec} - #phi_{SC} (rad)");
  //  h_ele_PhiMnPhimatchingObject_matched->GetYaxis()-> SetTitle("Events");
  //  h_ele_EtaMnEtamatchingObject_matched->GetXaxis()-> SetTitle("#eta_{rec} - #eta_{SC}");
  //  h_ele_EtaMnEtamatchingObject_matched->GetYaxis()-> SetTitle("Events");
  //  h_ele_PhiMnPhimatchingObject_matched->GetXaxis()-> SetTitle("#phi_{rec} - #phi_{SC} (rad)");
  //  h_ele_PhiMnPhimatchingObject_matched->GetYaxis()-> SetTitle("Events");
}

void DQMAnalyzer::endJob() {
  histfile_->cd();
  std::cout << "efficiency calculation " << std::endl;

  // efficiency vs pt
  TH1F *h_ele_ptEff = (TH1F *)h_ele_matchingObjectPt_matched->Clone("h_ele_ptEff");
  h_ele_ptEff->Reset();
  h_ele_ptEff->Divide(h_ele_matchingObjectPt_matched, h_matchingObjectPt, 1, 1, "b");
  h_ele_ptEff->GetXaxis()->SetTitle("p_{T} (GeV/c)");
  h_ele_ptEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs eta
  TH1F *h_ele_etaEff = (TH1F *)h_ele_matchingObjectEta_matched->Clone("h_ele_etaEff");
  h_ele_etaEff->Reset();
  h_ele_etaEff->Divide(h_ele_matchingObjectEta_matched, h_matchingObjectEta, 1, 1, "b");
  h_ele_etaEff->Print();
  h_ele_etaEff->GetXaxis()->SetTitle("#eta");
  h_ele_etaEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs |eta|
  TH1F *h_ele_absetaEff = (TH1F *)h_ele_matchingObjectAbsEta_matched->Clone("h_ele_absetaEff");
  h_ele_absetaEff->Reset();
  h_ele_absetaEff->Divide(h_ele_matchingObjectAbsEta_matched, h_matchingObjectAbsEta, 1, 1, "b");
  h_ele_absetaEff->GetXaxis()->SetTitle("|#eta|");
  h_ele_absetaEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs phi
  TH1F *h_ele_phiEff = (TH1F *)h_ele_matchingObjectPhi_matched->Clone("h_ele_phiEff");
  h_ele_phiEff->Reset();
  h_ele_phiEff->Divide(h_ele_matchingObjectPhi_matched, h_matchingObjectPhi, 1, 1, "b");
  h_ele_phiEff->GetXaxis()->SetTitle("#phi (rad)");
  h_ele_phiEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs z
  TH1F *h_ele_zEff = (TH1F *)h_ele_matchingObjectZ_matched->Clone("h_ele_zEff");
  h_ele_zEff->Reset();
  h_ele_zEff->Divide(h_ele_matchingObjectZ_matched, h_matchingObjectZ, 1, 1, "b");
  h_ele_zEff->Print();
  h_ele_zEff->GetXaxis()->SetTitle("z (cm)");
  h_ele_zEff->GetYaxis()->SetTitle("Efficiency");

  // classes

  // fbrem

  //profiles from 2D histos

  // mc truth

  h_matchingObjectNum->Write();

  // rec event

  histNum_->Write();

  // mc
  h_matchingObjectEta->Write();
  h_matchingObjectAbsEta->Write();
  h_matchingObjectP->Write();
  h_matchingObjectPt->Write();
  h_matchingObjectPhi->Write();
  h_matchingObjectZ->Write();

  // matched electrons
  h_ele_charge->Write();

  //h_ele_vertexP->Write();
  h_ele_vertexPt->Write();
  h_ele_vertexEta->Write();
  h_ele_vertexPhi->Write();
  h_ele_vertexX->Write();
  h_ele_vertexY->Write();
  h_ele_vertexZ->Write();

  h_ele_vertexTIP->Write();

  h_ele_Et->Write();

  h_ele_matchingObjectPt_matched->Write();
  h_ele_matchingObjectAbsEta_matched->Write();
  h_ele_matchingObjectEta_matched->Write();
  h_ele_matchingObjectPhi_matched->Write();
  h_ele_matchingObjectZ_matched->Write();

  //  h_ele_PoPmatchingObject_matched->Write();
  //  h_ele_PtoPtmatchingObject_matched->Write();
  //  h_ele_PoPmatchingObject_barrel_matched ->Write();
  //  h_ele_PoPmatchingObject_endcaps_matched->Write();
  //  h_ele_PtoPtmatchingObject_barrel_matched ->Write();
  //  h_ele_PtoPtmatchingObject_endcaps_matched->Write();
  //  h_ele_EtaMnEtamatchingObject_matched->Write();
  //  h_ele_PhiMnPhimatchingObject_matched ->Write();
  //  h_ele_PhiMnPhimatchingObject2_matched ->Write();

  // matched electron, superclusters
  histSclEn_->Write();
  histSclEt_->Write();
  histSclEta_->Write();
  histSclPhi_->Write();
  histSclSigEtaEta_->Write();

  // matched electron, gsf tracks
  h_ele_ambiguousTracks->Write();
  h_ele_ambiguousTracksVsEta->Write();
  h_ele_ambiguousTracksVsPhi->Write();
  h_ele_ambiguousTracksVsPt->Write();

  h_ele_foundHits->Write();
  h_ele_foundHitsVsEta->Write();
  h_ele_foundHitsVsPhi->Write();
  h_ele_foundHitsVsPt->Write();

  h_ele_lostHits->Write();
  h_ele_lostHitsVsEta->Write();
  h_ele_lostHitsVsPhi->Write();
  h_ele_lostHitsVsPt->Write();

  h_ele_chi2->Write();
  h_ele_chi2VsEta->Write();
  h_ele_chi2VsPhi->Write();
  h_ele_chi2VsPt->Write();

  h_ele_PinMnPout->Write();
  h_ele_PinMnPout_mode->Write();
  h_ele_outerP->Write();
  h_ele_outerP_mode->Write();
  h_ele_outerPt->Write();
  h_ele_outerPt_mode->Write();

  // matched electrons, matching
  h_ele_EoP->Write();
  //  h_ele_EoPout->Write();
  h_ele_EeleOPout->Write();
  h_ele_dEtaSc_propVtx->Write();
  h_ele_dPhiSc_propVtx->Write();
  h_ele_dEtaCl_propOut->Write();
  h_ele_dPhiCl_propOut->Write();
  h_ele_dEtaEleCl_propOut->Write();
  h_ele_dPhiEleCl_propOut->Write();
  h_ele_HoE->Write();

  h_ele_mee_os->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os->GetYaxis()->SetTitle("Events");
  h_ele_mee_os->Write();

  // classes

  // fbrem

  // Eff
  h_ele_etaEff->Write();
  h_ele_zEff->Write();
  h_ele_phiEff->Write();
  h_ele_absetaEff->Write();
  h_ele_ptEff->Write();

  // e/g et pflow electrons
  h_ele_mva->Write();
  h_ele_provenance->Write();

  // isolation
  h_ele_tkSumPt_dr03->GetXaxis()->SetTitle("TkIsoSum, cone 0.3 (GeV/c)");
  h_ele_tkSumPt_dr03->GetYaxis()->SetTitle("Events");
  h_ele_ecalRecHitSumEt_dr03->GetXaxis()->SetTitle("EcalIsoSum, cone 0.3 (GeV)");
  h_ele_ecalRecHitSumEt_dr03->GetYaxis()->SetTitle("Events");
  h_ele_hcalDepth1TowerSumEt_dr03->GetXaxis()->SetTitle("Hcal1IsoSum, cone 0.3 (GeV)");
  h_ele_hcalDepth1TowerSumEt_dr03->GetYaxis()->SetTitle("Events");
  h_ele_hcalDepth2TowerSumEt_dr03->GetXaxis()->SetTitle("Hcal2IsoSum, cone 0.3 (GeV)");
  h_ele_hcalDepth2TowerSumEt_dr03->GetYaxis()->SetTitle("Events");
  h_ele_tkSumPt_dr04->GetXaxis()->SetTitle("TkIsoSum, cone 0.4 (GeV/c)");
  h_ele_tkSumPt_dr04->GetYaxis()->SetTitle("Events");
  h_ele_ecalRecHitSumEt_dr04->GetXaxis()->SetTitle("EcalIsoSum, cone 0.4 (GeV)");
  h_ele_ecalRecHitSumEt_dr04->GetYaxis()->SetTitle("Events");
  h_ele_hcalDepth1TowerSumEt_dr04->GetXaxis()->SetTitle("Hcal1IsoSum, cone 0.4 (GeV)");
  h_ele_hcalDepth1TowerSumEt_dr04->GetYaxis()->SetTitle("Events");
  h_ele_hcalDepth2TowerSumEt_dr04->GetXaxis()->SetTitle("Hcal2IsoSum, cone 0.4 (GeV)");
  h_ele_hcalDepth2TowerSumEt_dr04->GetYaxis()->SetTitle("Events");

  h_ele_tkSumPt_dr03->Write();
  h_ele_ecalRecHitSumEt_dr03->Write();
  h_ele_hcalDepth1TowerSumEt_dr03->Write();
  h_ele_hcalDepth2TowerSumEt_dr03->Write();
  h_ele_tkSumPt_dr04->Write();
  h_ele_ecalRecHitSumEt_dr04->Write();
  h_ele_hcalDepth1TowerSumEt_dr04->Write();
  h_ele_hcalDepth2TowerSumEt_dr04->Write();
}

void DQMAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::LogInfo("DQMAnalyzer::analyze") << "Treating event " << iEvent.id();
  nEvents_++;
  if (!trigger(iEvent))
    return;
  nAfterTrigger_++;
  edm::LogInfo("DQMAnalyzer::analyze") << "Trigger OK";
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  iEvent.getByLabel(electronCollection_, gsfElectrons);
  edm::LogInfo("DQMAnalyzer::analyze") << "Event has " << gsfElectrons.product()->size() << " electrons";
  edm::Handle<reco::SuperClusterCollection> recoClusters;
  iEvent.getByLabel(matchingObjectCollection_, recoClusters);
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(beamSpot_, recoBeamSpotHandle);
  const reco::BeamSpot bs = *recoBeamSpotHandle;
  histNum_->Fill((*gsfElectrons).size());

  // selected rec electrons
  reco::GsfElectronCollection::const_iterator gsfIter;
  for (gsfIter = gsfElectrons->begin(); gsfIter != gsfElectrons->end(); gsfIter++) {
    // vertex TIP
    double vertexTIP = (gsfIter->vertex().x() - bs.position().x()) * (gsfIter->vertex().x() - bs.position().x()) +
                       (gsfIter->vertex().y() - bs.position().y()) * (gsfIter->vertex().y() - bs.position().y());
    vertexTIP = sqrt(vertexTIP);

    // select electrons
    if (!selected(gsfIter, vertexTIP))
      continue;

    // electron related distributions
    h_ele_charge->Fill(gsfIter->charge());
    //h_ele_vertexP->Fill( gsfIter->p() );
    h_ele_vertexPt->Fill(gsfIter->pt());
    h_ele_Et->Fill(gsfIter->superCluster()->energy() / cosh(gsfIter->superCluster()->eta()));
    h_ele_vertexEta->Fill(gsfIter->eta());
    h_ele_vertexPhi->Fill(gsfIter->phi());
    h_ele_vertexX->Fill(gsfIter->vertex().x());
    h_ele_vertexY->Fill(gsfIter->vertex().y());
    h_ele_vertexZ->Fill(gsfIter->vertex().z());
    h_ele_vertexTIP->Fill(vertexTIP);

    // supercluster related distributions
    reco::SuperClusterRef sclRef = gsfIter->superCluster();
    // ALREADY DONE IN GSF ELECTRON CORE
    //    if (!gsfIter->ecalDrivenSeed()&&gsfIter->trackerDrivenSeed())
    //      sclRef = gsfIter->parentSuperCluster() ;
    histSclEn_->Fill(sclRef->energy());
    double R = TMath::Sqrt(sclRef->x() * sclRef->x() + sclRef->y() * sclRef->y() + sclRef->z() * sclRef->z());
    double Rt = TMath::Sqrt(sclRef->x() * sclRef->x() + sclRef->y() * sclRef->y());
    histSclEt_->Fill(sclRef->energy() * (Rt / R));
    histSclEta_->Fill(sclRef->eta());
    histSclPhi_->Fill(sclRef->phi());
    histSclSigEtaEta_->Fill(gsfIter->scSigmaEtaEta());

    // track related distributions
    h_ele_ambiguousTracks->Fill(gsfIter->ambiguousGsfTracksSize());
    h_ele_ambiguousTracksVsEta->Fill(gsfIter->eta(), gsfIter->ambiguousGsfTracksSize());
    h_ele_ambiguousTracksVsPhi->Fill(gsfIter->phi(), gsfIter->ambiguousGsfTracksSize());
    h_ele_ambiguousTracksVsPt->Fill(gsfIter->pt(), gsfIter->ambiguousGsfTracksSize());
    if (!readAOD_) {  // track extra does not exist in AOD
      h_ele_foundHits->Fill(gsfIter->gsfTrack()->numberOfValidHits());
      h_ele_foundHitsVsEta->Fill(gsfIter->eta(), gsfIter->gsfTrack()->numberOfValidHits());
      h_ele_foundHitsVsPhi->Fill(gsfIter->phi(), gsfIter->gsfTrack()->numberOfValidHits());
      h_ele_foundHitsVsPt->Fill(gsfIter->pt(), gsfIter->gsfTrack()->numberOfValidHits());
      h_ele_lostHits->Fill(gsfIter->gsfTrack()->numberOfLostHits());
      h_ele_lostHitsVsEta->Fill(gsfIter->eta(), gsfIter->gsfTrack()->numberOfLostHits());
      h_ele_lostHitsVsPhi->Fill(gsfIter->phi(), gsfIter->gsfTrack()->numberOfLostHits());
      h_ele_lostHitsVsPt->Fill(gsfIter->pt(), gsfIter->gsfTrack()->numberOfLostHits());
      h_ele_chi2->Fill(gsfIter->gsfTrack()->normalizedChi2());
      h_ele_chi2VsEta->Fill(gsfIter->eta(), gsfIter->gsfTrack()->normalizedChi2());
      h_ele_chi2VsPhi->Fill(gsfIter->phi(), gsfIter->gsfTrack()->normalizedChi2());
      h_ele_chi2VsPt->Fill(gsfIter->pt(), gsfIter->gsfTrack()->normalizedChi2());
    }

    // from gsf track interface, hence using mean
    if (!readAOD_) {  // track extra does not exist in AOD
      h_ele_PinMnPout->Fill(gsfIter->gsfTrack()->innerMomentum().R() - gsfIter->gsfTrack()->outerMomentum().R());
      h_ele_outerP->Fill(gsfIter->gsfTrack()->outerMomentum().R());
      h_ele_outerPt->Fill(gsfIter->gsfTrack()->outerMomentum().Rho());
    }

    // from electron interface, hence using mode
    h_ele_PinMnPout_mode->Fill(gsfIter->trackMomentumAtVtx().R() - gsfIter->trackMomentumOut().R());
    h_ele_outerP_mode->Fill(gsfIter->trackMomentumOut().R());
    h_ele_outerPt_mode->Fill(gsfIter->trackMomentumOut().Rho());

    /*
    if (!readAOD_) { // track extra does not exist in AOD
            edm::RefToBase<TrajectorySeed> seed = gsfIter->gsfTrack()->extra()->seedRef();
      ElectronSeedRef elseed=seed.castTo<ElectronSeedRef>();
      h_ele_seed_dphi2_-> Fill(elseed->dPhiNeg(1));
            h_ele_seed_dphi2VsEta_-> Fill(gsfIter->eta(), elseed->dPhiNeg(1));
            h_ele_seed_dphi2VsPt_-> Fill(gsfIter->pt(), elseed->dPhiNeg(1)) ;
            h_ele_seed_drz2_-> Fill(elseed->dRZNeg(1));
            h_ele_seed_drz2VsEta_-> Fill(gsfIter->eta(), elseed->dRZNeg(1));
            h_ele_seed_drz2VsPt_-> Fill(gsfIter->pt(), elseed->dRZNeg(1));
            h_ele_seed_subdet2_-> Fill(elseed->subDet(1));
          }
    */

    // match distributions
    h_ele_EoP->Fill(gsfIter->eSuperClusterOverP());
    h_ele_EeleOPout->Fill(gsfIter->eEleClusterOverPout());
    h_ele_dEtaSc_propVtx->Fill(gsfIter->deltaEtaSuperClusterTrackAtVtx());
    h_ele_dPhiSc_propVtx->Fill(gsfIter->deltaPhiSuperClusterTrackAtVtx());
    h_ele_dEtaCl_propOut->Fill(gsfIter->deltaEtaSeedClusterTrackAtCalo());
    h_ele_dPhiCl_propOut->Fill(gsfIter->deltaPhiSeedClusterTrackAtCalo());
    h_ele_dEtaEleCl_propOut->Fill(gsfIter->deltaEtaEleClusterTrackAtCalo());
    h_ele_dPhiEleCl_propOut->Fill(gsfIter->deltaPhiEleClusterTrackAtCalo());
    h_ele_HoE->Fill(gsfIter->hadronicOverEm());

    //classes

    //fbrem

    h_ele_mva->Fill(gsfIter->mva_e_pi());
    if (gsfIter->ecalDrivenSeed())
      h_ele_provenance->Fill(1.);
    if (gsfIter->trackerDrivenSeed())
      h_ele_provenance->Fill(-1.);
    if (gsfIter->trackerDrivenSeed() || gsfIter->ecalDrivenSeed())
      h_ele_provenance->Fill(0.);
    if (gsfIter->trackerDrivenSeed() && !gsfIter->ecalDrivenSeed())
      h_ele_provenance->Fill(-2.);
    if (!gsfIter->trackerDrivenSeed() && gsfIter->ecalDrivenSeed())
      h_ele_provenance->Fill(2.);

    h_ele_tkSumPt_dr03->Fill(gsfIter->dr03TkSumPt());
    h_ele_ecalRecHitSumEt_dr03->Fill(gsfIter->dr03EcalRecHitSumEt());
    h_ele_hcalDepth1TowerSumEt_dr03->Fill(gsfIter->dr03HcalDepth1TowerSumEt());
    h_ele_hcalDepth2TowerSumEt_dr03->Fill(gsfIter->dr03HcalDepth2TowerSumEt());
    h_ele_tkSumPt_dr04->Fill(gsfIter->dr04TkSumPt());
    h_ele_ecalRecHitSumEt_dr04->Fill(gsfIter->dr04EcalRecHitSumEt());
    h_ele_hcalDepth1TowerSumEt_dr04->Fill(gsfIter->dr04HcalDepth1TowerSumEt());
    h_ele_hcalDepth2TowerSumEt_dr04->Fill(gsfIter->dr04HcalDepth2TowerSumEt());
  }

  // association matching object-reco electrons
  int matchingObjectNum = 0;
  reco::SuperClusterCollection::const_iterator moIter;
  for (moIter = recoClusters->begin(); moIter != recoClusters->end(); moIter++) {
    // number of matching objects
    matchingObjectNum++;

    if (moIter->energy() / cosh(moIter->eta()) > maxPtMatchingObject_ ||
        fabs(moIter->eta()) > maxAbsEtaMatchingObject_) {
      continue;
    }

    // suppress the endcaps
    //if (fabs(moIter->eta()) > 1.5) continue;
    // select central z
    //if ( fabs((*mcIter)->production_vertex()->position().z())>50.) continue;

    h_matchingObjectEta->Fill(moIter->eta());
    h_matchingObjectAbsEta->Fill(fabs(moIter->eta()));
    h_matchingObjectP->Fill(moIter->energy());
    h_matchingObjectPt->Fill(moIter->energy() / cosh(moIter->eta()));
    h_matchingObjectPhi->Fill(moIter->phi());
    h_matchingObjectZ->Fill(moIter->z());

    // find best matched electron
    bool okGsfFound = false;
    double gsfOkRatio = 999999.;
    reco::GsfElectron bestGsfElectron;
    reco::GsfElectronCollection::const_iterator gsfIter;
    for (gsfIter = gsfElectrons->begin(); gsfIter != gsfElectrons->end(); gsfIter++) {
      double vertexTIP = (gsfIter->vertex().x() - bs.position().x()) * (gsfIter->vertex().x() - bs.position().x()) +
                         (gsfIter->vertex().y() - bs.position().y()) * (gsfIter->vertex().y() - bs.position().y());
      vertexTIP = sqrt(vertexTIP);

      // select electrons
      if (!selected(gsfIter, vertexTIP))
        continue;

      if (Selection_ >= 4) {
        reco::GsfElectronCollection::const_iterator gsfIter2;
        for (gsfIter2 = gsfIter + 1; gsfIter2 != gsfElectrons->end(); gsfIter2++) {
          math::XYZTLorentzVector p12 = (*gsfIter).p4() + (*gsfIter2).p4();
          float mee2 = p12.Dot(p12);
          bool opsign = (gsfIter->charge() * gsfIter2->charge() < 0.);
          float invMass = sqrt(mee2);

          if (TPchecksign_ && !opsign)
            break;

          // conditions Tag
          if (TAGcheckclass_ && (gsfIter->classification() == GsfElectron::SHOWERING || gsfIter->isGap()))
            break;

          // conditions Probe
          if (PROBEetcut_ && (gsfIter2->superCluster()->energy() / cosh(gsfIter2->superCluster()->eta()) < minEt_))
            continue;
          if (PROBEcheckclass_ && (gsfIter2->classification() == GsfElectron::SHOWERING || gsfIter2->isGap()))
            continue;

          if (invMass < massLow_ || invMass > massHigh_)
            continue;

          h_ele_mee_os->Fill(invMass);
          bestGsfElectron = *gsfIter2;
          okGsfFound = true;
        }
      } else {
        reco::GsfElectronCollection::const_iterator gsfIter2;
        for (gsfIter2 = gsfIter + 1; gsfIter2 != gsfElectrons->end(); gsfIter2++) {
          math::XYZTLorentzVector p12 = (*gsfIter).p4() + (*gsfIter2).p4();
          float mee2 = p12.Dot(p12);
          //bool opsign = (gsfIter->charge()*gsfIter2->charge()<0.) ;
          float invMass = sqrt(mee2);
          h_ele_mee_os->Fill(invMass);
        }

        // matching with a cone in eta phi
        if (matchingCondition_ == "Cone") {
          double dphi = gsfIter->phi() - moIter->phi();
          if (fabs(dphi) > CLHEP::pi) {
            dphi = dphi < 0 ? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
          }
          double deltaR = sqrt(std::pow((moIter->eta() - gsfIter->eta()), 2) + std::pow(dphi, 2));
          if (deltaR < deltaR_) {
            //if ( (genPc->pdg_id() == 11) && (gsfIter->charge() < 0.) || (genPc->pdg_id() == -11) &&
            //(gsfIter->charge() > 0.) ){
            double tmpGsfRatio = gsfIter->p() / moIter->energy();
            if (fabs(tmpGsfRatio - 1) < fabs(gsfOkRatio - 1) && Selection_ != 4) {
              gsfOkRatio = tmpGsfRatio;
              bestGsfElectron = *gsfIter;
              okGsfFound = true;
            }
            //}
          }
        }
      }
    }  // loop over rec ele to look for the best one

    // analysis when the matching object is matched by a rec electron
    if (okGsfFound) {
      // generated distributions for matched electrons
      h_ele_matchingObjectPt_matched->Fill(moIter->energy() / cosh(moIter->eta()));
      h_ele_matchingObjectPhi_matched->Fill(moIter->phi());
      h_ele_matchingObjectAbsEta_matched->Fill(fabs(moIter->eta()));
      h_ele_matchingObjectEta_matched->Fill(moIter->eta());
      h_ele_matchingObjectZ_matched->Fill(moIter->z());

      // OBSOLETE
      //      // comparison electron vs matching object
      //      h_ele_EtaMnEtamatchingObject_matched->Fill( bestGsfElectron.eta()-moIter->eta());
      //
      //      h_ele_PhiMnPhimatchingObject_matched->Fill( bestGsfElectron.phi()-moIter->phi());
      //      h_ele_PhiMnPhimatchingObject2_matched->Fill( bestGsfElectron.phi()-moIter->phi());
      //
      //      h_ele_PoPmatchingObject_matched->Fill( bestGsfElectron.p()/moIter->energy());
      //      h_ele_PtoPtmatchingObject_matched->Fill( bestGsfElectron.pt()/moIter->energy()/cosh(moIter->eta()));
      //
      //      if (bestGsfElectron.isEB()) h_ele_PoPmatchingObject_barrel_matched->Fill( bestGsfElectron.p()/moIter->energy());
      //      if (bestGsfElectron.isEE()) h_ele_PoPmatchingObject_endcaps_matched->Fill( bestGsfElectron.p()/moIter->energy());
      //      if (bestGsfElectron.isEB()) h_ele_PtoPtmatchingObject_barrel_matched->Fill( bestGsfElectron.pt()/moIter->energy()/cosh(moIter->eta()));
      //      if (bestGsfElectron.isEE()) h_ele_PtoPtmatchingObject_endcaps_matched->Fill( bestGsfElectron.pt()/moIter->energy()/cosh(moIter->eta()));

      reco::SuperClusterRef sclRef = bestGsfElectron.superCluster();

      // add here distributions for matched electrons as for all electrons
      //..
    }  // gsf electron found

  }  // loop overmatching object

  h_matchingObjectNum->Fill(matchingObjectNum);
}

bool DQMAnalyzer::trigger(const edm::Event &e) {
  // retreive TriggerResults from the event
  edm::Handle<edm::TriggerResults> triggerResults;
  e.getByLabel(triggerResults_, triggerResults);

  bool accept = false;

  if (triggerResults.isValid()) {
    //std::cout << "TriggerResults found, number of HLT paths: " << triggerResults->size() << std::endl;

    // get trigger names
    const edm::TriggerNames &triggerNames = e.triggerNames(*triggerResults);
    if (nEvents_ == 1) {
      for (unsigned int i = 0; i < triggerNames.size(); i++) {
        //	      std::cout << "trigger path= " << triggerNames.triggerName(i) << std::endl;
      }
    }

    unsigned int n = HLTPathsByName_.size();
    for (unsigned int i = 0; i != n; i++) {
      HLTPathsByIndex_[i] = triggerNames.triggerIndex(HLTPathsByName_[i]);
    }

    // empty input vectors (n==0) means any trigger paths
    if (n == 0) {
      n = triggerResults->size();
      HLTPathsByName_.resize(n);
      HLTPathsByIndex_.resize(n);
      for (unsigned int i = 0; i != n; i++) {
        HLTPathsByName_[i] = triggerNames.triggerName(i);
        HLTPathsByIndex_[i] = i;
      }
    }

    //    if (nEvents_==1)
    //     {
    //      if (n>0)
    //       {
    //	      std::cout << "HLT trigger paths requested: index, name and valididty:" << std::endl;
    //	      for (unsigned int i=0; i!=n; i++)
    //	       {
    //	        bool validity = HLTPathsByIndex_[i]<triggerResults->size();
    //	        std::cout
    //	          << " " << HLTPathsByIndex_[i]
    //	          << " " << HLTPathsByName_[i]
    //	          << " " << validity << std::endl;
    //	       }
    //       }
    //     }

    // count number of requested HLT paths which have fired
    unsigned int fired = 0;
    for (unsigned int i = 0; i != n; i++) {
      if (HLTPathsByIndex_[i] < triggerResults->size()) {
        if (triggerResults->accept(HLTPathsByIndex_[i])) {
          fired++;
          //std::cout << "Fired HLT path= " << HLTPathsByName_[i] << std::endl ;
          accept = true;
        }
      }
    }
  }

  return accept;
}

bool DQMAnalyzer::selected(const reco::GsfElectronCollection::const_iterator &gsfIter, double vertexTIP) {
  if ((Selection_ > 0) && generalCut(gsfIter))
    return false;
  if ((Selection_ >= 1) && etCut(gsfIter))
    return false;
  if ((Selection_ >= 2) && isolationCut(gsfIter, vertexTIP))
    return false;
  if ((Selection_ >= 3) && idCut(gsfIter))
    return false;
  return true;
}

bool DQMAnalyzer::generalCut(const reco::GsfElectronCollection::const_iterator &gsfIter) {
  if (fabs(gsfIter->eta()) > maxAbsEta_)
    return true;
  if (gsfIter->pt() < minPt_)
    return true;

  if (gsfIter->isEB() && isEE_)
    return true;
  if (gsfIter->isEE() && isEB_)
    return true;
  if (gsfIter->isEBEEGap() && isNotEBEEGap_)
    return true;

  if (gsfIter->ecalDrivenSeed() && isTrackerDriven_)
    return true;
  if (gsfIter->trackerDrivenSeed() && isEcalDriven_)
    return true;

  return false;
}

bool DQMAnalyzer::etCut(const reco::GsfElectronCollection::const_iterator &gsfIter) {
  if (gsfIter->superCluster()->energy() / cosh(gsfIter->superCluster()->eta()) < minEt_)
    return true;

  return false;
}

bool DQMAnalyzer::isolationCut(const reco::GsfElectronCollection::const_iterator &gsfIter, double vertexTIP) {
  if (gsfIter->isEB() && vertexTIP > tipMaxBarrel_)
    return true;
  if (gsfIter->isEE() && vertexTIP > tipMaxEndcaps_)
    return true;

  if (gsfIter->dr03TkSumPt() > tkIso03Max_)
    return true;
  if (gsfIter->isEB() && gsfIter->dr03HcalDepth1TowerSumEt() > hcalIso03Depth1MaxBarrel_)
    return true;
  if (gsfIter->isEE() && gsfIter->dr03HcalDepth1TowerSumEt() > hcalIso03Depth1MaxEndcaps_)
    return true;
  if (gsfIter->isEE() && gsfIter->dr03HcalDepth2TowerSumEt() > hcalIso03Depth2MaxEndcaps_)
    return true;
  if (gsfIter->isEB() && gsfIter->dr03EcalRecHitSumEt() > ecalIso03MaxBarrel_)
    return true;
  if (gsfIter->isEE() && gsfIter->dr03EcalRecHitSumEt() > ecalIso03MaxEndcaps_)
    return true;

  return false;
}

bool DQMAnalyzer::idCut(const reco::GsfElectronCollection::const_iterator &gsfIter) {
  if (gsfIter->isEB() && gsfIter->eSuperClusterOverP() < eOverPMinBarrel_)
    return true;
  if (gsfIter->isEB() && gsfIter->eSuperClusterOverP() > eOverPMaxBarrel_)
    return true;
  if (gsfIter->isEE() && gsfIter->eSuperClusterOverP() < eOverPMinEndcaps_)
    return true;
  if (gsfIter->isEE() && gsfIter->eSuperClusterOverP() > eOverPMaxEndcaps_)
    return true;
  if (gsfIter->isEB() && fabs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) < dEtaMinBarrel_)
    return true;
  if (gsfIter->isEB() && fabs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) > dEtaMaxBarrel_)
    return true;
  if (gsfIter->isEE() && fabs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) < dEtaMinEndcaps_)
    return true;
  if (gsfIter->isEE() && fabs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) > dEtaMaxEndcaps_)
    return true;
  if (gsfIter->isEB() && fabs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) < dPhiMinBarrel_)
    return true;
  if (gsfIter->isEB() && fabs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) > dPhiMaxBarrel_)
    return true;
  if (gsfIter->isEE() && fabs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) < dPhiMinEndcaps_)
    return true;
  if (gsfIter->isEE() && fabs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) > dPhiMaxEndcaps_)
    return true;
  if (gsfIter->isEB() && gsfIter->scSigmaIEtaIEta() < sigIetaIetaMinBarrel_)
    return true;
  if (gsfIter->isEB() && gsfIter->scSigmaIEtaIEta() > sigIetaIetaMaxBarrel_)
    return true;
  if (gsfIter->isEE() && gsfIter->scSigmaIEtaIEta() < sigIetaIetaMinEndcaps_)
    return true;
  if (gsfIter->isEE() && gsfIter->scSigmaIEtaIEta() > sigIetaIetaMaxEndcaps_)
    return true;
  if (gsfIter->isEB() && gsfIter->hadronicOverEm() > hadronicOverEmMaxBarrel_)
    return true;
  if (gsfIter->isEE() && gsfIter->hadronicOverEm() > hadronicOverEmMaxEndcaps_)
    return true;

  return false;
}
