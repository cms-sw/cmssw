
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
#include "RecoEgamma/Examples/plugins/GsfElectronDataAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
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

using namespace reco;

GsfElectronDataAnalyzer::GsfElectronDataAnalyzer(const edm::ParameterSet &conf)
    : beamSpot_(conf.getParameter<edm::InputTag>("beamSpot")) {
  outputFile_ = conf.getParameter<std::string>("outputFile");
  histfile_ = new TFile(outputFile_.c_str(), "RECREATE");
  electronCollection_ = conf.getParameter<edm::InputTag>("electronCollection");
  readAOD_ = conf.getParameter<bool>("readAOD");

  matchingObjectCollection_ = conf.getParameter<edm::InputTag>("matchingObjectCollection");
  matchingCondition_ = conf.getParameter<std::string>("matchingCondition");
  // currently only one possible matching conditions
  assert(matchingCondition_ == "Cone");
  maxPtMatchingObject_ = conf.getParameter<double>("MaxPtMatchingObject");
  maxAbsEtaMatchingObject_ = conf.getParameter<double>("MaxAbsEtaMatchingObject");
  deltaR_ = conf.getParameter<double>("DeltaR");

  triggerResults_ = conf.getParameter<edm::InputTag>("triggerResults");
  HLTPathsByName_ = conf.getParameter<std::vector<std::string> >("hltPaths");
  HLTPathsByIndex_.resize(HLTPathsByName_.size());

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

GsfElectronDataAnalyzer::~GsfElectronDataAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  histfile_->Write();
  histfile_->Close();
}

void GsfElectronDataAnalyzer::beginJob() {
  histfile_->cd();

  nEvents_ = 0;
  nAfterTrigger_ = 0;

  // matching object
  std::string::size_type locSC = matchingObjectCollection_.label().find("SuperCluster", 0);
  std::string type_;
  if (locSC != std::string::npos) {
    std::cout << "Matching objects are SuperClusters " << std::endl;
    type_ = "SC";
  } else {
    std::cout << "Didn't recognize input matching objects!! " << std::endl;
  }

  std::string htitle, hlabel;
  hlabel = "h_" + type_ + "Num";
  htitle = "# " + type_ + "s";
  h_matchingObjectNum = new TH1F(hlabel.c_str(), htitle.c_str(), nbinfhits, 0., fhitsmax);

  // rec event

  histNum_ = new TH1F("h_recEleNum", "# rec electrons", 20, 0., 20.);

  // matching object distributions
  hlabel = "h_" + type_ + "_eta";
  htitle = type_ + " #eta";
  h_matchingObjectEta = new TH1F(hlabel.c_str(), htitle.c_str(), nbineta, etamin, etamax);
  hlabel = "h_" + type_ + "_abseta";
  htitle = type_ + " |#eta|";
  h_matchingObjectAbsEta = new TH1F(hlabel.c_str(), htitle.c_str(), nbineta / 2, 0., etamax);
  hlabel = "h_" + type_ + "_P";
  htitle = type_ + " p";
  h_matchingObjectP = new TH1F(hlabel.c_str(), htitle.c_str(), nbinp, 0., pmax);
  hlabel = "h_" + type_ + "_Pt";
  htitle = type_ + " pt";
  h_matchingObjectPt = new TH1F(hlabel.c_str(), htitle.c_str(), nbinpteff, 5., ptmax);
  hlabel = "h_" + type_ + "_phi";
  htitle = type_ + " phi";
  h_matchingObjectPhi = new TH1F(hlabel.c_str(), htitle.c_str(), nbinphi, phimin, phimax);
  hlabel = "h_" + type_ + "_z";
  htitle = type_ + " z";
  h_matchingObjectZ = new TH1F(hlabel.c_str(), htitle.c_str(), nbinxyz, -25, 25);

  // mee
  h_ele_mee_all =
      new TH1F("h_ele_mee_all", "ele pairs invariant mass, all charge combinations", nbinmee, meemin, meemax);
  h_ele_mee_os = new TH1F("h_ele_mee_os", "ele pairs invariant mass, opposite sign", nbinmee, meemin, meemax);
  h_ele_mee_os_ebeb =
      new TH1F("h_ele_mee_os_ebeb", "ele pairs invariant mass, opp. sign, EB-EB", nbinmee, meemin, meemax);
  h_ele_mee_os_ebeb->Sumw2();
  h_ele_mee_os_ebee =
      new TH1F("h_ele_mee_os_ebee", "ele pairs invariant mass, opp. sign, EB-EE", nbinmee, meemin, meemax);
  h_ele_mee_os_ebee->Sumw2();
  h_ele_mee_os_eeee =
      new TH1F("h_ele_mee_os_eeee", "ele pairs invariant mass, opp. sign, EE-EE", nbinmee, meemin, meemax);
  h_ele_mee_os_eeee->Sumw2();
  h_ele_mee_os_gg =
      new TH1F("h_ele_mee_os_gg", "ele pairs invariant mass, opp. sign, good-good", nbinmee, meemin, meemax);
  h_ele_mee_os_gg->Sumw2();
  h_ele_mee_os_gb =
      new TH1F("h_ele_mee_os_gb", "ele pairs invariant mass, opp. sign, good-bad", nbinmee, meemin, meemax);
  h_ele_mee_os_gb->Sumw2();
  h_ele_mee_os_bb =
      new TH1F("h_ele_mee_os_bb", "ele pairs invariant mass, opp. sign, bad-bad", nbinmee, meemin, meemax);
  h_ele_mee_os_bb->Sumw2();

  // duplicates
  h_ele_E2mnE1vsMee_all = new TH2F("h_ele_E2mnE1vsMee_all",
                                   "E2 - E1 vs ele pairs invariant mass, all electrons",
                                   nbinmee,
                                   meemin,
                                   meemax,
                                   100,
                                   -50.,
                                   50.);
  h_ele_E2mnE1vsMee_egeg_all = new TH2F("h_ele_E2mnE1vsMee_egeg_all",
                                        "E2 - E1 vs ele pairs invariant mass, ecal driven pairs, all electrons",
                                        nbinmee,
                                        meemin,
                                        meemax,
                                        100,
                                        -50.,
                                        50.);

  // recoed and matched electrons
  h_ele_charge = new TH1F("h_ele_charge", "ele charge", 5, -2., 2.);
  h_ele_chargeVsEta = new TH2F("h_ele_chargeVsEta", "ele charge vs eta", nbineta2D, etamin, etamax, 5, -2., 2.);
  h_ele_chargeVsPhi = new TH2F("h_ele_chargeVsPhi", "ele charge vs phi", nbinphi2D, phimin, phimax, 5, -2., 2.);
  h_ele_chargeVsPt = new TH2F("h_ele_chargeVsPt", "ele charge vs pt", nbinpt, 0., 100., 5, -2., 2.);
  h_ele_vertexP = new TH1F("h_ele_vertexP", "ele momentum", nbinp, 0., pmax);
  h_ele_vertexPt = new TH1F("h_ele_vertexPt", "ele transverse momentum", nbinpt, 0., ptmax);
  h_ele_Et = new TH1F("h_ele_Et", "ele SC transverse energy", nbinpt, 0., ptmax);
  h_ele_vertexPtVsEta =
      new TH2F("h_ele_vertexPtVsEta", "ele transverse momentum vs eta", nbineta2D, etamin, etamax, nbinpt2D, 0., ptmax);
  h_ele_vertexPtVsPhi =
      new TH2F("h_ele_vertexPtVsPhi", "ele transverse momentum vs phi", nbinphi2D, phimin, phimax, nbinpt2D, 0., ptmax);
  h_ele_matchingObjectPt_matched =
      new TH1F("h_ele_matchingObjectPt_matched", "Efficiency vs matching SC E_{T}", nbinpteff, 5., ptmax);
  h_ele_matchingObjectPt_matched->Sumw2();
  h_ele_vertexEta = new TH1F("h_ele_vertexEta", "ele momentum eta", nbineta, etamin, etamax);
  h_ele_vertexEtaVsPhi =
      new TH2F("h_ele_vertexEtaVsPhi", "ele momentum eta vs phi", nbineta2D, etamin, etamax, nbinphi2D, phimin, phimax);
  h_ele_matchingObjectAbsEta_matched =
      new TH1F("h_ele_matchingObjectAbsEta_matched", "Efficiency vs matching SC |#eta|", nbineta / 2, 0., 2.5);
  h_ele_matchingObjectAbsEta_matched->Sumw2();
  h_ele_matchingObjectEta_matched =
      new TH1F("h_ele_matchingObjectEta_matched", "Efficiency vs matching SC #eta", nbineta, etamin, etamax);
  h_ele_matchingObjectEta_matched->Sumw2();
  h_ele_matchingObjectPhi_matched =
      new TH1F("h_ele_matchingObjectPhi_matched", "Efficiency vs matching SC phi", nbinphi, phimin, phimax);
  h_ele_matchingObjectPhi_matched->Sumw2();
  h_ele_vertexPhi = new TH1F("h_ele_vertexPhi", "ele  momentum #phi", nbinphi, phimin, phimax);
  h_ele_vertexX = new TH1F("h_ele_vertexX", "ele vertex x", nbinxyz, -0.1, 0.1);
  h_ele_vertexY = new TH1F("h_ele_vertexY", "ele vertex y", nbinxyz, -0.1, 0.1);
  h_ele_vertexZ = new TH1F("h_ele_vertexZ", "ele vertex z", nbinxyz, -25, 25);
  h_ele_matchingObjectZ_matched =
      new TH1F("h_ele_matchingObjectZ_matched", "Efficiency vs matching SC z", nbinxyz, -25, 25);
  h_ele_matchingObjectZ_matched->Sumw2();
  h_ele_vertexTIP = new TH1F("h_ele_vertexTIP", "ele transverse impact parameter (wrt bs)", 90, 0., 0.15);
  h_ele_vertexTIPVsEta = new TH2F("h_ele_vertexTIPVsEta",
                                  "ele transverse impact parameter (wrt bs) vs eta",
                                  nbineta2D,
                                  etamin,
                                  etamax,
                                  45,
                                  0.,
                                  0.15);
  h_ele_vertexTIPVsPhi = new TH2F("h_ele_vertexTIPVsPhi",
                                  "ele transverse impact parameter (wrt bs) vs phi",
                                  nbinphi2D,
                                  phimin,
                                  phimax,
                                  45,
                                  0.,
                                  0.15);
  h_ele_vertexTIPVsPt = new TH2F("h_ele_vertexTIPVsPt",
                                 "ele transverse impact parameter (wrt bs) vs transverse momentum",
                                 nbinpt2D,
                                 0.,
                                 ptmax,
                                 45,
                                 0.,
                                 0.15);
  h_ele_PoPmatchingObject_matched = new TH1F(
      "h_ele_PoPmatchingObject_matched", "ele momentum / matching SC energy", nbinpoptrue, poptruemin, poptruemax);
  h_ele_PtoPtmatchingObject_matched = new TH1F("h_ele_PtoPtmatchingObject_matched",
                                               "ele trans momentum / matching SC trans energy",
                                               nbinpoptrue,
                                               poptruemin,
                                               poptruemax);
  h_ele_PoPmatchingObjectVsEta_matched = new TH2F("h_ele_PoPmatchingObjectVsEta_matched",
                                                  "ele momentum / matching SC energy vs eta",
                                                  nbineta2D,
                                                  etamin,
                                                  etamax,
                                                  50,
                                                  poptruemin,
                                                  poptruemax);
  h_ele_PoPmatchingObjectVsPhi_matched = new TH2F("h_ele_PoPmatchingObjectVsPhi_matched",
                                                  "ele momentum / matching SC energy vs phi",
                                                  nbinphi2D,
                                                  phimin,
                                                  phimax,
                                                  50,
                                                  poptruemin,
                                                  poptruemax);
  h_ele_PoPmatchingObjectVsPt_matched = new TH2F("h_ele_PoPmatchingObjectVsPt_matched",
                                                 "ele momentum / matching SC energy vs eta",
                                                 nbinpt2D,
                                                 0.,
                                                 ptmax,
                                                 50,
                                                 poptruemin,
                                                 poptruemax);
  h_ele_PoPmatchingObject_barrel_matched = new TH1F("h_ele_PoPmatchingObject_barrel_matched",
                                                    "ele momentum / matching SC energy, barrel",
                                                    nbinpoptrue,
                                                    poptruemin,
                                                    poptruemax);
  h_ele_PoPmatchingObject_endcaps_matched = new TH1F("h_ele_PoPmatchingObject_endcaps_matched",
                                                     "ele momentum / matching SC energy, endcaps",
                                                     nbinpoptrue,
                                                     poptruemin,
                                                     poptruemax);
  h_ele_PtoPtmatchingObject_barrel_matched = new TH1F("h_ele_PtoPmatchingObject_barrel_matched",
                                                      "ele trans momentum / matching SC trans energy, barrel",
                                                      nbinpoptrue,
                                                      poptruemin,
                                                      poptruemax);
  h_ele_PtoPtmatchingObject_endcaps_matched = new TH1F("h_ele_PtoPmatchingObject_endcaps_matched",
                                                       "ele trans momentum / matching SC trans energy, endcaps",
                                                       nbinpoptrue,
                                                       poptruemin,
                                                       poptruemax);
  h_ele_EtaMnEtamatchingObject_matched = new TH1F(
      "h_ele_EtaMnEtamatchingObject_matched", "ele momentum eta - matching SC eta", nbindeta, detamin, detamax);
  h_ele_EtaMnEtamatchingObjectVsEta_matched = new TH2F("h_ele_EtaMnEtamatchingObjectVsEta_matched",
                                                       "ele momentum eta - matching SC eta vs eta",
                                                       nbineta2D,
                                                       etamin,
                                                       etamax,
                                                       nbindeta / 2,
                                                       detamin,
                                                       detamax);
  h_ele_EtaMnEtamatchingObjectVsPhi_matched = new TH2F("h_ele_EtaMnEtamatchingObjectVsPhi_matched",
                                                       "ele momentum eta - matching SC eta vs phi",
                                                       nbinphi2D,
                                                       phimin,
                                                       phimax,
                                                       nbindeta / 2,
                                                       detamin,
                                                       detamax);
  h_ele_EtaMnEtamatchingObjectVsPt_matched = new TH2F("h_ele_EtaMnEtamatchingObjectVsPt_matched",
                                                      "ele momentum eta - matching SC eta vs pt",
                                                      nbinpt,
                                                      0.,
                                                      ptmax,
                                                      nbindeta / 2,
                                                      detamin,
                                                      detamax);
  h_ele_PhiMnPhimatchingObject_matched = new TH1F(
      "h_ele_PhiMnPhimatchingObject_matched", "ele momentum phi - matching SC phi", nbindphi, dphimin, dphimax);
  h_ele_PhiMnPhimatchingObject2_matched = new TH1F("h_ele_PhiMnPhimatchingObject2_matched",
                                                   "ele momentum phi - matching SC phi",
                                                   nbindphimatch2D,
                                                   dphimatchmin,
                                                   dphimatchmax);
  h_ele_PhiMnPhimatchingObjectVsEta_matched = new TH2F("h_ele_PhiMnPhimatchingObjectVsEta_matched",
                                                       "ele momentum phi - matching SC phi vs eta",
                                                       nbineta2D,
                                                       etamin,
                                                       etamax,
                                                       nbindphi / 2,
                                                       dphimin,
                                                       dphimax);
  h_ele_PhiMnPhimatchingObjectVsPhi_matched = new TH2F("h_ele_PhiMnPhimatchingObjectVsPhi_matched",
                                                       "ele momentum phi - matching SC phi vs phi",
                                                       nbinphi2D,
                                                       phimin,
                                                       phimax,
                                                       nbindphi / 2,
                                                       dphimin,
                                                       dphimax);
  h_ele_PhiMnPhimatchingObjectVsPt_matched = new TH2F("h_ele_PhiMnPhimatchingObjectVsPt_matched",
                                                      "ele momentum phi - matching SC phi vs pt",
                                                      nbinpt2D,
                                                      0.,
                                                      ptmax,
                                                      nbindphi / 2,
                                                      dphimin,
                                                      dphimax);

  // matched electron, superclusters
  histSclEn_ = new TH1F("h_scl_energy", "ele supercluster energy", nbinp, 0., pmax);
  histSclEoEmatchingObject_barrel_matched = new TH1F(
      "h_scl_EoEmatchingObject_barrel_matched", "ele supercluster energy / matching SC energy, barrel", 50, 0.2, 1.2);
  histSclEoEmatchingObject_endcaps_matched = new TH1F(
      "h_scl_EoEmatchingObject_endcaps_matched", "ele supercluster energy / matching SC energy, endcaps", 50, 0.2, 1.2);
  histSclEoEmatchingObject_barrel_new_matched = new TH1F("h_scl_EoEmatchingObject_barrel_new_matched",
                                                         "ele supercluster energy / matching SC energy, barrel",
                                                         nbinpoptrue,
                                                         poptruemin,
                                                         poptruemax);
  histSclEoEmatchingObject_endcaps_new_matched = new TH1F("h_scl_EoEmatchingObject_endcaps_new_matched",
                                                          "ele supercluster energy / matching SC energy, endcaps",
                                                          nbinpoptrue,
                                                          poptruemin,
                                                          poptruemax);
  histSclEt_ = new TH1F("h_scl_et", "ele supercluster transverse energy", nbinpt, 0., ptmax);
  histSclEtVsEta_ = new TH2F(
      "h_scl_etVsEta", "ele supercluster transverse energy vs eta", nbineta2D, etamin, etamax, nbinpt, 0., ptmax);
  histSclEtVsPhi_ = new TH2F(
      "h_scl_etVsPhi", "ele supercluster transverse energy vs phi", nbinphi2D, phimin, phimax, nbinpt, 0., ptmax);
  histSclEtaVsPhi_ =
      new TH2F("h_scl_etaVsPhi", "ele supercluster eta vs phi", nbinphi2D, phimin, phimax, nbineta2D, etamin, etamax);
  histSclEta_ = new TH1F("h_scl_eta", "ele supercluster eta", nbineta, etamin, etamax);
  histSclPhi_ = new TH1F("h_scl_phi", "ele supercluster phi", nbinphi, phimin, phimax);

  histSclSigEtaEta_ = new TH1F("h_scl_sigetaeta", "ele supercluster sigma eta eta", 100, 0., 0.05);
  histSclSigIEtaIEta_barrel_ =
      new TH1F("h_scl_sigietaieta_barrel", "ele supercluster sigma ieta ieta, barrel", 100, 0., 0.05);
  histSclSigIEtaIEta_endcaps_ =
      new TH1F("h_scl_sigietaieta_endcaps", "ele supercluster sigma ieta ieta, endcaps", 100, 0., 0.05);
  histSclE1x5_ = new TH1F("h_scl_E1x5", "ele supercluster energy in 1x5", nbinp, 0., pmax);
  histSclE1x5_barrel_ = new TH1F("h_scl_E1x_barrel5", "ele supercluster energy in 1x5 barrel", nbinp, 0., pmax);
  histSclE1x5_endcaps_ = new TH1F("h_scl_E1x5_endcaps", "ele supercluster energy in 1x5 endcaps", nbinp, 0., pmax);
  histSclE2x5max_ = new TH1F("h_scl_E2x5max", "ele supercluster energy in 2x5 max", nbinp, 0., pmax);
  histSclE2x5max_barrel_ =
      new TH1F("h_scl_E2x5max_barrel", "ele supercluster energy in 2x5 max barrel", nbinp, 0., pmax);
  histSclE2x5max_endcaps_ =
      new TH1F("h_scl_E2x5max_endcaps", "ele supercluster energy in 2x5 max endcaps", nbinp, 0., pmax);
  histSclE5x5_ = new TH1F("h_scl_E5x5", "ele supercluster energy in 5x5", nbinp, 0., pmax);
  histSclE5x5_barrel_ = new TH1F("h_scl_E5x5_barrel", "ele supercluster energy in 5x5 barrel", nbinp, 0., pmax);
  histSclE5x5_endcaps_ = new TH1F("h_scl_E5x5_endcaps", "ele supercluster energy in 5x5 endcaps", nbinp, 0., pmax);

  // matched electron, gsf tracks
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
  h_ele_PinMnPout = new TH1F("h_ele_PinMnPout", "ele track inner p - outer p, mean", nbinp, 0., 200.);
  h_ele_PinMnPout_mode = new TH1F("h_ele_PinMnPout_mode", "ele track inner p - outer p, mode", nbinp, 0., 100.);
  h_ele_PinMnPoutVsEta_mode = new TH2F("h_ele_PinMnPoutVsEta_mode",
                                       "ele track inner p - outer p vs eta, mode",
                                       nbineta2D,
                                       etamin,
                                       etamax,
                                       nbinp2D,
                                       0.,
                                       100.);
  h_ele_PinMnPoutVsPhi_mode = new TH2F("h_ele_PinMnPoutVsPhi_mode",
                                       "ele track inner p - outer p vs phi, mode",
                                       nbinphi2D,
                                       phimin,
                                       phimax,
                                       nbinp2D,
                                       0.,
                                       100.);
  h_ele_PinMnPoutVsPt_mode = new TH2F(
      "h_ele_PinMnPoutVsPt_mode", "ele track inner p - outer p vs pt, mode", nbinpt2D, 0., ptmax, nbinp2D, 0., 100.);
  h_ele_PinMnPoutVsE_mode = new TH2F(
      "h_ele_PinMnPoutVsE_mode", "ele track inner p - outer p vs E, mode", nbinp2D, 0., 200., nbinp2D, 0., 100.);
  h_ele_PinMnPoutVsChi2_mode = new TH2F(
      "h_ele_PinMnPoutVsChi2_mode", "ele track inner p - outer p vs track chi2, mode", 50, 0., 20., nbinp2D, 0., 100.);
  h_ele_outerP = new TH1F("h_ele_outerP", "ele track outer p, mean", nbinp, 0., pmax);
  h_ele_outerP_mode = new TH1F("h_ele_outerP_mode", "ele track outer p, mode", nbinp, 0., pmax);
  h_ele_outerPVsEta_mode =
      new TH2F("h_ele_outerPVsEta_mode", "ele track outer p vs eta mode", nbineta2D, etamin, etamax, 50, 0., pmax);
  h_ele_outerPt = new TH1F("h_ele_outerPt", "ele track outer p_{T}, mean", nbinpt, 0., ptmax);
  h_ele_outerPt_mode = new TH1F("h_ele_outerPt_mode", "ele track outer p_{T}, mode", nbinpt, 0., ptmax);
  h_ele_outerPtVsEta_mode = new TH2F(
      "h_ele_outerPtVsEta_mode", "ele track outer p_{T} vs eta, mode", nbineta2D, etamin, etamax, nbinpt2D, 0., ptmax);
  h_ele_outerPtVsPhi_mode = new TH2F(
      "h_ele_outerPtVsPhi_mode", "ele track outer p_{T} vs phi, mode", nbinphi2D, phimin, phimax, nbinpt2D, 0., ptmax);
  h_ele_outerPtVsPt_mode =
      new TH2F("h_ele_outerPtVsPt_mode", "ele track outer p_{T} vs pt, mode", nbinpt2D, 0., 100., nbinpt2D, 0., ptmax);

  // matched electrons, matching
  h_ele_EoP = new TH1F("h_ele_EoP", "ele E/P_{vertex}", nbineop, 0., eopmax);
  h_ele_EoPVsEta =
      new TH2F("h_ele_EoPVsEta", "ele E/P_{vertex} vs eta", nbineta2D, etamin, etamax, nbineop2D, 0., eopmaxsht);
  h_ele_EoPVsPhi =
      new TH2F("h_ele_EoPVsPhi", "ele E/P_{vertex} vs phi", nbinphi2D, phimin, phimax, nbineop2D, 0., eopmaxsht);
  h_ele_EoPVsE = new TH2F("h_ele_EoPVsE", "ele E/P_{vertex} vs E", 50, 0., pmax, 50, 0., 5.);
  h_ele_EseedOP = new TH1F("h_ele_EseedOP", "ele E_{seed}/P_{vertex}", nbineop, 0., eopmax);
  h_ele_EseedOPVsEta = new TH2F(
      "h_ele_EseedOPVsEta", "ele E_{seed}/P_{vertex} vs eta", nbineta2D, etamin, etamax, nbineop2D, 0., eopmaxsht);
  h_ele_EseedOPVsPhi = new TH2F(
      "h_ele_EseedOPVsPhi", "ele E_{seed}/P_{vertex} vs phi", nbinphi2D, phimin, phimax, nbineop2D, 0., eopmaxsht);
  h_ele_EseedOPVsE = new TH2F("h_ele_EseedOPVsE", "ele E_{seed}/P_{vertex} vs E", 50, 0., pmax, 50, 0., 5.);
  h_ele_EoPout = new TH1F("h_ele_EoPout", "ele E/P_{out}", nbineop, 0., eopmax);
  h_ele_EoPoutVsEta =
      new TH2F("h_ele_EoPoutVsEta", "ele E/P_{out} vs eta", nbineta2D, etamin, etamax, nbineop2D, 0., eopmaxsht);
  h_ele_EoPoutVsPhi =
      new TH2F("h_ele_EoPoutVsPhi", "ele E/P_{out} vs phi", nbinphi2D, phimin, phimax, nbineop2D, 0., eopmaxsht);
  h_ele_EoPoutVsE = new TH2F("h_ele_EoPoutVsE", "ele E/P_{out} vs E", nbinp2D, 0., pmax, nbineop2D, 0., eopmaxsht);
  h_ele_EeleOPout = new TH1F("h_ele_EeleOPout", "ele E_{ele}/P_{out}", nbineop, 0., eopmax);
  h_ele_EeleOPoutVsEta = new TH2F(
      "h_ele_EeleOPoutVsEta", "ele E_{ele}/P_{out} vs eta", nbineta2D, etamin, etamax, nbineop2D, 0., eopmaxsht);
  h_ele_EeleOPoutVsPhi = new TH2F(
      "h_ele_EeleOPoutVsPhi", "ele E_{ele}/P_{out} vs phi", nbinphi2D, phimin, phimax, nbineop2D, 0., eopmaxsht);
  h_ele_EeleOPoutVsE =
      new TH2F("h_ele_EeleOPoutVsE", "ele E_{ele}/P_{out} vs E", nbinp2D, 0., pmax, nbineop2D, 0., eopmaxsht);
  h_ele_dEtaSc_propVtx = new TH1F(
      "h_ele_dEtaSc_propVtx", "ele #eta_{sc} - #eta_{tr}, prop from vertex", nbindetamatch, detamatchmin, detamatchmax);
  h_ele_dEtaScVsEta_propVtx = new TH2F("h_ele_dEtaScVsEta_propVtx",
                                       "ele #eta_{sc} - #eta_{tr} vs eta, prop from vertex",
                                       nbineta2D,
                                       etamin,
                                       etamax,
                                       nbindetamatch2D,
                                       detamatchmin,
                                       detamatchmax);
  h_ele_dEtaScVsPhi_propVtx = new TH2F("h_ele_dEtaScVsPhi_propVtx",
                                       "ele #eta_{sc} - #eta_{tr} vs phi, prop from vertex",
                                       nbinphi2D,
                                       phimin,
                                       phimax,
                                       nbindetamatch2D,
                                       detamatchmin,
                                       detamatchmax);
  h_ele_dEtaScVsPt_propVtx = new TH2F("h_ele_dEtaScVsPt_propVtx",
                                      "ele #eta_{sc} - #eta_{tr} vs pt, prop from vertex",
                                      nbinpt2D,
                                      0.,
                                      ptmax,
                                      nbindetamatch2D,
                                      detamatchmin,
                                      detamatchmax);
  h_ele_dPhiSc_propVtx = new TH1F(
      "h_ele_dPhiSc_propVtx", "ele #phi_{sc} - #phi_{tr}, prop from vertex", nbindphimatch, dphimatchmin, dphimatchmax);
  h_ele_dPhiScVsEta_propVtx = new TH2F("h_ele_dPhiScVsEta_propVtx",
                                       "ele #phi_{sc} - #phi_{tr} vs eta, prop from vertex",
                                       nbineta2D,
                                       etamin,
                                       etamax,
                                       nbindphimatch2D,
                                       dphimatchmin,
                                       dphimatchmax);
  h_ele_dPhiScVsPhi_propVtx = new TH2F("h_ele_dPhiScVsPhi_propVtx",
                                       "ele #phi_{sc} - #phi_{tr} vs phi, prop from vertex",
                                       nbinphi2D,
                                       phimin,
                                       phimax,
                                       nbindphimatch2D,
                                       dphimatchmin,
                                       dphimatchmax);
  h_ele_dPhiScVsPt_propVtx = new TH2F("h_ele_dPhiScVsPt_propVtx",
                                      "ele #phi_{sc} - #phi_{tr} vs pt, prop from vertex",
                                      nbinpt2D,
                                      0.,
                                      ptmax,
                                      nbindphimatch2D,
                                      dphimatchmin,
                                      dphimatchmax);
  h_ele_dEtaCl_propOut = new TH1F("h_ele_dEtaCl_propOut",
                                  "ele #eta_{cl} - #eta_{tr}, prop from outermost",
                                  nbindetamatch,
                                  detamatchmin,
                                  detamatchmax);
  h_ele_dEtaClVsEta_propOut = new TH2F("h_ele_dEtaClVsEta_propOut",
                                       "ele #eta_{cl} - #eta_{tr} vs eta, prop from out",
                                       nbineta2D,
                                       etamin,
                                       etamax,
                                       nbindetamatch2D,
                                       detamatchmin,
                                       detamatchmax);
  h_ele_dEtaClVsPhi_propOut = new TH2F("h_ele_dEtaClVsPhi_propOut",
                                       "ele #eta_{cl} - #eta_{tr} vs phi, prop from out",
                                       nbinphi2D,
                                       phimin,
                                       phimax,
                                       nbindetamatch2D,
                                       detamatchmin,
                                       detamatchmax);
  h_ele_dEtaClVsPt_propOut = new TH2F("h_ele_dEtaScVsPt_propOut",
                                      "ele #eta_{cl} - #eta_{tr} vs pt, prop from out",
                                      nbinpt2D,
                                      0.,
                                      ptmax,
                                      nbindetamatch2D,
                                      detamatchmin,
                                      detamatchmax);
  h_ele_dPhiCl_propOut = new TH1F("h_ele_dPhiCl_propOut",
                                  "ele #phi_{cl} - #phi_{tr}, prop from outermost",
                                  nbindphimatch,
                                  dphimatchmin,
                                  dphimatchmax);
  h_ele_dPhiClVsEta_propOut = new TH2F("h_ele_dPhiClVsEta_propOut",
                                       "ele #phi_{cl} - #phi_{tr} vs eta, prop from out",
                                       nbineta2D,
                                       etamin,
                                       etamax,
                                       nbindphimatch2D,
                                       dphimatchmin,
                                       dphimatchmax);
  h_ele_dPhiClVsPhi_propOut = new TH2F("h_ele_dPhiClVsPhi_propOut",
                                       "ele #phi_{cl} - #phi_{tr} vs phi, prop from out",
                                       nbinphi2D,
                                       phimin,
                                       phimax,
                                       nbindphimatch2D,
                                       dphimatchmin,
                                       dphimatchmax);
  h_ele_dPhiClVsPt_propOut = new TH2F("h_ele_dPhiSClsPt_propOut",
                                      "ele #phi_{cl} - #phi_{tr} vs pt, prop from out",
                                      nbinpt2D,
                                      0.,
                                      ptmax,
                                      nbindphimatch2D,
                                      dphimatchmin,
                                      dphimatchmax);
  h_ele_dEtaEleCl_propOut = new TH1F("h_ele_dEtaEleCl_propOut",
                                     "ele #eta_{EleCl} - #eta_{tr}, prop from outermost",
                                     nbindetamatch,
                                     detamatchmin,
                                     detamatchmax);
  h_ele_dEtaEleClVsEta_propOut = new TH2F("h_ele_dEtaEleClVsEta_propOut",
                                          "ele #eta_{EleCl} - #eta_{tr} vs eta, prop from out",
                                          nbineta2D,
                                          etamin,
                                          etamax,
                                          nbindetamatch2D,
                                          detamatchmin,
                                          detamatchmax);
  h_ele_dEtaEleClVsPhi_propOut = new TH2F("h_ele_dEtaEleClVsPhi_propOut",
                                          "ele #eta_{EleCl} - #eta_{tr} vs phi, prop from out",
                                          nbinphi2D,
                                          phimin,
                                          phimax,
                                          nbindetamatch2D,
                                          detamatchmin,
                                          detamatchmax);
  h_ele_dEtaEleClVsPt_propOut = new TH2F("h_ele_dEtaScVsPt_propOut",
                                         "ele #eta_{EleCl} - #eta_{tr} vs pt, prop from out",
                                         nbinpt2D,
                                         0.,
                                         ptmax,
                                         nbindetamatch2D,
                                         detamatchmin,
                                         detamatchmax);
  h_ele_dPhiEleCl_propOut = new TH1F("h_ele_dPhiEleCl_propOut",
                                     "ele #phi_{EleCl} - #phi_{tr}, prop from outermost",
                                     nbindphimatch,
                                     dphimatchmin,
                                     dphimatchmax);
  h_ele_dPhiEleClVsEta_propOut = new TH2F("h_ele_dPhiEleClVsEta_propOut",
                                          "ele #phi_{EleCl} - #phi_{tr} vs eta, prop from out",
                                          nbineta2D,
                                          etamin,
                                          etamax,
                                          nbindphimatch2D,
                                          dphimatchmin,
                                          dphimatchmax);
  h_ele_dPhiEleClVsPhi_propOut = new TH2F("h_ele_dPhiEleClVsPhi_propOut",
                                          "ele #phi_{EleCl} - #phi_{tr} vs phi, prop from out",
                                          nbinphi2D,
                                          phimin,
                                          phimax,
                                          nbindphimatch2D,
                                          dphimatchmin,
                                          dphimatchmax);
  h_ele_dPhiEleClVsPt_propOut = new TH2F("h_ele_dPhiSEleClsPt_propOut",
                                         "ele #phi_{EleCl} - #phi_{tr} vs pt, prop from out",
                                         nbinpt2D,
                                         0.,
                                         ptmax,
                                         nbindphimatch2D,
                                         dphimatchmin,
                                         dphimatchmax);

  h_ele_HoE = new TH1F("h_ele_HoE", "ele hadronic energy / em energy", nbinhoe, hoemin, hoemax);
  h_ele_HoE_fiducial =
      new TH1F("h_ele_HoE_fiducial", "ele hadronic energy / em energy, fiducial region", nbinhoe, hoemin, hoemax);
  h_ele_HoEVsEta = new TH2F(
      "h_ele_HoEVsEta", "ele hadronic energy / em energy vs eta", nbineta, etamin, etamax, nbinhoe, hoemin, hoemax);
  h_ele_HoEVsPhi = new TH2F(
      "h_ele_HoEVsPhi", "ele hadronic energy / em energy vs phi", nbinphi2D, phimin, phimax, nbinhoe, hoemin, hoemax);
  h_ele_HoEVsE =
      new TH2F("h_ele_HoEVsE", "ele hadronic energy / em energy vs E", nbinp, 0., 300., nbinhoe, hoemin, hoemax);

  h_ele_seed_dphi2_ = new TH1F("h_ele_seedDphi2", "ele seed dphi 2nd layer", 50, -0.003, +0.003);
  h_ele_seed_dphi2VsEta_ =
      new TH2F("h_ele_seedDphi2VsEta", "ele seed dphi 2nd layer vs eta", nbineta2D, etamin, etamax, 50, -0.003, +0.003);
  h_ele_seed_dphi2VsPt_ =
      new TH2F("h_ele_seedDphi2VsPt", "ele seed dphi 2nd layer vs pt", nbinpt2D, 0., ptmax, 50, -0.003, +0.003);
  h_ele_seed_drz2_ = new TH1F("h_ele_seedDrz2", "ele seed dr/dz 2nd layer", 50, -0.03, +0.03);
  h_ele_seed_drz2VsEta_ =
      new TH2F("h_ele_seedDrz2VsEta", "ele seed dr/dz 2nd layer vs eta", nbineta2D, etamin, etamax, 50, -0.03, +0.03);
  h_ele_seed_drz2VsPt_ =
      new TH2F("h_ele_seedDrz2VsPt", "ele seed dr/dz 2nd layer vs pt", nbinpt2D, 0., ptmax, 50, -0.03, +0.03);
  h_ele_seed_subdet2_ = new TH1F("h_ele_seedSubdet2", "ele seed subdet 2nd layer", 10, 0., 10.);

  // classes
  h_ele_classes = new TH1F("h_ele_classes", "electron classes", 20, 0.0, 20.);
  h_ele_eta = new TH1F("h_ele_eta", "ele electron eta", nbineta / 2, 0.0, etamax);
  h_ele_eta_golden = new TH1F("h_ele_eta_golden", "ele electron eta golden", nbineta / 2, 0.0, etamax);
  h_ele_eta_bbrem = new TH1F("h_ele_eta_bbrem", "ele electron eta bbrem", nbineta / 2, 0.0, etamax);
  h_ele_eta_narrow = new TH1F("h_ele_eta_narrow", "ele electron eta narrow", nbineta / 2, 0.0, etamax);
  h_ele_eta_shower = new TH1F("h_ele_eta_show", "ele electron eta showering", nbineta / 2, 0.0, etamax);
  h_ele_PinVsPoutGolden_mode = new TH2F("h_ele_PinVsPoutGolden_mode",
                                        "ele track inner p vs outer p vs eta, golden, mode",
                                        nbinp2D,
                                        0.,
                                        pmax,
                                        50,
                                        0.,
                                        pmax);
  h_ele_PinVsPoutShowering_mode = new TH2F("h_ele_PinVsPoutShowering_mode",
                                           "ele track inner p vs outer p vs eta, Showering, mode",
                                           nbinp2D,
                                           0.,
                                           pmax,
                                           50,
                                           0.,
                                           pmax);
  h_ele_PinVsPoutGolden_mean = new TH2F("h_ele_PinVsPoutGolden_mean",
                                        "ele track inner p vs outer p vs eta, golden, mean",
                                        nbinp2D,
                                        0.,
                                        pmax,
                                        50,
                                        0.,
                                        pmax);
  h_ele_PinVsPoutShowering_mean = new TH2F("h_ele_PinVsPoutShowering_mean",
                                           "ele track inner p vs outer p vs eta, Showering, mean",
                                           nbinp2D,
                                           0.,
                                           pmax,
                                           50,
                                           0.,
                                           pmax);
  h_ele_PtinVsPtoutGolden_mode = new TH2F("h_ele_PtinVsPtoutGolden_mode",
                                          "ele track inner pt vs outer pt vs eta, golden, mode",
                                          nbinpt2D,
                                          0.,
                                          ptmax,
                                          50,
                                          0.,
                                          ptmax);
  h_ele_PtinVsPtoutShowering_mode = new TH2F("h_ele_PtinVsPtoutShowering_mode",
                                             "ele track inner pt vs outer pt vs eta, showering, mode",
                                             nbinpt2D,
                                             0.,
                                             ptmax,
                                             50,
                                             0.,
                                             ptmax);
  h_ele_PtinVsPtoutGolden_mean = new TH2F("h_ele_PtinVsPtoutGolden_mean",
                                          "ele track inner pt vs outer pt vs eta, golden, mean",
                                          nbinpt2D,
                                          0.,
                                          ptmax,
                                          50,
                                          0.,
                                          ptmax);
  h_ele_PtinVsPtoutShowering_mean = new TH2F("h_ele_PtinVsPtoutShowering_mean",
                                             "ele track inner pt vs outer pt vs eta, showering, mean",
                                             nbinpt2D,
                                             0.,
                                             ptmax,
                                             50,
                                             0.,
                                             ptmax);
  histSclEoEmatchingObjectGolden_barrel = new TH1F("h_scl_EoEmatchingObject golden, barrel",
                                                   "ele supercluster energy over matchingObject energy, golden, barrel",
                                                   100,
                                                   0.2,
                                                   1.2);
  histSclEoEmatchingObjectGolden_endcaps =
      new TH1F("h_scl_EoEmatchingObject golden, endcaps",
               "ele supercluster energy over matchingObject energy, golden, endcaps",
               100,
               0.2,
               1.2);
  histSclEoEmatchingObjectShowering_barrel =
      new TH1F("h_scl_EoEmatchingObject Showering, barrel",
               "ele supercluster energy over matchingObject energy, showering, barrel",
               100,
               0.2,
               1.2);
  histSclEoEmatchingObjectShowering_endcaps =
      new TH1F("h_scl_EoEmatchingObject Showering, endcaps",
               "ele supercluster energy over matchingObject energy, showering, endcaps",
               100,
               0.2,
               1.2);

  // isolation
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

  // fbrem
  h_ele_fbrem = new TH1F("h_ele_fbrem", "ele brem fraction, mode", 100, 0., 1.);
  h_ele_fbremVsEta_mode =
      new TProfile("h_ele_fbremvsEtamode", "mean ele brem fraction vs eta, mode", nbineta2D, etamin, etamax, 0., 1.);
  h_ele_fbremVsEta_mean =
      new TProfile("h_ele_fbremvsEtamean", "mean ele brem fraction vs eta, mean", nbineta2D, etamin, etamax, 0., 1.);

  // e/g et pflow electrons
  h_ele_mva = new TH1F("h_ele_mva", "ele identification mva", 100, -1., 1.);
  h_ele_provenance = new TH1F("h_ele_provenance", "ele provenance", 5, -2., 3.);

  // histos titles
  h_matchingObjectNum->GetXaxis()->SetTitle("N_{SC}");
  h_matchingObjectNum->GetYaxis()->SetTitle("Events");
  h_matchingObjectEta->GetXaxis()->SetTitle("#eta_{SC}");
  h_matchingObjectEta->GetYaxis()->SetTitle("Events");
  h_matchingObjectP->GetXaxis()->SetTitle("E_{SC} (GeV)");
  h_matchingObjectP->GetYaxis()->SetTitle("Events");
  h_ele_foundHits->GetXaxis()->SetTitle("N_{hits}");
  h_ele_foundHits->GetYaxis()->SetTitle("Events");
  h_ele_ambiguousTracks->GetXaxis()->SetTitle("N_{ambiguous tracks}");
  h_ele_ambiguousTracks->GetYaxis()->SetTitle("Events");
  h_ele_lostHits->GetXaxis()->SetTitle("N_{lost hits}");
  h_ele_lostHits->GetYaxis()->SetTitle("Events");
  h_ele_chi2->GetXaxis()->SetTitle("#Chi^{2}");
  h_ele_chi2->GetYaxis()->SetTitle("Events");
  h_ele_charge->GetXaxis()->SetTitle("charge");
  h_ele_charge->GetYaxis()->SetTitle("Events");
  h_ele_vertexP->GetXaxis()->SetTitle("p_{vertex} (GeV/c)");
  h_ele_vertexP->GetYaxis()->SetTitle("Events");
  h_ele_vertexPt->GetXaxis()->SetTitle("p_{T vertex} (GeV/c)");
  h_ele_vertexPt->GetYaxis()->SetTitle("Events");
  h_ele_Et->GetXaxis()->SetTitle("E_{T} (GeV)");
  h_ele_Et->GetYaxis()->SetTitle("Events");
  h_ele_vertexEta->GetXaxis()->SetTitle("#eta");
  h_ele_vertexEta->GetYaxis()->SetTitle("Events");
  h_ele_vertexPhi->GetXaxis()->SetTitle("#phi (rad)");
  h_ele_vertexPhi->GetYaxis()->SetTitle("Events");
  h_ele_PoPmatchingObject_matched->GetXaxis()->SetTitle("P/E_{SC}");
  h_ele_PoPmatchingObject_matched->GetYaxis()->SetTitle("Events");
  h_ele_PoPmatchingObject_barrel_matched->GetXaxis()->SetTitle("P/E_{SC}");
  h_ele_PoPmatchingObject_barrel_matched->GetYaxis()->SetTitle("Events");
  h_ele_PoPmatchingObject_endcaps_matched->GetXaxis()->SetTitle("P/E_{SC}");
  h_ele_PoPmatchingObject_endcaps_matched->GetYaxis()->SetTitle("Events");
  h_ele_PtoPtmatchingObject_matched->GetXaxis()->SetTitle("P_{T}/E_{T}^{SC}");
  h_ele_PtoPtmatchingObject_matched->GetYaxis()->SetTitle("Events");
  h_ele_PtoPtmatchingObject_barrel_matched->GetXaxis()->SetTitle("P_{T}/E_{T}^{SC}");
  h_ele_PtoPtmatchingObject_barrel_matched->GetYaxis()->SetTitle("Events");
  h_ele_PtoPtmatchingObject_endcaps_matched->GetXaxis()->SetTitle("P_{T}/E_{T}^{SC}");
  h_ele_PtoPtmatchingObject_endcaps_matched->GetYaxis()->SetTitle("Events");
  histSclSigEtaEta_->GetXaxis()->SetTitle("#sigma_{#eta #eta}");
  histSclSigEtaEta_->GetYaxis()->SetTitle("Events");
  histSclSigIEtaIEta_barrel_->GetXaxis()->SetTitle("#sigma_{i#eta i#eta}");
  histSclSigIEtaIEta_barrel_->GetYaxis()->SetTitle("Events");
  histSclSigIEtaIEta_endcaps_->GetXaxis()->SetTitle("#sigma_{i#eta i#eta}");
  histSclSigIEtaIEta_endcaps_->GetYaxis()->SetTitle("Events");
  histSclE1x5_->GetXaxis()->SetTitle("E1x5 (GeV)");
  histSclE1x5_->GetYaxis()->SetTitle("Events");
  histSclE1x5_barrel_->GetXaxis()->SetTitle("E1x5 (GeV)");
  histSclE1x5_barrel_->GetYaxis()->SetTitle("Events");
  histSclE1x5_endcaps_->GetXaxis()->SetTitle("E1x5 (GeV)");
  histSclE1x5_endcaps_->GetYaxis()->SetTitle("Events");
  histSclE2x5max_->GetXaxis()->SetTitle("E2x5 (GeV)");
  histSclE2x5max_->GetYaxis()->SetTitle("Events");
  histSclE2x5max_barrel_->GetXaxis()->SetTitle("E2x5 (GeV)");
  histSclE2x5max_barrel_->GetYaxis()->SetTitle("Events");
  histSclE2x5max_endcaps_->GetXaxis()->SetTitle("E2x5 (GeV)");
  histSclE2x5max_endcaps_->GetYaxis()->SetTitle("Events");
  histSclE5x5_->GetXaxis()->SetTitle("E5x5 (GeV)");
  histSclE5x5_->GetYaxis()->SetTitle("Events");
  histSclE5x5_barrel_->GetXaxis()->SetTitle("E5x5 (GeV)");
  histSclE5x5_barrel_->GetYaxis()->SetTitle("Events");
  histSclE5x5_endcaps_->GetXaxis()->SetTitle("E5x5 (GeV)");
  histSclE5x5_endcaps_->GetYaxis()->SetTitle("Events");
  h_ele_EtaMnEtamatchingObject_matched->GetXaxis()->SetTitle("#eta_{rec} - #eta_{SC}");
  h_ele_EtaMnEtamatchingObject_matched->GetYaxis()->SetTitle("Events");
  h_ele_PhiMnPhimatchingObject_matched->GetXaxis()->SetTitle("#phi_{rec} - #phi_{SC} (rad)");
  h_ele_PhiMnPhimatchingObject_matched->GetYaxis()->SetTitle("Events");
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
  h_ele_EseedOP->GetXaxis()->SetTitle("E_{seed}/P_{vertex}");
  h_ele_EseedOP->GetYaxis()->SetTitle("Events");
  h_ele_EoPout->GetXaxis()->SetTitle("E_{seed}/P_{out}");
  h_ele_EoPout->GetYaxis()->SetTitle("Events");
  h_ele_EeleOPout->GetXaxis()->SetTitle("E_{ele}/P_{out}");
  h_ele_EeleOPout->GetYaxis()->SetTitle("Events");
  h_ele_vertexX->GetXaxis()->SetTitle("x (cm)");
  h_ele_vertexX->GetYaxis()->SetTitle("Events");
  h_ele_vertexY->GetXaxis()->SetTitle("y (cm)");
  h_ele_vertexY->GetYaxis()->SetTitle("Events");
  h_ele_vertexZ->GetXaxis()->SetTitle("z (cm)");
  h_ele_vertexZ->GetYaxis()->SetTitle("Events");
  h_ele_vertexTIP->GetXaxis()->SetTitle("TIP (cm)");
  h_ele_vertexTIP->GetYaxis()->SetTitle("Events");
  h_ele_vertexTIPVsEta->GetYaxis()->SetTitle("TIP (cm)");
  h_ele_vertexTIPVsEta->GetXaxis()->SetTitle("#eta");
  h_ele_vertexTIPVsPhi->GetYaxis()->SetTitle("TIP (cm)");
  h_ele_vertexTIPVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  h_ele_vertexTIPVsPt->GetYaxis()->SetTitle("TIP (cm)");
  h_ele_vertexTIPVsEta->GetXaxis()->SetTitle("p_{T} (GeV/c)");
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
  h_ele_HoE_fiducial->GetXaxis()->SetTitle("H/E");
  h_ele_HoE_fiducial->GetYaxis()->SetTitle("Events");
  h_ele_fbrem->GetXaxis()->SetTitle("P_{in} - P_{out} / P_{in}");
  h_ele_fbrem->GetYaxis()->SetTitle("Events");
  h_ele_seed_dphi2_->GetXaxis()->SetTitle("#phi_{hit}-#phi_{pred} (rad)");
  h_ele_seed_dphi2_->GetYaxis()->SetTitle("Events");
  h_ele_seed_drz2_->GetXaxis()->SetTitle("r(z)_{hit}-r(z)_{pred} (cm)");
  h_ele_seed_drz2_->GetYaxis()->SetTitle("Events");
  h_ele_seed_subdet2_->GetXaxis()->SetTitle("2nd hit subdet Id");
  h_ele_seed_subdet2_->GetYaxis()->SetTitle("Events");
  h_ele_classes->GetXaxis()->SetTitle("class Id");
  h_ele_classes->GetYaxis()->SetTitle("Events");
  h_ele_mee_all->GetXaxis()->SetTitle("m_{ee} (GeV/c^{2})");
  h_ele_mee_all->GetYaxis()->SetTitle("Events");
  h_ele_mee_os->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os->GetYaxis()->SetTitle("Events");
  h_ele_mee_os_ebeb->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os_ebeb->GetYaxis()->SetTitle("Events");
  h_ele_mee_os_ebee->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os_ebee->GetYaxis()->SetTitle("Events");
  h_ele_mee_os_eeee->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os_eeee->GetYaxis()->SetTitle("Events");
  h_ele_mee_os_gg->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os_gg->GetYaxis()->SetTitle("Events");
  h_ele_mee_os_gb->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os_gb->GetYaxis()->SetTitle("Events");
  h_ele_mee_os_bb->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os_bb->GetYaxis()->SetTitle("Events");
  h_ele_E2mnE1vsMee_all->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_E2mnE1vsMee_all->GetYaxis()->SetTitle("E2 - E1 (GeV)");
  h_ele_E2mnE1vsMee_egeg_all->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_E2mnE1vsMee_egeg_all->GetYaxis()->SetTitle("E2 - E1 (GeV)");
  histNum_->GetXaxis()->SetTitle("N_{ele}");
  histNum_->GetYaxis()->SetTitle("Events");
  h_ele_fbremVsEta_mode->GetXaxis()->SetTitle("#eta");
  h_ele_fbremVsEta_mean->GetXaxis()->SetTitle("#eta");
}

void GsfElectronDataAnalyzer::endJob() {
  histfile_->cd();
  std::cout << "efficiency calculation " << std::endl;
  // efficiency vs eta
  TH1F *h_ele_etaEff = (TH1F *)h_ele_matchingObjectEta_matched->Clone("h_ele_etaEff");
  h_ele_etaEff->Reset();
  h_ele_etaEff->Divide(h_ele_matchingObjectEta_matched, h_matchingObjectEta, 1, 1, "b");
  h_ele_etaEff->Print();
  h_ele_etaEff->GetXaxis()->SetTitle("#eta");
  h_ele_etaEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs z
  TH1F *h_ele_zEff = (TH1F *)h_ele_matchingObjectZ_matched->Clone("h_ele_zEff");
  h_ele_zEff->Reset();
  h_ele_zEff->Divide(h_ele_matchingObjectZ_matched, h_matchingObjectZ, 1, 1, "b");
  h_ele_zEff->Print();
  h_ele_zEff->GetXaxis()->SetTitle("z (cm)");
  h_ele_zEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs |eta|
  TH1F *h_ele_absetaEff = (TH1F *)h_ele_matchingObjectAbsEta_matched->Clone("h_ele_absetaEff");
  h_ele_absetaEff->Reset();
  h_ele_absetaEff->Divide(h_ele_matchingObjectAbsEta_matched, h_matchingObjectAbsEta, 1, 1, "b");
  h_ele_absetaEff->GetXaxis()->SetTitle("|#eta|");
  h_ele_absetaEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs pt
  TH1F *h_ele_ptEff = (TH1F *)h_ele_matchingObjectPt_matched->Clone("h_ele_ptEff");
  h_ele_ptEff->Reset();
  h_ele_ptEff->Divide(h_ele_matchingObjectPt_matched, h_matchingObjectPt, 1, 1, "b");
  h_ele_ptEff->GetXaxis()->SetTitle("p_{T} (GeV/c)");
  h_ele_ptEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs phi
  TH1F *h_ele_phiEff = (TH1F *)h_ele_matchingObjectPhi_matched->Clone("h_ele_phiEff");
  h_ele_phiEff->Reset();
  h_ele_phiEff->Divide(h_ele_matchingObjectPhi_matched, h_matchingObjectPhi, 1, 1, "b");
  h_ele_phiEff->GetXaxis()->SetTitle("#phi (rad)");
  h_ele_phiEff->GetYaxis()->SetTitle("Efficiency");

  // classes
  TH1F *h_ele_eta_goldenFrac = (TH1F *)h_ele_eta_golden->Clone("h_ele_eta_goldenFrac");
  h_ele_eta_goldenFrac->Reset();
  h_ele_eta_goldenFrac->Divide(h_ele_eta_golden, h_ele_eta, 1, 1);
  h_ele_eta_goldenFrac->GetXaxis()->SetTitle("|#eta|");
  h_ele_eta_goldenFrac->GetYaxis()->SetTitle("Fraction of electrons");
  h_ele_eta_goldenFrac->SetTitle("fraction of golden electrons vs eta");
  TH1F *h_ele_eta_bbremFrac = (TH1F *)h_ele_eta_bbrem->Clone("h_ele_eta_bbremFrac");
  h_ele_eta_bbremFrac->Reset();
  h_ele_eta_bbremFrac->Divide(h_ele_eta_bbrem, h_ele_eta, 1, 1);
  h_ele_eta_bbremFrac->GetXaxis()->SetTitle("|#eta|");
  h_ele_eta_bbremFrac->GetYaxis()->SetTitle("Fraction of electrons");
  h_ele_eta_bbremFrac->SetTitle("fraction of big brem electrons vs eta");
  TH1F *h_ele_eta_narrowFrac = (TH1F *)h_ele_eta_narrow->Clone("h_ele_eta_narrowFrac");
  h_ele_eta_narrowFrac->Reset();
  h_ele_eta_narrowFrac->Divide(h_ele_eta_narrow, h_ele_eta, 1, 1);
  h_ele_eta_narrowFrac->GetXaxis()->SetTitle("|#eta|");
  h_ele_eta_narrowFrac->GetYaxis()->SetTitle("Fraction of electrons");
  h_ele_eta_narrowFrac->SetTitle("fraction of narrow electrons vs eta");
  TH1F *h_ele_eta_showerFrac = (TH1F *)h_ele_eta_shower->Clone("h_ele_eta_showerFrac");
  h_ele_eta_showerFrac->Reset();
  h_ele_eta_showerFrac->Divide(h_ele_eta_shower, h_ele_eta, 1, 1);
  h_ele_eta_showerFrac->GetXaxis()->SetTitle("|#eta|");
  h_ele_eta_showerFrac->GetYaxis()->SetTitle("Fraction of electrons");
  h_ele_eta_showerFrac->SetTitle("fraction of showering electrons vs eta");

  // fbrem
  TH1F *h_ele_xOverX0VsEta = new TH1F("h_ele_xOverx0VsEta", "mean X/X_0 vs eta", nbineta / 2, 0.0, 2.5);
  for (int ibin = 1; ibin < h_ele_fbremVsEta_mean->GetNbinsX() + 1; ibin++) {
    double xOverX0 = 0.;
    if (h_ele_fbremVsEta_mean->GetBinContent(ibin) > 0.)
      xOverX0 = -log(h_ele_fbremVsEta_mean->GetBinContent(ibin));
    h_ele_xOverX0VsEta->SetBinContent(ibin, xOverX0);
  }

  //profiles from 2D histos
  TProfile *p_ele_PoPmatchingObjectVsEta_matched = h_ele_PoPmatchingObjectVsEta_matched->ProfileX();
  p_ele_PoPmatchingObjectVsEta_matched->SetTitle("mean ele momentum / matching SC energy vs eta");
  p_ele_PoPmatchingObjectVsEta_matched->GetXaxis()->SetTitle("#eta");
  p_ele_PoPmatchingObjectVsEta_matched->GetYaxis()->SetTitle("<P/E_{matching SC}>");
  p_ele_PoPmatchingObjectVsEta_matched->Write();
  TProfile *p_ele_PoPmatchingObjectVsPhi_matched = h_ele_PoPmatchingObjectVsPhi_matched->ProfileX();
  p_ele_PoPmatchingObjectVsPhi_matched->SetTitle("mean ele momentum / gen momentum vs phi");
  p_ele_PoPmatchingObjectVsPhi_matched->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_PoPmatchingObjectVsPhi_matched->GetYaxis()->SetTitle("<P/E_{matching SC}>");
  p_ele_PoPmatchingObjectVsPhi_matched->Write();
  TProfile *p_ele_EtaMnEtamatchingObjectVsEta_matched = h_ele_EtaMnEtamatchingObjectVsEta_matched->ProfileX();
  p_ele_EtaMnEtamatchingObjectVsEta_matched->GetXaxis()->SetTitle("#eta");
  p_ele_EtaMnEtamatchingObjectVsEta_matched->GetYaxis()->SetTitle("<#eta_{rec} - #eta_{matching SC}>");
  p_ele_EtaMnEtamatchingObjectVsEta_matched->Write();
  TProfile *p_ele_EtaMnEtamatchingObjectVsPhi_matched = h_ele_EtaMnEtamatchingObjectVsPhi_matched->ProfileX();
  p_ele_EtaMnEtamatchingObjectVsPhi_matched->GetXaxis()->SetTitle("#phi");
  p_ele_EtaMnEtamatchingObjectVsPhi_matched->GetYaxis()->SetTitle("<#eta_{rec} - #eta_{matching SC}>");
  p_ele_EtaMnEtamatchingObjectVsPhi_matched->Write();
  TProfile *p_ele_PhiMnPhimatchingObjectVsEta_matched = h_ele_PhiMnPhimatchingObjectVsEta_matched->ProfileX();
  p_ele_PhiMnPhimatchingObjectVsEta_matched->GetXaxis()->SetTitle("#eta");
  p_ele_PhiMnPhimatchingObjectVsEta_matched->GetYaxis()->SetTitle("<#phi_{rec} - #phi_{matching SC}> (rad)");
  p_ele_PhiMnPhimatchingObjectVsEta_matched->Write();
  TProfile *p_ele_PhiMnPhimatchingObjectVsPhi_matched = h_ele_PhiMnPhimatchingObjectVsPhi_matched->ProfileX();
  p_ele_PhiMnPhimatchingObjectVsPhi_matched->GetXaxis()->SetTitle("#phi");
  p_ele_PhiMnPhimatchingObjectVsPhi_matched->GetYaxis()->SetTitle("<#phi_{rec} - #phi_{matching SC}> (rad)");
  p_ele_PhiMnPhimatchingObjectVsPhi_matched->Write();
  TProfile *p_ele_vertexPtVsEta = h_ele_vertexPtVsEta->ProfileX();
  p_ele_vertexPtVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_vertexPtVsEta->GetYaxis()->SetTitle("<p_{T}> (GeV/c)");
  p_ele_vertexPtVsEta->Write();
  TProfile *p_ele_vertexPtVsPhi = h_ele_vertexPtVsPhi->ProfileX();
  p_ele_vertexPtVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_vertexPtVsPhi->GetYaxis()->SetTitle("<p_{T}> (GeV/c)");
  p_ele_vertexPtVsPhi->Write();
  TProfile *p_ele_EoPVsEta = h_ele_EoPVsEta->ProfileX();
  p_ele_EoPVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_EoPVsEta->GetYaxis()->SetTitle("<E/P_{vertex}>");
  p_ele_EoPVsEta->Write();
  TProfile *p_ele_EoPVsPhi = h_ele_EoPVsPhi->ProfileX();
  p_ele_EoPVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_EoPVsPhi->GetYaxis()->SetTitle("<E/P_{vertex}>");
  p_ele_EoPVsPhi->Write();
  TProfile *p_ele_EoPoutVsEta = h_ele_EoPoutVsEta->ProfileX();
  p_ele_EoPoutVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_EoPoutVsEta->GetYaxis()->SetTitle("<E_{seed}/P_{out}>");
  p_ele_EoPoutVsEta->Write();
  TProfile *p_ele_EoPoutVsPhi = h_ele_EoPoutVsPhi->ProfileX();
  p_ele_EoPoutVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_EoPoutVsPhi->GetYaxis()->SetTitle("<E_{seed}/P_{out}>");
  p_ele_EoPoutVsPhi->Write();
  TProfile *p_ele_EeleOPoutVsEta = h_ele_EeleOPoutVsEta->ProfileX();
  p_ele_EeleOPoutVsEta->SetTitle("mean ele Eele/pout vs eta");
  p_ele_EeleOPoutVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_EeleOPoutVsEta->GetYaxis()->SetTitle("<E_{ele}/P_{out}>");
  p_ele_EeleOPoutVsEta->Write();
  TProfile *p_ele_EeleOPoutVsPhi = h_ele_EeleOPoutVsPhi->ProfileX();
  p_ele_EeleOPoutVsPhi->SetTitle("mean ele Eele/pout vs phi");
  p_ele_EeleOPoutVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_EeleOPoutVsPhi->GetYaxis()->SetTitle("<E_{ele}/P_{out}>");
  p_ele_EeleOPoutVsPhi->Write();
  TProfile *p_ele_HoEVsEta = h_ele_HoEVsEta->ProfileX();
  p_ele_HoEVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_HoEVsEta->GetYaxis()->SetTitle("<H/E>");
  p_ele_HoEVsEta->Write();
  TProfile *p_ele_HoEVsPhi = h_ele_HoEVsPhi->ProfileX();
  p_ele_HoEVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_HoEVsPhi->GetYaxis()->SetTitle("<H/E>");
  p_ele_HoEVsPhi->Write();
  TProfile *p_ele_chi2VsEta = h_ele_chi2VsEta->ProfileX();
  p_ele_chi2VsEta->GetXaxis()->SetTitle("#eta");
  p_ele_chi2VsEta->GetYaxis()->SetTitle("<#Chi^{2}>");
  p_ele_chi2VsEta->Write();
  TProfile *p_ele_chi2VsPhi = h_ele_chi2VsPhi->ProfileX();
  p_ele_chi2VsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_chi2VsPhi->GetYaxis()->SetTitle("<#Chi^{2}>");
  p_ele_chi2VsPhi->Write();
  TProfile *p_ele_foundHitsVsEta = h_ele_foundHitsVsEta->ProfileX();
  p_ele_foundHitsVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_foundHitsVsEta->GetYaxis()->SetTitle("<N_{hits}>");
  p_ele_foundHitsVsEta->Write();
  TProfile *p_ele_foundHitsVsPhi = h_ele_foundHitsVsPhi->ProfileX();
  p_ele_foundHitsVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_foundHitsVsPhi->GetYaxis()->SetTitle("<N_{hits}>");
  p_ele_foundHitsVsPhi->Write();
  TProfile *p_ele_lostHitsVsEta = h_ele_lostHitsVsEta->ProfileX();
  p_ele_lostHitsVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_lostHitsVsEta->GetYaxis()->SetTitle("<N_{hits}>");
  p_ele_lostHitsVsEta->Write();
  TProfile *p_ele_lostHitsVsPhi = h_ele_lostHitsVsPhi->ProfileX();
  p_ele_lostHitsVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_lostHitsVsPhi->GetYaxis()->SetTitle("<N_{hits}>");
  p_ele_lostHitsVsPhi->Write();
  TProfile *p_ele_vertexTIPVsEta = h_ele_vertexTIPVsEta->ProfileX();
  p_ele_vertexTIPVsEta->SetTitle("mean tip (wrt gen vtx) vs eta");
  p_ele_vertexTIPVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_vertexTIPVsEta->GetYaxis()->SetTitle("<TIP> (cm)");
  p_ele_vertexTIPVsEta->Write();
  TProfile *p_ele_vertexTIPVsPhi = h_ele_vertexTIPVsPhi->ProfileX();
  p_ele_vertexTIPVsPhi->SetTitle("mean tip (wrt gen vtx) vs phi");
  p_ele_vertexTIPVsPhi->GetXaxis()->SetTitle("#phi");
  p_ele_vertexTIPVsPhi->GetYaxis()->SetTitle("<TIP> (cm)");
  p_ele_vertexTIPVsPhi->Write();
  TProfile *p_ele_vertexTIPVsPt = h_ele_vertexTIPVsPt->ProfileX();
  p_ele_vertexTIPVsPt->SetTitle("mean tip (wrt gen vtx) vs phi");
  p_ele_vertexTIPVsPt->GetXaxis()->SetTitle("p_{T} (GeV/c)");
  p_ele_vertexTIPVsPt->GetYaxis()->SetTitle("<TIP> (cm)");
  p_ele_vertexTIPVsPt->Write();

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

  h_ele_mee_all->Write();
  h_ele_mee_os->Write();
  h_ele_mee_os_ebeb->Write();
  h_ele_mee_os_ebee->Write();
  h_ele_mee_os_eeee->Write();
  h_ele_mee_os_gg->Write();
  h_ele_mee_os_gb->Write();
  h_ele_mee_os_bb->Write();
  h_ele_E2mnE1vsMee_all->Write();
  h_ele_E2mnE1vsMee_egeg_all->Write();

  // matched electrons
  h_ele_charge->Write();
  h_ele_chargeVsEta->Write();
  h_ele_chargeVsPhi->Write();
  h_ele_chargeVsPt->Write();
  h_ele_vertexP->Write();
  h_ele_vertexPt->Write();
  h_ele_Et->Write();
  h_ele_vertexPtVsEta->Write();
  h_ele_vertexPtVsPhi->Write();
  h_ele_matchingObjectPt_matched->Write();
  h_ele_vertexEta->Write();
  h_ele_vertexEtaVsPhi->Write();
  h_ele_matchingObjectAbsEta_matched->Write();
  h_ele_matchingObjectEta_matched->Write();
  h_ele_matchingObjectPhi_matched->Write();
  h_ele_vertexPhi->Write();
  h_ele_vertexX->Write();
  h_ele_vertexY->Write();
  h_ele_vertexZ->Write();
  h_ele_vertexTIP->Write();
  h_ele_matchingObjectZ_matched->Write();
  h_ele_vertexTIPVsEta->Write();
  h_ele_vertexTIPVsPhi->Write();
  h_ele_vertexTIPVsPt->Write();
  h_ele_PoPmatchingObject_matched->Write();
  h_ele_PtoPtmatchingObject_matched->Write();
  h_ele_PoPmatchingObjectVsEta_matched->Write();
  h_ele_PoPmatchingObjectVsPhi_matched->Write();
  h_ele_PoPmatchingObjectVsPt_matched->Write();
  h_ele_PoPmatchingObject_barrel_matched->Write();
  h_ele_PoPmatchingObject_endcaps_matched->Write();
  h_ele_PtoPtmatchingObject_barrel_matched->Write();
  h_ele_PtoPtmatchingObject_endcaps_matched->Write();
  h_ele_EtaMnEtamatchingObject_matched->Write();
  h_ele_EtaMnEtamatchingObjectVsEta_matched->Write();
  h_ele_EtaMnEtamatchingObjectVsPhi_matched->Write();
  h_ele_EtaMnEtamatchingObjectVsPt_matched->Write();
  h_ele_PhiMnPhimatchingObject_matched->Write();
  h_ele_PhiMnPhimatchingObject2_matched->Write();
  h_ele_PhiMnPhimatchingObjectVsEta_matched->Write();
  h_ele_PhiMnPhimatchingObjectVsPhi_matched->Write();
  h_ele_PhiMnPhimatchingObjectVsPt_matched->Write();

  // matched electron, superclusters
  histSclEn_->Write();
  histSclEoEmatchingObject_barrel_matched->Write();
  histSclEoEmatchingObject_endcaps_matched->Write();
  histSclEoEmatchingObject_barrel_new_matched->Write();
  histSclEoEmatchingObject_endcaps_new_matched->Write();
  histSclEt_->Write();
  histSclEtVsEta_->Write();
  histSclEtVsPhi_->Write();
  histSclEtaVsPhi_->Write();
  histSclEta_->Write();
  histSclPhi_->Write();
  histSclSigEtaEta_->Write();
  histSclSigIEtaIEta_barrel_->Write();
  histSclSigIEtaIEta_endcaps_->Write();
  histSclE1x5_->Write();
  histSclE1x5_barrel_->Write();
  histSclE1x5_endcaps_->Write();
  histSclE2x5max_->Write();
  histSclE2x5max_barrel_->Write();
  histSclE2x5max_endcaps_->Write();
  histSclE5x5_->Write();
  histSclE5x5_barrel_->Write();
  histSclE5x5_endcaps_->Write();

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
  h_ele_PinMnPoutVsEta_mode->Write();
  h_ele_PinMnPoutVsPhi_mode->Write();
  h_ele_PinMnPoutVsPt_mode->Write();
  h_ele_PinMnPoutVsE_mode->Write();
  h_ele_PinMnPoutVsChi2_mode->Write();
  h_ele_outerP->Write();
  h_ele_outerP_mode->Write();
  h_ele_outerPVsEta_mode->Write();
  h_ele_outerPt->Write();
  h_ele_outerPt_mode->Write();
  h_ele_outerPtVsEta_mode->Write();
  h_ele_outerPtVsPhi_mode->Write();
  h_ele_outerPtVsPt_mode->Write();

  // matched electrons, matching
  h_ele_EoP->Write();
  h_ele_EoPVsEta->Write();
  h_ele_EoPVsPhi->Write();
  h_ele_EoPVsE->Write();
  h_ele_EseedOP->Write();
  h_ele_EseedOPVsEta->Write();
  h_ele_EseedOPVsPhi->Write();
  h_ele_EseedOPVsE->Write();
  h_ele_EoPout->Write();
  h_ele_EoPoutVsEta->Write();
  h_ele_EoPoutVsPhi->Write();
  h_ele_EoPoutVsE->Write();
  h_ele_EeleOPout->Write();
  h_ele_EeleOPoutVsEta->Write();
  h_ele_EeleOPoutVsPhi->Write();
  h_ele_EeleOPoutVsE->Write();
  h_ele_dEtaSc_propVtx->Write();
  h_ele_dEtaScVsEta_propVtx->Write();
  h_ele_dEtaScVsPhi_propVtx->Write();
  h_ele_dEtaScVsPt_propVtx->Write();
  h_ele_dPhiSc_propVtx->Write();
  h_ele_dPhiScVsEta_propVtx->Write();
  h_ele_dPhiScVsPhi_propVtx->Write();
  h_ele_dPhiScVsPt_propVtx->Write();
  h_ele_dEtaCl_propOut->Write();
  h_ele_dEtaClVsEta_propOut->Write();
  h_ele_dEtaClVsPhi_propOut->Write();
  h_ele_dEtaClVsPt_propOut->Write();
  h_ele_dPhiCl_propOut->Write();
  h_ele_dPhiClVsEta_propOut->Write();
  h_ele_dPhiClVsPhi_propOut->Write();
  h_ele_dPhiClVsPt_propOut->Write();
  h_ele_dEtaEleCl_propOut->Write();
  h_ele_dEtaEleClVsEta_propOut->Write();
  h_ele_dEtaEleClVsPhi_propOut->Write();
  h_ele_dEtaEleClVsPt_propOut->Write();
  h_ele_dPhiEleCl_propOut->Write();
  h_ele_dPhiEleClVsEta_propOut->Write();
  h_ele_dPhiEleClVsPhi_propOut->Write();
  h_ele_dPhiEleClVsPt_propOut->Write();
  h_ele_HoE->Write();
  h_ele_HoE_fiducial->Write();
  h_ele_HoEVsEta->Write();
  h_ele_HoEVsPhi->Write();
  h_ele_HoEVsE->Write();

  h_ele_seed_dphi2_->Write();
  h_ele_seed_subdet2_->Write();
  TProfile *p_ele_seed_dphi2VsEta_ = h_ele_seed_dphi2VsEta_->ProfileX();
  p_ele_seed_dphi2VsEta_->SetTitle("mean ele seed dphi 2nd layer vs eta");
  p_ele_seed_dphi2VsEta_->GetXaxis()->SetTitle("#eta");
  p_ele_seed_dphi2VsEta_->GetYaxis()->SetTitle("<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)");
  p_ele_seed_dphi2VsEta_->SetMinimum(-0.004);
  p_ele_seed_dphi2VsEta_->SetMaximum(0.004);
  p_ele_seed_dphi2VsEta_->Write();
  TProfile *p_ele_seed_dphi2VsPt_ = h_ele_seed_dphi2VsPt_->ProfileX();
  p_ele_seed_dphi2VsPt_->SetTitle("mean ele seed dphi 2nd layer vs pt");
  p_ele_seed_dphi2VsPt_->GetXaxis()->SetTitle("p_{T} (GeV/c)");
  p_ele_seed_dphi2VsPt_->GetYaxis()->SetTitle("<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)");
  p_ele_seed_dphi2VsPt_->Write();
  p_ele_seed_dphi2VsPt_->SetMinimum(-0.004);
  p_ele_seed_dphi2VsPt_->SetMaximum(0.004);
  h_ele_seed_drz2_->Write();
  TProfile *p_ele_seed_drz2VsEta_ = h_ele_seed_drz2VsEta_->ProfileX();
  p_ele_seed_drz2VsEta_->SetTitle("mean ele seed dr(dz) 2nd layer vs eta");
  p_ele_seed_drz2VsEta_->GetXaxis()->SetTitle("#eta");
  p_ele_seed_drz2VsEta_->GetYaxis()->SetTitle("<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)");
  p_ele_seed_drz2VsEta_->SetMinimum(-0.15);
  p_ele_seed_drz2VsEta_->SetMaximum(0.15);
  p_ele_seed_drz2VsEta_->Write();
  TProfile *p_ele_seed_drz2VsPt_ = h_ele_seed_drz2VsPt_->ProfileX();
  p_ele_seed_drz2VsPt_->SetTitle("mean ele seed dr(dz) 2nd layer vs pt");
  p_ele_seed_drz2VsPt_->GetXaxis()->SetTitle("p_{T} (GeV/c)");
  p_ele_seed_drz2VsPt_->GetYaxis()->SetTitle("<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)");
  p_ele_seed_drz2VsPt_->SetMinimum(-0.15);
  p_ele_seed_drz2VsPt_->SetMaximum(0.15);
  p_ele_seed_drz2VsPt_->Write();

  // classes
  h_ele_classes->Write();
  h_ele_eta->Write();
  h_ele_eta_golden->Write();
  h_ele_eta_bbrem->Write();
  h_ele_eta_narrow->Write();
  h_ele_eta_shower->Write();
  h_ele_PinVsPoutGolden_mode->Write();
  h_ele_PinVsPoutShowering_mode->Write();
  h_ele_PinVsPoutGolden_mean->Write();
  h_ele_PinVsPoutShowering_mean->Write();
  h_ele_PtinVsPtoutGolden_mode->Write();
  h_ele_PtinVsPtoutShowering_mode->Write();
  h_ele_PtinVsPtoutGolden_mean->Write();
  h_ele_PtinVsPtoutShowering_mean->Write();
  histSclEoEmatchingObjectGolden_barrel->Write();
  histSclEoEmatchingObjectGolden_endcaps->Write();
  histSclEoEmatchingObjectShowering_barrel->Write();
  histSclEoEmatchingObjectShowering_endcaps->Write();

  // fbrem
  h_ele_fbrem->Write();
  h_ele_fbremVsEta_mode->Write();
  h_ele_fbremVsEta_mean->Write();
  h_ele_etaEff->Write();
  h_ele_zEff->Write();
  h_ele_phiEff->Write();
  h_ele_absetaEff->Write();
  h_ele_ptEff->Write();
  h_ele_eta_goldenFrac->Write();
  h_ele_eta_bbremFrac->Write();
  h_ele_eta_narrowFrac->Write();
  h_ele_eta_showerFrac->Write();
  h_ele_xOverX0VsEta->Write();

  // e/g et pflow electrons
  h_ele_mva->Write();
  h_ele_provenance->Write();

  // isolation
  h_ele_tkSumPt_dr03->GetXaxis()->SetTitle("TkIsoSum, cone 0.3 (GeV/c)");
  h_ele_tkSumPt_dr03->GetYaxis()->SetTitle("Events");
  h_ele_tkSumPt_dr03->Write();
  h_ele_ecalRecHitSumEt_dr03->GetXaxis()->SetTitle("EcalIsoSum, cone 0.3 (GeV)");
  h_ele_ecalRecHitSumEt_dr03->GetYaxis()->SetTitle("Events");
  h_ele_ecalRecHitSumEt_dr03->Write();
  h_ele_hcalDepth1TowerSumEt_dr03->GetXaxis()->SetTitle("Hcal1IsoSum, cone 0.3 (GeV)");
  h_ele_hcalDepth1TowerSumEt_dr03->GetYaxis()->SetTitle("Events");
  h_ele_hcalDepth1TowerSumEt_dr03->Write();
  h_ele_hcalDepth2TowerSumEt_dr03->GetXaxis()->SetTitle("Hcal2IsoSum, cone 0.3 (GeV)");
  h_ele_hcalDepth2TowerSumEt_dr03->GetYaxis()->SetTitle("Events");
  h_ele_hcalDepth2TowerSumEt_dr03->Write();
  h_ele_tkSumPt_dr04->GetXaxis()->SetTitle("TkIsoSum, cone 0.4 (GeV/c)");
  h_ele_tkSumPt_dr04->GetYaxis()->SetTitle("Events");
  h_ele_tkSumPt_dr04->Write();
  h_ele_ecalRecHitSumEt_dr04->GetXaxis()->SetTitle("EcalIsoSum, cone 0.4 (GeV)");
  h_ele_ecalRecHitSumEt_dr04->GetYaxis()->SetTitle("Events");
  h_ele_ecalRecHitSumEt_dr04->Write();
  h_ele_hcalDepth1TowerSumEt_dr04->GetXaxis()->SetTitle("Hcal1IsoSum, cone 0.4 (GeV)");
  h_ele_hcalDepth1TowerSumEt_dr04->GetYaxis()->SetTitle("Events");
  h_ele_hcalDepth1TowerSumEt_dr04->Write();
  h_ele_hcalDepth2TowerSumEt_dr04->GetXaxis()->SetTitle("Hcal2IsoSum, cone 0.4 (GeV)");
  h_ele_hcalDepth2TowerSumEt_dr04->GetYaxis()->SetTitle("Events");
  h_ele_hcalDepth2TowerSumEt_dr04->Write();
}

void GsfElectronDataAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  std::cout << "analyzing new event " << std::endl;
  nEvents_++;

  // check event pass requested triggers if any
  if (!trigger(iEvent))
    return;

  std::cout << "new event passing trigger " << std::endl;
  nAfterTrigger_++;

  // get reco electrons
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  iEvent.getByLabel(electronCollection_, gsfElectrons);
  edm::LogInfo("") << "\n\n =================> Treating event " << iEvent.id() << " Number of electrons "
                   << gsfElectrons.product()->size();

  // get reco supercluster collection
  edm::Handle<reco::SuperClusterCollection> recoClusters;
  iEvent.getByLabel(matchingObjectCollection_, recoClusters);

  // get the beamspot from the Event:
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(beamSpot_, recoBeamSpotHandle);
  const BeamSpot bs = *recoBeamSpotHandle;

  histNum_->Fill((*gsfElectrons).size());

  // selected rec electrons
  for (reco::GsfElectronCollection::const_iterator gsfIter = gsfElectrons->begin(); gsfIter != gsfElectrons->end();
       gsfIter++) {
    // select electrons
    if (gsfIter->superCluster()->energy() / cosh(gsfIter->superCluster()->eta()) < minEt_)
      continue;
    if (std::abs(gsfIter->eta()) > maxAbsEta_)
      continue;
    if (gsfIter->pt() < minPt_)
      continue;

    if (gsfIter->isEB() && isEE_)
      continue;
    if (gsfIter->isEE() && isEB_)
      continue;
    if (gsfIter->isEBEEGap() && isNotEBEEGap_)
      continue;

    if (gsfIter->ecalDrivenSeed() && isTrackerDriven_)
      continue;
    if (gsfIter->trackerDrivenSeed() && isEcalDriven_)
      continue;

    if (gsfIter->isEB() && gsfIter->eSuperClusterOverP() < eOverPMinBarrel_)
      continue;
    if (gsfIter->isEB() && gsfIter->eSuperClusterOverP() > eOverPMaxBarrel_)
      continue;
    if (gsfIter->isEE() && gsfIter->eSuperClusterOverP() < eOverPMinEndcaps_)
      continue;
    if (gsfIter->isEE() && gsfIter->eSuperClusterOverP() > eOverPMaxEndcaps_)
      continue;
    if (gsfIter->isEB() && std::abs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) < dEtaMinBarrel_)
      continue;
    if (gsfIter->isEB() && std::abs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) > dEtaMaxBarrel_)
      continue;
    if (gsfIter->isEE() && std::abs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) < dEtaMinEndcaps_)
      continue;
    if (gsfIter->isEE() && std::abs(gsfIter->deltaEtaSuperClusterTrackAtVtx()) > dEtaMaxEndcaps_)
      continue;
    if (gsfIter->isEB() && std::abs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) < dPhiMinBarrel_)
      continue;
    if (gsfIter->isEB() && std::abs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) > dPhiMaxBarrel_)
      continue;
    if (gsfIter->isEE() && std::abs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) < dPhiMinEndcaps_)
      continue;
    if (gsfIter->isEE() && std::abs(gsfIter->deltaPhiSuperClusterTrackAtVtx()) > dPhiMaxEndcaps_)
      continue;
    if (gsfIter->isEB() && gsfIter->scSigmaIEtaIEta() < sigIetaIetaMinBarrel_)
      continue;
    if (gsfIter->isEB() && gsfIter->scSigmaIEtaIEta() > sigIetaIetaMaxBarrel_)
      continue;
    if (gsfIter->isEE() && gsfIter->scSigmaIEtaIEta() < sigIetaIetaMinEndcaps_)
      continue;
    if (gsfIter->isEE() && gsfIter->scSigmaIEtaIEta() > sigIetaIetaMaxEndcaps_)
      continue;
    if (gsfIter->isEB() && gsfIter->hadronicOverEm() > hadronicOverEmMaxBarrel_)
      continue;
    if (gsfIter->isEE() && gsfIter->hadronicOverEm() > hadronicOverEmMaxEndcaps_)
      continue;
    if (gsfIter->mva_e_pi() < mvaMin_)
      continue;

    double d = (gsfIter->vertex().x() - bs.position().x()) * (gsfIter->vertex().x() - bs.position().x()) +
               (gsfIter->vertex().y() - bs.position().y()) * (gsfIter->vertex().y() - bs.position().y());
    d = sqrt(d);
    if (gsfIter->isEB() && d > tipMaxBarrel_)
      continue;
    if (gsfIter->isEE() && d > tipMaxEndcaps_)
      continue;

    if (gsfIter->dr03TkSumPt() > tkIso03Max_)
      continue;
    if (gsfIter->isEB() && gsfIter->dr03HcalDepth1TowerSumEt() > hcalIso03Depth1MaxBarrel_)
      continue;
    if (gsfIter->isEE() && gsfIter->dr03HcalDepth1TowerSumEt() > hcalIso03Depth1MaxEndcaps_)
      continue;
    if (gsfIter->isEE() && gsfIter->dr03HcalDepth2TowerSumEt() > hcalIso03Depth2MaxEndcaps_)
      continue;
    if (gsfIter->isEB() && gsfIter->dr03EcalRecHitSumEt() > ecalIso03MaxBarrel_)
      continue;
    if (gsfIter->isEE() && gsfIter->dr03EcalRecHitSumEt() > ecalIso03MaxEndcaps_)
      continue;

    // electron related distributions
    h_ele_charge->Fill(gsfIter->charge());
    h_ele_chargeVsEta->Fill(gsfIter->eta(), gsfIter->charge());
    h_ele_chargeVsPhi->Fill(gsfIter->phi(), gsfIter->charge());
    h_ele_chargeVsPt->Fill(gsfIter->pt(), gsfIter->charge());
    h_ele_vertexP->Fill(gsfIter->p());
    h_ele_vertexPt->Fill(gsfIter->pt());
    h_ele_Et->Fill(gsfIter->superCluster()->energy() / cosh(gsfIter->superCluster()->eta()));
    h_ele_vertexPtVsEta->Fill(gsfIter->eta(), gsfIter->pt());
    h_ele_vertexPtVsPhi->Fill(gsfIter->phi(), gsfIter->pt());
    h_ele_vertexEta->Fill(gsfIter->eta());

    h_ele_vertexEtaVsPhi->Fill(gsfIter->phi(), gsfIter->eta());
    h_ele_vertexPhi->Fill(gsfIter->phi());
    h_ele_vertexX->Fill(gsfIter->vertex().x());
    h_ele_vertexY->Fill(gsfIter->vertex().y());
    h_ele_vertexZ->Fill(gsfIter->vertex().z());
    h_ele_vertexTIP->Fill(d);
    h_ele_vertexTIPVsEta->Fill(gsfIter->eta(), d);
    h_ele_vertexTIPVsPhi->Fill(gsfIter->phi(), d);
    h_ele_vertexTIPVsPt->Fill(gsfIter->pt(), d);

    // supercluster related distributions
    reco::SuperClusterRef sclRef = gsfIter->superCluster();
    if (!gsfIter->ecalDrivenSeed() && gsfIter->trackerDrivenSeed())
      sclRef = gsfIter->parentSuperCluster();
    histSclEn_->Fill(sclRef->energy());
    double R = TMath::Sqrt(sclRef->x() * sclRef->x() + sclRef->y() * sclRef->y() + sclRef->z() * sclRef->z());
    double Rt = TMath::Sqrt(sclRef->x() * sclRef->x() + sclRef->y() * sclRef->y());
    histSclEt_->Fill(sclRef->energy() * (Rt / R));
    histSclEtVsEta_->Fill(sclRef->eta(), sclRef->energy() * (Rt / R));
    histSclEtVsPhi_->Fill(sclRef->phi(), sclRef->energy() * (Rt / R));
    histSclEta_->Fill(sclRef->eta());
    histSclEtaVsPhi_->Fill(sclRef->phi(), sclRef->eta());
    histSclPhi_->Fill(sclRef->phi());
    histSclSigEtaEta_->Fill(gsfIter->scSigmaEtaEta());
    if (gsfIter->isEB())
      histSclSigIEtaIEta_barrel_->Fill(gsfIter->scSigmaIEtaIEta());
    if (gsfIter->isEE())
      histSclSigIEtaIEta_endcaps_->Fill(gsfIter->scSigmaIEtaIEta());
    histSclE1x5_->Fill(gsfIter->scE1x5());
    if (gsfIter->isEB())
      histSclE1x5_barrel_->Fill(gsfIter->scE1x5());
    if (gsfIter->isEE())
      histSclE1x5_endcaps_->Fill(gsfIter->scE1x5());
    histSclE2x5max_->Fill(gsfIter->scE2x5Max());
    if (gsfIter->isEB())
      histSclE2x5max_barrel_->Fill(gsfIter->scE2x5Max());
    if (gsfIter->isEE())
      histSclE2x5max_endcaps_->Fill(gsfIter->scE2x5Max());
    histSclE5x5_->Fill(gsfIter->scE5x5());
    if (gsfIter->isEB())
      histSclE5x5_barrel_->Fill(gsfIter->scE5x5());
    if (gsfIter->isEE())
      histSclE5x5_endcaps_->Fill(gsfIter->scE5x5());

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
    h_ele_PinMnPoutVsEta_mode->Fill(gsfIter->eta(),
                                    gsfIter->trackMomentumAtVtx().R() - gsfIter->trackMomentumOut().R());
    h_ele_PinMnPoutVsPhi_mode->Fill(gsfIter->phi(),
                                    gsfIter->trackMomentumAtVtx().R() - gsfIter->trackMomentumOut().R());
    h_ele_PinMnPoutVsPt_mode->Fill(gsfIter->pt(), gsfIter->trackMomentumAtVtx().R() - gsfIter->trackMomentumOut().R());
    h_ele_PinMnPoutVsE_mode->Fill(gsfIter->caloEnergy(),
                                  gsfIter->trackMomentumAtVtx().R() - gsfIter->trackMomentumOut().R());
    if (!readAOD_)  // track extra does not exist in AOD
      h_ele_PinMnPoutVsChi2_mode->Fill(gsfIter->gsfTrack()->normalizedChi2(),
                                       gsfIter->trackMomentumAtVtx().R() - gsfIter->trackMomentumOut().R());
    h_ele_outerP_mode->Fill(gsfIter->trackMomentumOut().R());
    h_ele_outerPVsEta_mode->Fill(gsfIter->eta(), gsfIter->trackMomentumOut().R());
    h_ele_outerPt_mode->Fill(gsfIter->trackMomentumOut().Rho());
    h_ele_outerPtVsEta_mode->Fill(gsfIter->eta(), gsfIter->trackMomentumOut().Rho());
    h_ele_outerPtVsPhi_mode->Fill(gsfIter->phi(), gsfIter->trackMomentumOut().Rho());
    h_ele_outerPtVsPt_mode->Fill(gsfIter->pt(), gsfIter->trackMomentumOut().Rho());

    if (!readAOD_) {  // track extra does not exist in AOD
      edm::RefToBase<TrajectorySeed> seed = gsfIter->gsfTrack()->extra()->seedRef();
      ElectronSeedRef elseed = seed.castTo<ElectronSeedRef>();
      h_ele_seed_dphi2_->Fill(elseed->dPhiNeg(1));
      h_ele_seed_dphi2VsEta_->Fill(gsfIter->eta(), elseed->dPhiNeg(1));
      h_ele_seed_dphi2VsPt_->Fill(gsfIter->pt(), elseed->dPhiNeg(1));
      h_ele_seed_drz2_->Fill(elseed->dRZNeg(1));
      h_ele_seed_drz2VsEta_->Fill(gsfIter->eta(), elseed->dRZNeg(1));
      h_ele_seed_drz2VsPt_->Fill(gsfIter->pt(), elseed->dRZNeg(1));
      h_ele_seed_subdet2_->Fill(elseed->subDet(1));
    }
    // match distributions
    h_ele_EoP->Fill(gsfIter->eSuperClusterOverP());
    h_ele_EoPVsEta->Fill(gsfIter->eta(), gsfIter->eSuperClusterOverP());
    h_ele_EoPVsPhi->Fill(gsfIter->phi(), gsfIter->eSuperClusterOverP());
    h_ele_EoPVsE->Fill(gsfIter->caloEnergy(), gsfIter->eSuperClusterOverP());
    h_ele_EseedOP->Fill(gsfIter->eSeedClusterOverP());
    h_ele_EseedOPVsEta->Fill(gsfIter->eta(), gsfIter->eSeedClusterOverP());
    h_ele_EseedOPVsPhi->Fill(gsfIter->phi(), gsfIter->eSeedClusterOverP());
    h_ele_EseedOPVsE->Fill(gsfIter->caloEnergy(), gsfIter->eSeedClusterOverP());
    h_ele_EoPout->Fill(gsfIter->eSeedClusterOverPout());
    h_ele_EoPoutVsEta->Fill(gsfIter->eta(), gsfIter->eSeedClusterOverPout());
    h_ele_EoPoutVsPhi->Fill(gsfIter->phi(), gsfIter->eSeedClusterOverPout());
    h_ele_EoPoutVsE->Fill(gsfIter->caloEnergy(), gsfIter->eSeedClusterOverPout());
    h_ele_EeleOPout->Fill(gsfIter->eEleClusterOverPout());
    h_ele_EeleOPoutVsEta->Fill(gsfIter->eta(), gsfIter->eEleClusterOverPout());
    h_ele_EeleOPoutVsPhi->Fill(gsfIter->phi(), gsfIter->eEleClusterOverPout());
    h_ele_EeleOPoutVsE->Fill(gsfIter->caloEnergy(), gsfIter->eEleClusterOverPout());
    h_ele_dEtaSc_propVtx->Fill(gsfIter->deltaEtaSuperClusterTrackAtVtx());
    h_ele_dEtaScVsEta_propVtx->Fill(gsfIter->eta(), gsfIter->deltaEtaSuperClusterTrackAtVtx());
    h_ele_dEtaScVsPhi_propVtx->Fill(gsfIter->phi(), gsfIter->deltaEtaSuperClusterTrackAtVtx());
    h_ele_dEtaScVsPt_propVtx->Fill(gsfIter->pt(), gsfIter->deltaEtaSuperClusterTrackAtVtx());
    h_ele_dPhiSc_propVtx->Fill(gsfIter->deltaPhiSuperClusterTrackAtVtx());
    h_ele_dPhiScVsEta_propVtx->Fill(gsfIter->eta(), gsfIter->deltaPhiSuperClusterTrackAtVtx());
    h_ele_dPhiScVsPhi_propVtx->Fill(gsfIter->phi(), gsfIter->deltaPhiSuperClusterTrackAtVtx());
    h_ele_dPhiScVsPt_propVtx->Fill(gsfIter->pt(), gsfIter->deltaPhiSuperClusterTrackAtVtx());
    h_ele_dEtaCl_propOut->Fill(gsfIter->deltaEtaSeedClusterTrackAtCalo());
    h_ele_dEtaClVsEta_propOut->Fill(gsfIter->eta(), gsfIter->deltaEtaSeedClusterTrackAtCalo());
    h_ele_dEtaClVsPhi_propOut->Fill(gsfIter->phi(), gsfIter->deltaEtaSeedClusterTrackAtCalo());
    h_ele_dEtaClVsPt_propOut->Fill(gsfIter->pt(), gsfIter->deltaEtaSeedClusterTrackAtCalo());
    h_ele_dPhiCl_propOut->Fill(gsfIter->deltaPhiSeedClusterTrackAtCalo());
    h_ele_dPhiClVsEta_propOut->Fill(gsfIter->eta(), gsfIter->deltaPhiSeedClusterTrackAtCalo());
    h_ele_dPhiClVsPhi_propOut->Fill(gsfIter->phi(), gsfIter->deltaPhiSeedClusterTrackAtCalo());
    h_ele_dPhiClVsPt_propOut->Fill(gsfIter->pt(), gsfIter->deltaPhiSeedClusterTrackAtCalo());
    h_ele_dEtaEleCl_propOut->Fill(gsfIter->deltaEtaEleClusterTrackAtCalo());
    h_ele_dEtaEleClVsEta_propOut->Fill(gsfIter->eta(), gsfIter->deltaEtaEleClusterTrackAtCalo());
    h_ele_dEtaEleClVsPhi_propOut->Fill(gsfIter->phi(), gsfIter->deltaEtaEleClusterTrackAtCalo());
    h_ele_dEtaEleClVsPt_propOut->Fill(gsfIter->pt(), gsfIter->deltaEtaEleClusterTrackAtCalo());
    h_ele_dPhiEleCl_propOut->Fill(gsfIter->deltaPhiEleClusterTrackAtCalo());
    h_ele_dPhiEleClVsEta_propOut->Fill(gsfIter->eta(), gsfIter->deltaPhiEleClusterTrackAtCalo());
    h_ele_dPhiEleClVsPhi_propOut->Fill(gsfIter->phi(), gsfIter->deltaPhiEleClusterTrackAtCalo());
    h_ele_dPhiEleClVsPt_propOut->Fill(gsfIter->pt(), gsfIter->deltaPhiEleClusterTrackAtCalo());
    h_ele_HoE->Fill(gsfIter->hadronicOverEm());
    if (!gsfIter->isEBEtaGap() && !gsfIter->isEBPhiGap() && !gsfIter->isEBEEGap() && !gsfIter->isEERingGap() &&
        !gsfIter->isEEDeeGap())
      h_ele_HoE_fiducial->Fill(gsfIter->hadronicOverEm());
    h_ele_HoEVsEta->Fill(gsfIter->eta(), gsfIter->hadronicOverEm());
    h_ele_HoEVsPhi->Fill(gsfIter->phi(), gsfIter->hadronicOverEm());
    h_ele_HoEVsE->Fill(gsfIter->caloEnergy(), gsfIter->hadronicOverEm());

    //classes
    int eleClass = gsfIter->classification();
    if (gsfIter->isEE())
      eleClass += 10;
    h_ele_classes->Fill(eleClass);

    h_ele_eta->Fill(std::abs(gsfIter->eta()));
    if (gsfIter->classification() == GsfElectron::GOLDEN)
      h_ele_eta_golden->Fill(std::abs(gsfIter->eta()));
    if (gsfIter->classification() == GsfElectron::BIGBREM)
      h_ele_eta_bbrem->Fill(std::abs(gsfIter->eta()));
    //if (gsfIter->classification() == GsfElectron::OLDNARROW) h_ele_eta_narrow ->Fill(std::abs(gsfIter->eta()));
    if (gsfIter->classification() == GsfElectron::SHOWERING)
      h_ele_eta_shower->Fill(std::abs(gsfIter->eta()));

    //fbrem
    double fbrem_mean = 0.;
    if (!readAOD_)  // track extra does not exist in AOD
      fbrem_mean = 1. - gsfIter->gsfTrack()->outerMomentum().R() / gsfIter->gsfTrack()->innerMomentum().R();
    double fbrem_mode = gsfIter->fbrem();
    h_ele_fbrem->Fill(fbrem_mode);
    h_ele_fbremVsEta_mode->Fill(gsfIter->eta(), fbrem_mode);
    if (!readAOD_)  // track extra does not exist in AOD
      h_ele_fbremVsEta_mean->Fill(gsfIter->eta(), fbrem_mean);

    if (gsfIter->classification() == GsfElectron::GOLDEN)
      h_ele_PinVsPoutGolden_mode->Fill(gsfIter->trackMomentumOut().R(), gsfIter->trackMomentumAtVtx().R());
    if (gsfIter->classification() == GsfElectron::SHOWERING)
      h_ele_PinVsPoutShowering_mode->Fill(gsfIter->trackMomentumOut().R(), gsfIter->trackMomentumAtVtx().R());
    if (!readAOD_)  // track extra does not exist in AOD
      if (gsfIter->classification() == GsfElectron::GOLDEN)
        h_ele_PinVsPoutGolden_mean->Fill(gsfIter->gsfTrack()->outerMomentum().R(),
                                         gsfIter->gsfTrack()->innerMomentum().R());
    if (!readAOD_)  // track extra does not exist in AOD
      if (gsfIter->classification() == GsfElectron::SHOWERING)
        h_ele_PinVsPoutShowering_mean->Fill(gsfIter->gsfTrack()->outerMomentum().R(),
                                            gsfIter->gsfTrack()->innerMomentum().R());
    if (gsfIter->classification() == GsfElectron::GOLDEN)
      h_ele_PtinVsPtoutGolden_mode->Fill(gsfIter->trackMomentumOut().Rho(), gsfIter->trackMomentumAtVtx().Rho());
    if (gsfIter->classification() == GsfElectron::SHOWERING)
      h_ele_PtinVsPtoutShowering_mode->Fill(gsfIter->trackMomentumOut().Rho(), gsfIter->trackMomentumAtVtx().Rho());
    if (!readAOD_)  // track extra does not exist in AOD
      if (gsfIter->classification() == GsfElectron::GOLDEN)
        h_ele_PtinVsPtoutGolden_mean->Fill(gsfIter->gsfTrack()->outerMomentum().Rho(),
                                           gsfIter->gsfTrack()->innerMomentum().Rho());
    if (!readAOD_)  // track extra does not exist in AOD
      if (gsfIter->classification() == GsfElectron::SHOWERING)
        h_ele_PtinVsPtoutShowering_mean->Fill(gsfIter->gsfTrack()->outerMomentum().Rho(),
                                              gsfIter->gsfTrack()->innerMomentum().Rho());

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

    float enrj1 = gsfIter->superCluster()->energy();
    // mee
    for (reco::GsfElectronCollection::const_iterator gsfIter2 = gsfIter + 1; gsfIter2 != gsfElectrons->end();
         gsfIter2++) {
      math::XYZTLorentzVector p12 = (*gsfIter).p4() + (*gsfIter2).p4();
      float mee2 = p12.Dot(p12);
      float enrj2 = gsfIter2->superCluster()->energy();
      h_ele_mee_all->Fill(sqrt(mee2));
      h_ele_E2mnE1vsMee_all->Fill(sqrt(mee2), enrj2 - enrj1);
      if (gsfIter->ecalDrivenSeed() && gsfIter2->ecalDrivenSeed())
        h_ele_E2mnE1vsMee_egeg_all->Fill(sqrt(mee2), enrj2 - enrj1);
      if (gsfIter->charge() * gsfIter2->charge() < 0.) {
        h_ele_mee_os->Fill(sqrt(mee2));
        if (gsfIter->isEB() && gsfIter2->isEB())
          h_ele_mee_os_ebeb->Fill(sqrt(mee2));
        if ((gsfIter->isEB() && gsfIter2->isEE()) || (gsfIter->isEE() && gsfIter2->isEB()))
          h_ele_mee_os_ebee->Fill(sqrt(mee2));
        if (gsfIter->isEE() && gsfIter2->isEE())
          h_ele_mee_os_eeee->Fill(sqrt(mee2));
        if ((gsfIter->classification()==GsfElectron::GOLDEN && gsfIter2->classification()==GsfElectron::GOLDEN) ||
		 (gsfIter->classification()==GsfElectron::GOLDEN && gsfIter2->classification()==GsfElectron::BIGBREM) ||
		 //(gsfIter->classification()==GsfElectron::GOLDEN && gsfIter2->classification()==GsfElectron::OLDNARROW) ||
		 (gsfIter->classification()==GsfElectron::BIGBREM && gsfIter2->classification()==GsfElectron::GOLDEN) ||
		 (gsfIter->classification()==GsfElectron::BIGBREM && gsfIter2->classification()==GsfElectron::BIGBREM)/* ||
		 (gsfIter->classification()==GsfElectron::BIGBREM && gsfIter2->classification()==GsfElectron::OLDNARROW) ||
		 (gsfIter->classification()==GsfElectron::OLDNARROW && gsfIter2->classification()==GsfElectron::GOLDEN) ||
		 (gsfIter->classification()==GsfElectron::OLDNARROW && gsfIter2->classification()==GsfElectron::BIGBREM) ||
		 (gsfIter->classification()==GsfElectron::OLDNARROW && gsfIter2->classification()==GsfElectron::OLDNARROW)*/)
	       {
          h_ele_mee_os_gg->Fill(sqrt(mee2));
        } else if ((gsfIter->classification() == GsfElectron::SHOWERING &&
                    gsfIter2->classification() == GsfElectron::SHOWERING) ||
                   (gsfIter->classification() == GsfElectron::SHOWERING && gsfIter2->isGap()) ||
                   (gsfIter->isGap() && gsfIter2->classification() == GsfElectron::SHOWERING) ||
                   (gsfIter->isGap() && gsfIter2->isGap())) {
          h_ele_mee_os_bb->Fill(sqrt(mee2));
        } else {
          h_ele_mee_os_gb->Fill(sqrt(mee2));
        }
      }
    }
  }

  // association matching object-reco electrons
  int matchingObjectNum = 0;

  for (reco::SuperClusterCollection::const_iterator moIter = recoClusters->begin(); moIter != recoClusters->end();
       moIter++) {
    // number of matching objects
    matchingObjectNum++;

    if (moIter->energy() / cosh(moIter->eta()) > maxPtMatchingObject_ ||
        std::abs(moIter->eta()) > maxAbsEtaMatchingObject_)
      continue;

    // suppress the endcaps
    //if (std::abs(moIter->eta()) > 1.5) continue;
    // select central z
    //if ( std::abs((*mcIter)->production_vertex()->position().z())>50.) continue;

    h_matchingObjectEta->Fill(moIter->eta());
    h_matchingObjectAbsEta->Fill(std::abs(moIter->eta()));
    h_matchingObjectP->Fill(moIter->energy());
    h_matchingObjectPt->Fill(moIter->energy() / cosh(moIter->eta()));
    h_matchingObjectPhi->Fill(moIter->phi());
    h_matchingObjectZ->Fill(moIter->z());

    // looking for the best matching gsf electron
    bool okGsfFound = false;
    double gsfOkRatio = 999999.;

    // find matching electron
    reco::GsfElectron bestGsfElectron;
    for (reco::GsfElectronCollection::const_iterator gsfIter = gsfElectrons->begin(); gsfIter != gsfElectrons->end();
         gsfIter++) {
      // matching with a cone in eta phi
      if (matchingCondition_ == "Cone") {
        double dphi = gsfIter->phi() - moIter->phi();
        if (std::abs(dphi) > CLHEP::pi)
          dphi = dphi < 0 ? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
        double deltaR = sqrt(std::pow((moIter->eta() - gsfIter->eta()), 2) + std::pow(dphi, 2));
        if (deltaR < deltaR_) {
          //if ( (genPc->pdg_id() == 11) && (gsfIter->charge() < 0.) || (genPc->pdg_id() == -11) &&
          //(gsfIter->charge() > 0.) ){
          double tmpGsfRatio = gsfIter->p() / moIter->energy();
          if (std::abs(tmpGsfRatio - 1) < std::abs(gsfOkRatio - 1)) {
            gsfOkRatio = tmpGsfRatio;
            bestGsfElectron = *gsfIter;
            okGsfFound = true;
          }
          //}
        }
      }
    }  // loop over rec ele to look for the best one

    // analysis when the matching object is matched by a rec electron
    if (okGsfFound) {
      // generated distributions for matched electrons
      h_ele_matchingObjectPt_matched->Fill(moIter->energy() / cosh(moIter->eta()));
      h_ele_matchingObjectPhi_matched->Fill(moIter->phi());
      h_ele_matchingObjectAbsEta_matched->Fill(std::abs(moIter->eta()));
      h_ele_matchingObjectEta_matched->Fill(moIter->eta());
      h_ele_matchingObjectZ_matched->Fill(moIter->z());

      // comparison electron vs matching object
      h_ele_EtaMnEtamatchingObject_matched->Fill(bestGsfElectron.eta() - moIter->eta());
      h_ele_EtaMnEtamatchingObjectVsEta_matched->Fill(bestGsfElectron.eta(), bestGsfElectron.eta() - moIter->eta());
      h_ele_EtaMnEtamatchingObjectVsPhi_matched->Fill(bestGsfElectron.phi(), bestGsfElectron.eta() - moIter->eta());
      h_ele_EtaMnEtamatchingObjectVsPt_matched->Fill(bestGsfElectron.pt(), bestGsfElectron.eta() - moIter->eta());
      h_ele_PhiMnPhimatchingObject_matched->Fill(bestGsfElectron.phi() - moIter->phi());
      h_ele_PhiMnPhimatchingObject2_matched->Fill(bestGsfElectron.phi() - moIter->phi());
      h_ele_PhiMnPhimatchingObjectVsEta_matched->Fill(bestGsfElectron.eta(), bestGsfElectron.phi() - moIter->phi());
      h_ele_PhiMnPhimatchingObjectVsPhi_matched->Fill(bestGsfElectron.phi(), bestGsfElectron.phi() - moIter->phi());
      h_ele_PhiMnPhimatchingObjectVsPt_matched->Fill(bestGsfElectron.pt(), bestGsfElectron.phi() - moIter->phi());
      h_ele_PoPmatchingObject_matched->Fill(bestGsfElectron.p() / moIter->energy());
      h_ele_PtoPtmatchingObject_matched->Fill(bestGsfElectron.pt() / moIter->energy() / cosh(moIter->eta()));
      h_ele_PoPmatchingObjectVsEta_matched->Fill(bestGsfElectron.eta(), bestGsfElectron.p() / moIter->energy());
      h_ele_PoPmatchingObjectVsPhi_matched->Fill(bestGsfElectron.phi(), bestGsfElectron.p() / moIter->energy());
      h_ele_PoPmatchingObjectVsPt_matched->Fill(bestGsfElectron.py(), bestGsfElectron.p() / moIter->energy());
      if (bestGsfElectron.isEB())
        h_ele_PoPmatchingObject_barrel_matched->Fill(bestGsfElectron.p() / moIter->energy());
      if (bestGsfElectron.isEE())
        h_ele_PoPmatchingObject_endcaps_matched->Fill(bestGsfElectron.p() / moIter->energy());
      if (bestGsfElectron.isEB())
        h_ele_PtoPtmatchingObject_barrel_matched->Fill(bestGsfElectron.pt() / moIter->energy() / cosh(moIter->eta()));
      if (bestGsfElectron.isEE())
        h_ele_PtoPtmatchingObject_endcaps_matched->Fill(bestGsfElectron.pt() / moIter->energy() / cosh(moIter->eta()));

      reco::SuperClusterRef sclRef = bestGsfElectron.superCluster();
      if (bestGsfElectron.isEB())
        histSclEoEmatchingObject_barrel_matched->Fill(sclRef->energy() / moIter->energy());
      if (bestGsfElectron.isEE())
        histSclEoEmatchingObject_endcaps_matched->Fill(sclRef->energy() / moIter->energy());
      if (bestGsfElectron.isEB())
        histSclEoEmatchingObject_barrel_new_matched->Fill(sclRef->energy() / moIter->energy());
      if (bestGsfElectron.isEE())
        histSclEoEmatchingObject_endcaps_new_matched->Fill(sclRef->energy() / moIter->energy());

      // add here distributions for matched electrons as for all electrons
      //..

    }  // gsf electron found

  }  // loop overmatching object

  h_matchingObjectNum->Fill(matchingObjectNum);
}

bool GsfElectronDataAnalyzer::trigger(const edm::Event &e) {
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
        std::cout << "trigger path= " << triggerNames.triggerName(i) << std::endl;
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

    if (nEvents_ == 1) {
      if (n > 0) {
        std::cout << "HLT trigger paths requested: index, name and valididty:" << std::endl;
        for (unsigned int i = 0; i != n; i++) {
          bool validity = HLTPathsByIndex_[i] < triggerResults->size();
          std::cout << " " << HLTPathsByIndex_[i] << " " << HLTPathsByName_[i] << " " << validity << std::endl;
        }
      }
    }

    // count number of requested HLT paths which have fired
    unsigned int fired = 0;
    for (unsigned int i = 0; i != n; i++) {
      if (HLTPathsByIndex_[i] < triggerResults->size()) {
        if (triggerResults->accept(HLTPathsByIndex_[i])) {
          fired++;
          std::cout << "Fired HLT path= " << HLTPathsByName_[i] << std::endl;
          accept = true;
        }
      }
    }
  }

  return accept;
}
