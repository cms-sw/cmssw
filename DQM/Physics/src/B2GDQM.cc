#include "DQM/Physics/src/B2GDQM.h"

#include <memory>

// DQM
#include "DQMServices/Core/interface/DQMStore.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Provenance/interface/EventID.h"

// Candidate handling
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

// Vertex utilities
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// Other
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/RefToBase.h"

// Math
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

// vertexing

// Transient tracks
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// ROOT
#include "TLorentzVector.h"

// STDLIB
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>

using namespace edm;
using namespace std;
using namespace reco;
using namespace trigger;

//
// -- Constructor
//
B2GDQM::B2GDQM(const edm::ParameterSet& ps) {
  edm::LogInfo("B2GDQM") << " Starting B2GDQM "
                         << "\n";

  typedef std::vector<edm::InputTag> vtag;

  // Get parameters from configuration file
  // Trigger
  theTriggerResultsCollection = ps.getParameter<InputTag>("triggerResultsCollection");
  triggerToken_ = consumes<edm::TriggerResults>(theTriggerResultsCollection);

  // Jets
  jetLabels_ = ps.getParameter<std::vector<edm::InputTag> >("jetLabels");
  for (std::vector<edm::InputTag>::const_iterator jetlabel = jetLabels_.begin(), jetlabelEnd = jetLabels_.end();
       jetlabel != jetlabelEnd;
       ++jetlabel) {
    jetTokens_.push_back(consumes<edm::View<reco::Jet> >(*jetlabel));
  }
  sdjetLabel_ = ps.getParameter<edm::InputTag>("sdjetLabel");
  sdjetToken_ = consumes<edm::View<reco::BasicJet> >(sdjetLabel_);

  muonToken_ = consumes<edm::View<reco::Muon> >(ps.getParameter<edm::InputTag>("muonSrc"));
  electronToken_ = consumes<edm::View<reco::GsfElectron> >(ps.getParameter<edm::InputTag>("electronSrc"));

  jetPtMins_ = ps.getParameter<std::vector<double> >("jetPtMins");
  allHadPtCut_ = ps.getParameter<double>("allHadPtCut");
  allHadRapidityCut_ = ps.getParameter<double>("allHadRapidityCut");
  allHadDeltaPhiCut_ = ps.getParameter<double>("allHadDeltaPhiCut");

  semiMu_HadJetPtCut_ = ps.getParameter<double>("semiMu_HadJetPtCut");
  semiMu_LepJetPtCut_ = ps.getParameter<double>("semiMu_LepJetPtCut");
  semiMu_dphiHadCut_ = ps.getParameter<double>("semiMu_dphiHadCut");
  semiMu_dRMin_ = ps.getParameter<double>("semiMu_dRMin");
  semiMu_ptRel_ = ps.getParameter<double>("semiMu_ptRel");
  muonSelect_ = std::make_shared<StringCutObjectSelector<reco::Muon> >(

      ps.getParameter<std::string>("muonSelect"));

  semiE_HadJetPtCut_ = ps.getParameter<double>("semiE_HadJetPtCut");
  semiE_LepJetPtCut_ = ps.getParameter<double>("semiE_LepJetPtCut");
  semiE_dphiHadCut_ = ps.getParameter<double>("semiE_dphiHadCut");
  semiE_dRMin_ = ps.getParameter<double>("semiE_dRMin");
  semiE_ptRel_ = ps.getParameter<double>("semiE_ptRel");
  elecSelect_ = std::make_shared<StringCutObjectSelector<reco::GsfElectron> >(

      ps.getParameter<std::string>("elecSelect"));

  PFJetCorService_ = ps.getParameter<std::string>("PFJetCorService");

  // MET
  PFMETLabel_ = ps.getParameter<InputTag>("pfMETCollection");
  PFMETToken_ = consumes<std::vector<reco::PFMET> >(PFMETLabel_);
}

//
// -- Destructor
//
B2GDQM::~B2GDQM() {
  edm::LogInfo("B2GDQM") << " Deleting B2GDQM "
                         << "\n";
}

//
//  -- Book histograms
//
void B2GDQM::bookHistograms(DQMStore::IBooker& bei, edm::Run const&, edm::EventSetup const&) {
  bei.setCurrentFolder("Physics/B2G");

  //--- Jets

  for (unsigned int icoll = 0; icoll < jetLabels_.size(); ++icoll) {
    std::stringstream ss;
    ss << "Physics/B2G/" << jetLabels_[icoll].label();
    bei.setCurrentFolder(ss.str());
    pfJet_pt.push_back(bei.book1D("pfJet_pt", "Pt of PFJet (GeV)", 50, 0.0, 1000));
    pfJet_y.push_back(bei.book1D("pfJet_y", "Rapidity of PFJet", 60, -6.0, 6.0));
    pfJet_phi.push_back(bei.book1D("pfJet_phi", "#phi of PFJet (radians)", 60, -3.14159, 3.14159));
    pfJet_m.push_back(bei.book1D("pfJet_m", "Mass of PFJet (GeV)", 50, 0.0, 500));
    pfJet_chef.push_back(bei.book1D("pfJet_pfchef", "PFJetID CHEF", 50, 0.0, 1.0));
    pfJet_nhef.push_back(bei.book1D("pfJet_pfnhef", "PFJetID NHEF", 50, 0.0, 1.0));
    pfJet_cemf.push_back(bei.book1D("pfJet_pfcemf", "PFJetID CEMF", 50, 0.0, 1.0));
    pfJet_nemf.push_back(bei.book1D("pfJet_pfnemf", "PFJetID NEMF", 50, 0.0, 1.0));

    boostedJet_subjetPt.push_back(bei.book1D("boostedJet_subjetPt", "Pt of subjets (GeV)", 50, 0.0, 500));
    boostedJet_subjetY.push_back(bei.book1D("boostedJet_subjetY", "Rapidity of subjets", 60, -6.0, 6.0));
    boostedJet_subjetPhi.push_back(
        bei.book1D("boostedJet_subjetPhi", "#phi of subjets (radians)", 60, -3.14159, 3.14159));
    boostedJet_subjetM.push_back(bei.book1D("boostedJet_subjetM", "Mass of subjets (GeV)", 50, 0.0, 250.));
    boostedJet_subjetN.push_back(bei.book1D("boostedJet_subjetN", "Number of subjets", 10, 0, 10));
    boostedJet_massDrop.push_back(bei.book1D("boostedJet_massDrop", "Mass drop for W-like jets", 50, 0.0, 1.0));
    boostedJet_wMass.push_back(bei.book1D("boostedJet_wMass", "W Mass for top-like jets", 50, 0.0, 250.0));
  }

  bei.setCurrentFolder("Physics/B2G/MET");
  pfMet_pt = bei.book1D("pfMet_pt", "Pf Missing p_{T}; GeV", 50, 0.0, 500);
  pfMet_phi = bei.book1D("pfMet_phi", "Pf Missing p_{T} #phi;#phi (radians)", 35, -3.5, 3.5);

  //--- Mu+Jets
  bei.setCurrentFolder("Physics/B2G/SemiMu");
  semiMu_muPt = bei.book1D("semiMu_muPt", "Pt of Muon in #mu+Jets Channel (GeV)", 50, 0.0, 1000);
  semiMu_muEta = bei.book1D("semiMu_muEta", "#eta of Muon in #mu+Jets Channel", 60, -6.0, 6.0);
  semiMu_muPhi = bei.book1D("semiMu_muPhi", "#phi of Muon in #mu+Jets Channel (radians)", 60, -3.14159, 3.14159);
  semiMu_muDRMin = bei.book1D("semiMu_muDRMin", "#Delta R(E,nearest jet) in #mu+Jets Channel", 50, 0, 10.0);
  semiMu_muPtRel = bei.book1D("semiMu_muPtRel", "p_{T}^{REL} in #mu+Jets Channel", 60, 0, 300.);
  semiMu_hadJetDR = bei.book1D("semiMu_hadJetDR", "#Delta R(E,had jet) in #mu+Jets Channel", 50, 0, 10.0);
  semiMu_hadJetPt =
      bei.book1D("semiMu_hadJetPt", "Pt of Leading Hadronic Jet in #mu+Jets Channel (GeV)", 50, 0.0, 1000);
  semiMu_hadJetY = bei.book1D("semiMu_hadJetY", "Rapidity of Leading Hadronic Jet in #mu+Jets Channel", 60, -6.0, 6.0);
  semiMu_hadJetPhi = bei.book1D(
      "semiMu_hadJetPhi", "#phi of Leading Hadronic Jet in #mu+Jets Channel (radians)", 60, -3.14159, 3.14159);
  semiMu_hadJetMass =
      bei.book1D("semiMu_hadJetMass", "Mass of Leading Hadronic Jet in #mu+Jets Channel (GeV)", 50, 0.0, 500);
  semiMu_hadJetWMass =
      bei.book1D("semiMu_hadJetwMass", "W Mass for Leading Hadronic Jet in #mu+Jets Channel (GeV)", 50, 0.0, 250.0);
  semiMu_mttbar = bei.book1D("semiMu_mttbar", "Mass of #mu+Jets ttbar Candidate", 100, 0., 5000.);

  //--- E+Jets
  bei.setCurrentFolder("Physics/B2G/SemiE");
  semiE_ePt = bei.book1D("semiE_ePt", "Pt of Electron in e+Jets Channel (GeV)", 50, 0.0, 1000);
  semiE_eEta = bei.book1D("semiE_eEta", "#eta of Electron in e+Jets Channel", 60, -6.0, 6.0);
  semiE_ePhi = bei.book1D("semiE_ePhi", "#phi of Electron in e+Jets Channel (radians)", 60, -3.14159, 3.14159);
  semiE_eDRMin = bei.book1D("semiE_eDRMin", "#Delta R(E,nearest jet) in e+Jets Channel", 50, 0, 10.0);
  semiE_ePtRel = bei.book1D("semiE_ePtRel", "p_{T}^{REL} in e+Jets Channel", 60, 0, 300.);
  semiE_hadJetDR = bei.book1D("semiE_hadJetDR", "#Delta R(E,had jet) in e+Jets Channel", 50, 0, 10.0);
  semiE_hadJetPt = bei.book1D("semiE_hadJetPt", "Pt of Leading Hadronic Jet in e+Jets Channel (GeV)", 50, 0.0, 1000);
  semiE_hadJetY = bei.book1D("semiE_hadJetY", "Rapidity of Leading Hadronic Jet in e+Jets Channel", 60, -6.0, 6.0);
  semiE_hadJetPhi =
      bei.book1D("semiE_hadJetPhi", "#phi of Leading Hadronic Jet in e+Jets Channel (radians)", 60, -3.14159, 3.14159);
  semiE_hadJetMass =
      bei.book1D("semiE_hadJetMass", "Mass of Leading Hadronic Jet in e+Jets Channel (GeV)", 50, 0.0, 500);
  semiE_hadJetWMass =
      bei.book1D("semiE_hadJetwMass", "W Mass for Leading Hadronic Jet in e+Jets Channel (GeV)", 50, 0.0, 250.0);
  semiE_mttbar = bei.book1D("semiE_mttbar", "Mass of e+Jets ttbar Candidate", 100, 0., 5000.);

  //--- All-hadronic
  bei.setCurrentFolder("Physics/B2G/AllHad");
  allHad_pt0 = bei.book1D("allHad_pt0", "Pt of Leading All-Hadronic PFJet (GeV)", 50, 0.0, 1000);
  allHad_y0 = bei.book1D("allHad_y0", "Rapidity of Leading All-Hadronic PFJet", 60, -6.0, 6.0);
  allHad_phi0 = bei.book1D("allHad_phi0", "#phi of Leading All-Hadronic PFJet (radians)", 60, -3.14159, 3.14159);
  allHad_mass0 = bei.book1D("allHad_mass0", "Mass of Leading All-Hadronic PFJet (GeV)", 50, 0.0, 500);
  allHad_wMass0 = bei.book1D("allHad_wMass0", "W Mass for Leading All-Hadronic PFJet (GeV)", 50, 0.0, 250.0);
  allHad_pt1 = bei.book1D("allHad_pt1", "Pt of Subleading All-Hadronic PFJet (GeV)", 50, 0.0, 1000);
  allHad_y1 = bei.book1D("allHad_y1", "Rapidity of Subleading All-Hadronic PFJet", 60, -6.0, 6.0);
  allHad_phi1 = bei.book1D("allHad_phi1", "#phi of Subleading All-Hadronic PFJet (radians)", 60, -3.14159, 3.14159);
  allHad_mass1 = bei.book1D("allHad_mass1", "Mass of Subleading All-Hadronic PFJet (GeV)", 50, 0.0, 500);
  allHad_wMass1 = bei.book1D("allHad_wMass1", "W Mass for Subleading All-Hadronic PFJet (GeV)", 50, 0.0, 250.0);
  allHad_mttbar = bei.book1D("allHad_mttbar", "Mass of All-Hadronic ttbar Candidate", 100, 0., 5000.);
}

//
//  -- Analyze
//
void B2GDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  analyzeJets(iEvent, iSetup);
  analyzeSemiMu(iEvent, iSetup);
  analyzeSemiE(iEvent, iSetup);
  analyzeAllHad(iEvent, iSetup);
}

void B2GDQM::analyzeJets(const Event& iEvent, const edm::EventSetup& iSetup) {
  // Loop over the different types of jets,
  //   Loop over the jets in that collection,
  //     fill PF jet information as well as substructure
  //     information for boosted jets.
  // Utilizes the CMS top-tagging algorithm and the "mass drop" W-tagger.
  for (unsigned int icoll = 0; icoll < jetLabels_.size(); ++icoll) {
    edm::Handle<edm::View<reco::Jet> > pfJetCollection;
    bool ValidPFJets = iEvent.getByToken(jetTokens_[icoll], pfJetCollection);
    if (!ValidPFJets)
      continue;
    edm::View<reco::Jet> const& pfjets = *pfJetCollection;

    // Jet Correction

    for (edm::View<reco::Jet>::const_iterator jet = pfjets.begin(), jetEnd = pfjets.end(); jet != jetEnd; ++jet) {
      if (jet->pt() < jetPtMins_[icoll])
        continue;
      pfJet_pt[icoll]->Fill(jet->pt());
      pfJet_y[icoll]->Fill(jet->rapidity());
      pfJet_phi[icoll]->Fill(jet->phi());
      pfJet_m[icoll]->Fill(jet->mass());

      // Dynamic cast the base class (reco::Jet) to the derived class (PFJet)
      // to access the PFJet information
      reco::PFJet const* pfjet = dynamic_cast<reco::PFJet const*>(&*jet);

      if (pfjet != nullptr) {
        pfJet_chef[icoll]->Fill(pfjet->chargedHadronEnergyFraction());
        pfJet_nhef[icoll]->Fill(pfjet->neutralHadronEnergyFraction());
        pfJet_cemf[icoll]->Fill(pfjet->chargedEmEnergyFraction());
        pfJet_nemf[icoll]->Fill(pfjet->neutralEmEnergyFraction());
      }

      // Dynamic cast the base class (reco::Jet) to the derived class (BasicJet)
      // to access the substructure information
      reco::BasicJet const* basicjet = dynamic_cast<reco::BasicJet const*>(&*jet);

      if (basicjet != nullptr) {
        boostedJet_subjetN[icoll]->Fill(jet->numberOfDaughters());

        for (unsigned int ida = 0; ida < jet->numberOfDaughters(); ++ida) {
          reco::Candidate const* subjet = jet->daughter(ida);
          boostedJet_subjetPt[icoll]->Fill(subjet->pt());
          boostedJet_subjetY[icoll]->Fill(subjet->rapidity());
          boostedJet_subjetPhi[icoll]->Fill(subjet->phi());
          boostedJet_subjetM[icoll]->Fill(subjet->mass());
        }
        // Check the various tagging algorithms
        if ((jetLabels_[icoll].label() == "ak8PFJetsPuppiSoftdrop")) {
          if (jet->numberOfDaughters() > 1) {
            reco::Candidate const* da0 = jet->daughter(0);
            reco::Candidate const* da1 = jet->daughter(1);
            if (da0->mass() > da1->mass()) {
              boostedJet_wMass[icoll]->Fill(da0->mass());
              boostedJet_massDrop[icoll]->Fill(da0->mass() / jet->mass());
            } else {
              boostedJet_wMass[icoll]->Fill(da1->mass());
              boostedJet_massDrop[icoll]->Fill(da1->mass() / jet->mass());
            }

          } else {
            boostedJet_massDrop[icoll]->Fill(-1.0);
          }

        }  // end if collection is AK8 PFJets Puppi soft-drop

      }  // end if basic jet != 0
    }
  }

  // PFMETs
  edm::Handle<std::vector<reco::PFMET> > pfMETCollection;
  bool ValidPFMET = iEvent.getByToken(PFMETToken_, pfMETCollection);
  if (!ValidPFMET)
    return;

  pfMet_pt->Fill((*pfMETCollection)[0].pt());
  pfMet_phi->Fill((*pfMETCollection)[0].phi());
}

void B2GDQM::analyzeAllHad(const Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<reco::BasicJet> > jetCollection;
  bool validJets = iEvent.getByToken(sdjetToken_, jetCollection);
  if (!validJets)
    return;

  // Require two back-to-back jets at high pt with |delta y| < 1.0
  if (jetCollection->size() < 2)
    return;
  edm::Ptr<reco::BasicJet> jet0 = jetCollection->ptrAt(0);
  edm::Ptr<reco::BasicJet> jet1 = jetCollection->ptrAt(1);
  if (jet0.isAvailable() == false || jet1.isAvailable() == false)
    return;
  if (jet0->pt() < allHadPtCut_ || jet1->pt() < allHadPtCut_)
    return;
  if (std::abs(jet0->rapidity() - jet1->rapidity()) > allHadRapidityCut_)
    return;
  if (std::abs(reco::deltaPhi(jet0->phi(), jet1->phi())) < M_PI * 0.5)
    return;

  allHad_pt0->Fill(jet0->pt());
  allHad_y0->Fill(jet0->rapidity());
  allHad_phi0->Fill(jet0->phi());
  allHad_mass0->Fill(jet0->mass());
  if (jet0->numberOfDaughters() > 2) {
    double wMass =
        jet0->daughter(0)->mass() >= jet0->daughter(1)->mass() ? jet0->daughter(0)->mass() : jet0->daughter(1)->mass();
    allHad_wMass0->Fill(wMass);
  } else {
    allHad_wMass0->Fill(-1.0);
  }

  allHad_pt1->Fill(jet1->pt());
  allHad_y1->Fill(jet1->rapidity());
  allHad_phi1->Fill(jet1->phi());
  allHad_mass1->Fill(jet1->mass());
  if (jet1->numberOfDaughters() > 2) {
    double wMass =
        jet1->daughter(0)->mass() >= jet1->daughter(1)->mass() ? jet1->daughter(0)->mass() : jet1->daughter(1)->mass();
    allHad_wMass1->Fill(wMass);
  } else {
    allHad_wMass1->Fill(-1.0);
  }

  auto p4cand = (jet0->p4() + jet1->p4());
  allHad_mttbar->Fill(p4cand.mass());
}

void B2GDQM::analyzeSemiMu(const Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<reco::Muon> > muonCollection;
  bool validMuons = iEvent.getByToken(muonToken_, muonCollection);

  if (!validMuons)
    return;
  if (muonCollection->empty())
    return;
  reco::Muon const& muon = muonCollection->at(0);
  if (!(*muonSelect_)(muon))
    return;

  edm::Handle<edm::View<reco::BasicJet> > jetCollection;
  bool validJets = iEvent.getByToken(sdjetToken_, jetCollection);
  if (!validJets)
    return;
  if (jetCollection->size() < 2)
    return;

  double pt0 = -1.0;
  double dRMin = 999.0;
  edm::Ptr<reco::BasicJet> hadJet;  // highest pt jet with dphi(lep,jet) > pi/2
  edm::Ptr<reco::BasicJet> lepJet;  // closest jet to lepton with pt > ptMin

  for (auto ijet = jetCollection->begin(), ijetBegin = ijet, ijetEnd = jetCollection->end(); ijet != ijetEnd; ++ijet) {
    // Hadronic jets
    if (std::abs(reco::deltaPhi(muon, *ijet)) > M_PI * 0.5) {
      if (ijet->pt() > pt0 && ijet->p() > semiMu_HadJetPtCut_) {
        hadJet = jetCollection->ptrAt(ijet - ijetBegin);
        pt0 = hadJet->pt();
      }
    }
    // Leptonic jets
    else if (ijet->pt() > semiMu_LepJetPtCut_) {
      auto idRMin = reco::deltaR(muon, *ijet);
      if (idRMin < dRMin) {
        lepJet = jetCollection->ptrAt(ijet - ijetBegin);
        dRMin = idRMin;
      }
    }
  }
  if (hadJet.isAvailable() == false || lepJet.isAvailable() == false)
    return;

  auto lepJetP4 = lepJet->p4();
  const auto& muonP4 = muon.p4();

  double tot = lepJetP4.mag2();
  double ss = muonP4.Dot(lepJet->p4());
  double per = muonP4.mag2();
  if (tot > 0.0)
    per -= ss * ss / tot;
  if (per < 0)
    per = 0;
  double ptRel = per;
  bool pass2D = dRMin > semiMu_dRMin_ || ptRel > semiMu_ptRel_;

  if (!pass2D)
    return;

  semiMu_muPt->Fill(muon.pt());
  semiMu_muEta->Fill(muon.eta());
  semiMu_muPhi->Fill(muon.phi());
  semiMu_muDRMin->Fill(dRMin);
  semiMu_muPtRel->Fill(ptRel);

  semiMu_hadJetDR->Fill(reco::deltaR(muon, *hadJet));
  semiMu_mttbar->Fill(0.0);

  semiMu_hadJetPt->Fill(hadJet->pt());
  semiMu_hadJetY->Fill(hadJet->rapidity());
  semiMu_hadJetPhi->Fill(hadJet->phi());
  semiMu_hadJetMass->Fill(hadJet->mass());
  if (hadJet->numberOfDaughters() > 2) {
    double wMass = hadJet->daughter(0)->mass() >= hadJet->daughter(1)->mass() ? hadJet->daughter(0)->mass()
                                                                              : hadJet->daughter(1)->mass();
    semiMu_hadJetWMass->Fill(wMass);
  } else {
    semiMu_hadJetWMass->Fill(-1.0);
  }
}

void B2GDQM::analyzeSemiE(const Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<reco::GsfElectron> > electronCollection;
  bool validElectrons = iEvent.getByToken(electronToken_, electronCollection);

  if (!validElectrons)
    return;
  if (electronCollection->empty())
    return;
  reco::GsfElectron const& electron = electronCollection->at(0);
  if (!(*elecSelect_)(electron))
    return;

  edm::Handle<edm::View<reco::BasicJet> > jetCollection;
  bool validJets = iEvent.getByToken(sdjetToken_, jetCollection);
  if (!validJets)
    return;
  if (jetCollection->size() < 2)
    return;

  double pt0 = -1.0;
  double dRMin = 999.0;
  edm::Ptr<reco::BasicJet> hadJet;  // highest pt jet with dphi(lep,jet) > pi/2
  edm::Ptr<reco::BasicJet> lepJet;  // closest jet to lepton with pt > ptMin

  for (auto ijet = jetCollection->begin(), ijetBegin = ijet, ijetEnd = jetCollection->end(); ijet != ijetEnd; ++ijet) {
    // Hadronic jets
    if (std::abs(reco::deltaPhi(electron, *ijet)) > M_PI * 0.5) {
      if (ijet->pt() > pt0 && ijet->p() > semiE_HadJetPtCut_) {
        hadJet = jetCollection->ptrAt(ijet - ijetBegin);
        pt0 = hadJet->pt();
      }
    }
    // Leptonic jets
    else if (ijet->pt() > semiE_LepJetPtCut_) {
      auto idRMin = reco::deltaR(electron, *ijet);
      if (idRMin < dRMin) {
        lepJet = jetCollection->ptrAt(ijet - ijetBegin);
        dRMin = idRMin;
      }
    }
  }
  if (hadJet.isAvailable() == false || lepJet.isAvailable() == false)
    return;

  auto lepJetP4 = lepJet->p4();
  const auto& electronP4 = electron.p4();

  double tot = lepJetP4.mag2();
  double ss = electronP4.Dot(lepJet->p4());
  double per = electronP4.mag2();
  if (tot > 0.0)
    per -= ss * ss / tot;
  if (per < 0)
    per = 0;
  double ptRel = per;
  bool pass2D = dRMin > semiE_dRMin_ || ptRel > semiE_ptRel_;

  if (!pass2D)
    return;

  semiE_ePt->Fill(electron.pt());
  semiE_eEta->Fill(electron.eta());
  semiE_ePhi->Fill(electron.phi());
  semiE_eDRMin->Fill(dRMin);
  semiE_ePtRel->Fill(ptRel);

  semiE_hadJetDR->Fill(reco::deltaR(electron, *hadJet));
  semiE_mttbar->Fill(0.0);

  semiE_hadJetPt->Fill(hadJet->pt());
  semiE_hadJetY->Fill(hadJet->rapidity());
  semiE_hadJetPhi->Fill(hadJet->phi());
  semiE_hadJetMass->Fill(hadJet->mass());
  if (hadJet->numberOfDaughters() > 2) {
    double wMass = hadJet->daughter(0)->mass() >= hadJet->daughter(1)->mass() ? hadJet->daughter(0)->mass()
                                                                              : hadJet->daughter(1)->mass();
    semiE_hadJetWMass->Fill(wMass);
  } else {
    semiE_hadJetWMass->Fill(-1.0);
  }
}
