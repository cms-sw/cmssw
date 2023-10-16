/* This Class Header */
#include "DQMOffline/Muon/interface/DiMuonHistograms.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "TLorentzVector.h"
#include "TFile.h"
#include <vector>
#include <cmath>

/* C++ Headers */
#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;
using namespace edm;

DiMuonHistograms::DiMuonHistograms(const edm::ParameterSet& pSet) {
  // initialise parameters:
  parameters = pSet;

  // counter
  nTightTight = 0;
  nMediumMedium = 0;
  nLooseLoose = 0;
  nGlbGlb = 0;

  // declare consumes:
  theMuonCollectionLabel_ = consumes<edm::View<reco::Muon> >(parameters.getParameter<edm::InputTag>("MuonCollection"));
  theVertexLabel_ = consumes<reco::VertexCollection>(parameters.getParameter<edm::InputTag>("VertexLabel"));

  theBeamSpotLabel_ = mayConsume<reco::BeamSpot>(parameters.getParameter<edm::InputTag>("BeamSpotLabel"));

  etaBin = parameters.getParameter<int>("etaBin");
  etaBBin = parameters.getParameter<int>("etaBBin");
  etaEBin = parameters.getParameter<int>("etaEBin");

  etaBMin = parameters.getParameter<double>("etaBMin");
  etaBMax = parameters.getParameter<double>("etaBMax");
  etaECMin = parameters.getParameter<double>("etaECMin");
  etaECMax = parameters.getParameter<double>("etaECMax");

  LowMassMin = parameters.getParameter<double>("LowMassMin");
  LowMassMax = parameters.getParameter<double>("LowMassMax");
  HighMassMin = parameters.getParameter<double>("HighMassMin");
  HighMassMax = parameters.getParameter<double>("HighMassMax");

  theFolder = parameters.getParameter<string>("folder");
}

DiMuonHistograms::~DiMuonHistograms() {}

void DiMuonHistograms::bookHistograms(DQMStore::IBooker& ibooker,
                                      edm::Run const& /*iRun*/,
                                      edm::EventSetup const& /* iSetup */) {
  ibooker.cd();
  ibooker.setCurrentFolder(theFolder);

  int nBin[3] = {etaBin, etaBBin, etaEBin};
  EtaName[0] = "";
  EtaName[1] = "_Barrel";
  EtaName[2] = "_EndCap";
  test = ibooker.book1D("test", "InvMass_{Tight,Tight}", 100, 0., 200.);
  for (unsigned int iEtaRegion = 0; iEtaRegion < 3; iEtaRegion++) {
    GlbGlbMuon_LM.push_back(ibooker.book1D("GlbGlbMuon_LM" + EtaName[iEtaRegion],
                                           "InvMass_{GLB,GLB}" + EtaName[iEtaRegion],
                                           nBin[iEtaRegion],
                                           LowMassMin,
                                           LowMassMax));
    TrkTrkMuon_LM.push_back(ibooker.book1D("TrkTrkMuon_LM" + EtaName[iEtaRegion],
                                           "InvMass_{TRK,TRK}" + EtaName[iEtaRegion],
                                           nBin[iEtaRegion],
                                           LowMassMin,
                                           LowMassMax));
    StaTrkMuon_LM.push_back(ibooker.book1D("StaTrkMuon_LM" + EtaName[iEtaRegion],
                                           "InvMass_{STA,TRK}" + EtaName[iEtaRegion],
                                           nBin[iEtaRegion],
                                           LowMassMin,
                                           LowMassMax));

    GlbGlbMuon_HM.push_back(ibooker.book1D("GlbGlbMuon_HM" + EtaName[iEtaRegion],
                                           "InvMass_{GLB,GLB}" + EtaName[iEtaRegion],
                                           nBin[iEtaRegion],
                                           HighMassMin,
                                           HighMassMax));
    TrkTrkMuon_HM.push_back(ibooker.book1D("TrkTrkMuon_HM" + EtaName[iEtaRegion],
                                           "InvMass_{TRK,TRK}" + EtaName[iEtaRegion],
                                           nBin[iEtaRegion],
                                           HighMassMin,
                                           HighMassMax));
    StaTrkMuon_HM.push_back(ibooker.book1D("StaTrkMuon_HM" + EtaName[iEtaRegion],
                                           "InvMass_{STA,TRK}" + EtaName[iEtaRegion],
                                           nBin[iEtaRegion],
                                           HighMassMin,
                                           HighMassMax));

    // arround the Z peak
    TightTightMuon.push_back(ibooker.book1D("TightTightMuon" + EtaName[iEtaRegion],
                                            "InvMass_{Tight,Tight}" + EtaName[iEtaRegion],
                                            nBin[iEtaRegion],
                                            HighMassMin,
                                            HighMassMax));
    MediumMediumMuon.push_back(ibooker.book1D("MediumMediumMuon" + EtaName[iEtaRegion],
                                              "InvMass_{Medium,Medium}" + EtaName[iEtaRegion],
                                              nBin[iEtaRegion],
                                              HighMassMin,
                                              HighMassMax));
    LooseLooseMuon.push_back(ibooker.book1D("LooseLooseMuon" + EtaName[iEtaRegion],
                                            "InvMass_{Loose,Loose}" + EtaName[iEtaRegion],
                                            nBin[iEtaRegion],
                                            HighMassMin,
                                            HighMassMax));
    //Fraction of bad hits in the tracker track to the total
    TightTightMuonBadFrac.push_back(ibooker.book1D(
        "TightTightMuonBadFrac" + EtaName[iEtaRegion], "BadFrac_{Tight,Tight}" + EtaName[iEtaRegion], 10, 0, 0.4));
    MediumMediumMuonBadFrac.push_back(ibooker.book1D(
        "MediumMediumMuonBadFrac" + EtaName[iEtaRegion], "BadFrac_{Medium,Medium}" + EtaName[iEtaRegion], 10, 0, 0.4));
    LooseLooseMuonBadFrac.push_back(ibooker.book1D(
        "LooseLooseMuonBadFrac" + EtaName[iEtaRegion], "BadFrac_{Loose,Loose}" + EtaName[iEtaRegion], 10, 0, 0.4));

    // low-mass resonances
    SoftSoftMuon.push_back(ibooker.book1D(
        "SoftSoftMuon" + EtaName[iEtaRegion], "InvMass_{Soft,Soft}" + EtaName[iEtaRegion], nBin[iEtaRegion], 0.0, 55.0));
    SoftSoftMuonBadFrac.push_back(ibooker.book1D(
        "SoftSoftMuonBadFrac" + EtaName[iEtaRegion], "BadFrac_{Soft,Soft}" + EtaName[iEtaRegion], 10, 0, 0.4));
  }
}

void DiMuonHistograms::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LogTrace(metname) << "[DiMuonHistograms] Analyze the mu in different eta regions";
  edm::Handle<edm::View<reco::Muon> > muons;
  iEvent.getByToken(theMuonCollectionLabel_, muons);

  // =================================================================================
  // Look for the Primary Vertex (and use the BeamSpot instead, if you can't find it):
  reco::Vertex::Point posVtx;
  reco::Vertex::Error errVtx;
  unsigned int theIndexOfThePrimaryVertex = 999.;

  edm::Handle<reco::VertexCollection> vertex;
  iEvent.getByToken(theVertexLabel_, vertex);
  if (vertex.isValid()) {
    for (unsigned int ind = 0; ind < vertex->size(); ++ind) {
      if ((*vertex)[ind].isValid() && !((*vertex)[ind].isFake())) {
        theIndexOfThePrimaryVertex = ind;
        break;
      }
    }
  }

  if (theIndexOfThePrimaryVertex < 100) {
    posVtx = ((*vertex)[theIndexOfThePrimaryVertex]).position();
    errVtx = ((*vertex)[theIndexOfThePrimaryVertex]).error();
  } else {
    LogInfo("RecoMuonValidator") << "reco::PrimaryVertex not found, use BeamSpot position instead\n";

    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByToken(theBeamSpotLabel_, recoBeamSpotHandle);
    reco::BeamSpot bs = *recoBeamSpotHandle;

    posVtx = bs.position();
    errVtx(0, 0) = bs.BeamWidthX();
    errVtx(1, 1) = bs.BeamWidthY();
    errVtx(2, 2) = bs.sigmaZ();
  }

  const reco::Vertex vtx(posVtx, errVtx);

  if (!muons.isValid())
    return;

  // Loop on muon collection
  TLorentzVector Mu1, Mu2;
  float charge = 99.;
  float InvMass = -99.;

  //Eta regions
  double EtaCutMin[] = {0, etaBMin, etaECMin};
  double EtaCutMax[] = {2.4, etaBMax, etaECMax};

  for (edm::View<reco::Muon>::const_iterator muon1 = muons->begin(); muon1 != muons->end(); ++muon1) {
    LogTrace(metname) << "[DiMuonHistograms] loop over 1st muon" << endl;

    // Loop on second muons to fill invariant mass plots
    for (edm::View<reco::Muon>::const_iterator muon2 = muon1; muon2 != muons->end(); ++muon2) {
      LogTrace(metname) << "[DiMuonHistograms] loop over 2nd muon" << endl;
      if (muon1 == muon2)
        continue;

      // Global-Global Muon
      if (muon1->isGlobalMuon() && muon2->isGlobalMuon()) {
        LogTrace(metname) << "[DiMuonHistograms] Glb-Glb pair" << endl;
        reco::TrackRef recoCombinedGlbTrack1 = muon1->combinedMuon();
        reco::TrackRef recoCombinedGlbTrack2 = muon2->combinedMuon();
        Mu1.SetPxPyPzE(recoCombinedGlbTrack1->px(),
                       recoCombinedGlbTrack1->py(),
                       recoCombinedGlbTrack1->pz(),
                       recoCombinedGlbTrack1->p());
        Mu2.SetPxPyPzE(recoCombinedGlbTrack2->px(),
                       recoCombinedGlbTrack2->py(),
                       recoCombinedGlbTrack2->pz(),
                       recoCombinedGlbTrack2->p());

        charge = recoCombinedGlbTrack1->charge() * recoCombinedGlbTrack2->charge();
        if (charge < 0) {
          InvMass = (Mu1 + Mu2).M();
          for (unsigned int iEtaRegion = 0; iEtaRegion < 3; iEtaRegion++) {
            if (fabs(recoCombinedGlbTrack1->eta()) > EtaCutMin[iEtaRegion] &&
                fabs(recoCombinedGlbTrack1->eta()) < EtaCutMax[iEtaRegion] &&
                fabs(recoCombinedGlbTrack2->eta()) > EtaCutMin[iEtaRegion] &&
                fabs(recoCombinedGlbTrack2->eta()) < EtaCutMax[iEtaRegion]) {
              if (InvMass < LowMassMax)
                GlbGlbMuon_LM[iEtaRegion]->Fill(InvMass);
              if (InvMass > HighMassMin)
                GlbGlbMuon_HM[iEtaRegion]->Fill(InvMass);
            }
          }
        }
        // Also Tight-Tight Muon Selection
        if (muon::isTightMuon(*muon1, vtx) && muon::isTightMuon(*muon2, vtx)) {
          test->Fill(InvMass);
          LogTrace(metname) << "[DiMuonHistograms] Tight-Tight pair" << endl;
          if (charge < 0) {
            for (unsigned int iEtaRegion = 0; iEtaRegion < 3; iEtaRegion++) {
              if (fabs(recoCombinedGlbTrack1->eta()) > EtaCutMin[iEtaRegion] &&
                  fabs(recoCombinedGlbTrack1->eta()) < EtaCutMax[iEtaRegion] &&
                  fabs(recoCombinedGlbTrack2->eta()) > EtaCutMin[iEtaRegion] &&
                  fabs(recoCombinedGlbTrack2->eta()) < EtaCutMax[iEtaRegion]) {
                if (InvMass > 55. && InvMass < 125.) {
                  TightTightMuon[iEtaRegion]->Fill(InvMass);
                  TightTightMuonBadFrac[iEtaRegion]->Fill(muon1->innerTrack()->lost() / muon1->innerTrack()->found());
                }
              }
            }
          }
        }
        // Also Medium-Medium Muon Selection
        if (muon::isMediumMuon(*muon1) && muon::isMediumMuon(*muon2)) {
          test->Fill(InvMass);
          LogTrace(metname) << "[DiMuonHistograms] Medium-Medium pair" << endl;
          if (charge < 0) {
            for (unsigned int iEtaRegion = 0; iEtaRegion < 3; iEtaRegion++) {
              if (fabs(recoCombinedGlbTrack1->eta()) > EtaCutMin[iEtaRegion] &&
                  fabs(recoCombinedGlbTrack1->eta()) < EtaCutMax[iEtaRegion] &&
                  fabs(recoCombinedGlbTrack2->eta()) > EtaCutMin[iEtaRegion] &&
                  fabs(recoCombinedGlbTrack2->eta()) < EtaCutMax[iEtaRegion]) {
                if (InvMass > 55. && InvMass < 125.) {
                  MediumMediumMuon[iEtaRegion]->Fill(InvMass);
                  MediumMediumMuonBadFrac[iEtaRegion]->Fill(muon1->innerTrack()->lost() / muon1->innerTrack()->found());
                }
              }
            }
          }
        }
        // Also Loose-Loose Muon Selection
        if (muon::isLooseMuon(*muon1) && muon::isLooseMuon(*muon2)) {
          test->Fill(InvMass);
          LogTrace(metname) << "[DiMuonHistograms] Loose-Loose pair" << endl;
          if (charge < 0) {
            for (unsigned int iEtaRegion = 0; iEtaRegion < 3; iEtaRegion++) {
              if (fabs(recoCombinedGlbTrack1->eta()) > EtaCutMin[iEtaRegion] &&
                  fabs(recoCombinedGlbTrack1->eta()) < EtaCutMax[iEtaRegion] &&
                  fabs(recoCombinedGlbTrack2->eta()) > EtaCutMin[iEtaRegion] &&
                  fabs(recoCombinedGlbTrack2->eta()) < EtaCutMax[iEtaRegion]) {
                if (InvMass > 55. && InvMass < 125.) {
                  LooseLooseMuon[iEtaRegion]->Fill(InvMass);
                  LooseLooseMuonBadFrac[iEtaRegion]->Fill(muon1->innerTrack()->lost() / muon1->innerTrack()->found());
                }
              }
            }
          }
        }
      }

      // Now check for STA-TRK
      if (muon2->isStandAloneMuon() && muon1->isTrackerMuon()) {
        LogTrace(metname) << "[DiMuonHistograms] STA-Trk pair" << endl;
        reco::TrackRef recoStaTrack = muon2->standAloneMuon();
        reco::TrackRef recoTrack = muon1->track();
        Mu2.SetPxPyPzE(recoStaTrack->px(), recoStaTrack->py(), recoStaTrack->pz(), recoStaTrack->p());
        Mu1.SetPxPyPzE(recoTrack->px(), recoTrack->py(), recoTrack->pz(), recoTrack->p());

        charge = recoStaTrack->charge() * recoTrack->charge();
        if (charge < 0) {
          InvMass = (Mu1 + Mu2).M();
          for (unsigned int iEtaRegion = 0; iEtaRegion < 3; iEtaRegion++) {
            if (fabs(recoStaTrack->eta()) > EtaCutMin[iEtaRegion] &&
                fabs(recoStaTrack->eta()) < EtaCutMax[iEtaRegion] && fabs(recoTrack->eta()) > EtaCutMin[iEtaRegion] &&
                fabs(recoTrack->eta()) < EtaCutMax[iEtaRegion]) {
              if (InvMass < LowMassMax)
                StaTrkMuon_LM[iEtaRegion]->Fill(InvMass);
              if (InvMass > HighMassMin)
                StaTrkMuon_HM[iEtaRegion]->Fill(InvMass);
            }
          }
        }
      }
      if (muon1->isStandAloneMuon() && muon2->isTrackerMuon()) {
        LogTrace(metname) << "[DiMuonHistograms] STA-Trk pair" << endl;
        reco::TrackRef recoStaTrack = muon1->standAloneMuon();
        reco::TrackRef recoTrack = muon2->track();
        Mu1.SetPxPyPzE(recoStaTrack->px(), recoStaTrack->py(), recoStaTrack->pz(), recoStaTrack->p());
        Mu2.SetPxPyPzE(recoTrack->px(), recoTrack->py(), recoTrack->pz(), recoTrack->p());

        charge = recoStaTrack->charge() * recoTrack->charge();
        if (charge < 0) {
          InvMass = (Mu1 + Mu2).M();
          for (unsigned int iEtaRegion = 0; iEtaRegion < 3; iEtaRegion++) {
            if (fabs(recoStaTrack->eta()) > EtaCutMin[iEtaRegion] &&
                fabs(recoStaTrack->eta()) < EtaCutMax[iEtaRegion] && fabs(recoTrack->eta()) > EtaCutMin[iEtaRegion] &&
                fabs(recoTrack->eta()) < EtaCutMax[iEtaRegion]) {
              if (InvMass < LowMassMax)
                StaTrkMuon_LM[iEtaRegion]->Fill(InvMass);
              if (InvMass > HighMassMin)
                StaTrkMuon_HM[iEtaRegion]->Fill(InvMass);
            }
          }
        }
      }

      // TRK-TRK dimuon
      if (muon1->isTrackerMuon() && muon2->isTrackerMuon()) {
        LogTrace(metname) << "[DiMuonHistograms] Trk-Trk dimuon pair" << endl;
        reco::TrackRef recoTrack2 = muon2->track();
        reco::TrackRef recoTrack1 = muon1->track();
        Mu2.SetPxPyPzE(recoTrack2->px(), recoTrack2->py(), recoTrack2->pz(), recoTrack2->p());
        Mu1.SetPxPyPzE(recoTrack1->px(), recoTrack1->py(), recoTrack1->pz(), recoTrack1->p());

        charge = recoTrack1->charge() * recoTrack2->charge();
        if (charge < 0) {
          InvMass = (Mu1 + Mu2).M();
          for (unsigned int iEtaRegion = 0; iEtaRegion < 3; iEtaRegion++) {
            if (fabs(recoTrack1->eta()) > EtaCutMin[iEtaRegion] && fabs(recoTrack1->eta()) < EtaCutMax[iEtaRegion] &&
                fabs(recoTrack2->eta()) > EtaCutMin[iEtaRegion] && fabs(recoTrack2->eta()) < EtaCutMax[iEtaRegion]) {
              if (InvMass < LowMassMax)
                TrkTrkMuon_LM[iEtaRegion]->Fill(InvMass);
              if (InvMass > HighMassMin)
                TrkTrkMuon_HM[iEtaRegion]->Fill(InvMass);
            }
          }
        }

        LogTrace(metname) << "[DiMuonHistograms] Soft-Soft pair" << endl;

        if (muon::isSoftMuon(*muon1, vtx) && muon::isSoftMuon(*muon2, vtx)) {
          if (charge < 0) {
            InvMass = (Mu1 + Mu2).M();
            for (unsigned int iEtaRegion = 0; iEtaRegion < 3; iEtaRegion++) {
              if (fabs(recoTrack1->eta()) > EtaCutMin[iEtaRegion] && fabs(recoTrack1->eta()) < EtaCutMax[iEtaRegion] &&
                  fabs(recoTrack2->eta()) > EtaCutMin[iEtaRegion] && fabs(recoTrack2->eta()) < EtaCutMax[iEtaRegion]) {
                SoftSoftMuon[iEtaRegion]->Fill(InvMass);
                SoftSoftMuonBadFrac[iEtaRegion]->Fill(muon1->innerTrack()->lost() / muon1->innerTrack()->found());
              }
            }
          }
        }
      }
    }  //muon2
  }    //Muon1
}
