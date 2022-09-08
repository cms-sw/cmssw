//This code is for filling the step1 menu objects, for full tree go for L1AnalysisPhaseII.c
#include "L1Trigger/L1TNtuples/interface/L1AnalysisPhaseIIStep1.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"
#include "L1Trigger/Phase2L1GMT/interface/Constants.h"
#include "L1Trigger/L1TTrackMatch/interface/L1TkHTMissEmulatorProducer.h"

L1Analysis::L1AnalysisPhaseIIStep1::L1AnalysisPhaseIIStep1() {}

L1Analysis::L1AnalysisPhaseIIStep1::~L1AnalysisPhaseIIStep1() {}

void L1Analysis::L1AnalysisPhaseIIStep1::SetVertices(float z0Puppi,
                                                     const edm::Handle<std::vector<l1t::VertexWord> > TkPrimaryVertex) {
  //l1extra_.z0Puppi = z0Puppi;
  l1extra_.z0L1TkPV = TkPrimaryVertex->at(0).z0();
  for (unsigned int i = 0; i < TkPrimaryVertex->size(); i++) {
    l1extra_.z0L1TkAll.push_back(TkPrimaryVertex->at(i).z0());
    //l1extra_.sumL1TkPV.push_back(TkPrimaryVertex->at(i).sum());
    l1extra_.nL1TkPVs++;
  }
}

void L1Analysis::L1AnalysisPhaseIIStep1::SetCaloTau(const edm::Handle<l1t::TauBxCollection> calotau,
                                                    unsigned maxL1Extra) {
  for (int ibx = calotau->getFirstBX(); ibx <= calotau->getLastBX(); ++ibx) {
    for (l1t::TauBxCollection::const_iterator it = calotau->begin(ibx);
         it != calotau->end(ibx) && l1extra_.nCaloTaus < maxL1Extra;
         it++) {
      if (it->pt() > 0) {
        l1extra_.caloTauPt.push_back(it->pt());
        l1extra_.caloTauEt.push_back(it->et());
        l1extra_.caloTauEta.push_back(it->eta());
        l1extra_.caloTauPhi.push_back(it->phi());
        l1extra_.caloTauIEt.push_back(it->hwPt());
        l1extra_.caloTauIEta.push_back(it->hwEta());
        l1extra_.caloTauIPhi.push_back(it->hwPhi());
        l1extra_.caloTauIso.push_back(it->hwIso());
        l1extra_.caloTauBx.push_back(ibx);
        l1extra_.caloTauTowerIPhi.push_back(it->towerIPhi());
        l1extra_.caloTauTowerIEta.push_back(it->towerIEta());
        l1extra_.caloTauRawEt.push_back(it->rawEt());
        l1extra_.caloTauIsoEt.push_back(it->isoEt());
        l1extra_.caloTauNTT.push_back(it->nTT());
        l1extra_.caloTauHasEM.push_back(it->hasEM());
        l1extra_.caloTauIsMerged.push_back(it->isMerged());
        l1extra_.caloTauHwQual.push_back(it->hwQual());
        l1extra_.nCaloTaus++;
      }
    }
  }
}

void L1Analysis::L1AnalysisPhaseIIStep1::SetHPSPFTaus(const edm::Handle<l1t::HPSPFTauCollection> l1HPSPFTaus,
                                                      unsigned maxL1Extra) {
  for (unsigned int i = 0; i < l1HPSPFTaus->size() && l1extra_.nHPSTaus < maxL1Extra; i++) {
    if (l1HPSPFTaus->at(i).pt() < 1)
      continue;
    l1extra_.hpsTauPt.push_back(l1HPSPFTaus->at(i).pt());
    l1extra_.hpsTauEt.push_back(l1HPSPFTaus->at(i).et());
    l1extra_.hpsTauEta.push_back(l1HPSPFTaus->at(i).eta());
    l1extra_.hpsTauPhi.push_back(l1HPSPFTaus->at(i).phi());
    l1extra_.hpsTauChg.push_back(l1HPSPFTaus->at(i).charge());
    l1extra_.hpsTauType.push_back(l1HPSPFTaus->at(i).tauType());
    l1extra_.hpsTauPassTightRelIso.push_back(l1HPSPFTaus->at(i).passTightRelIso());
    bool hpsTauPassTightRelIsoMenuVar = (l1HPSPFTaus->at(i).sumChargedIso() / l1HPSPFTaus->at(i).pt()) < 0.05;
    l1extra_.hpsTauPassTightRelIsoMenu.push_back(hpsTauPassTightRelIsoMenuVar);
    l1extra_.hpsTauZ0.push_back(l1HPSPFTaus->at(i).z());
    //if (!l1HPSPFTaus->at(i).leadChargedPFCand().isNull()){
    //    l1extra_.hpsTauZ0.push_back(l1HPSPFTaus->at(i).leadChargedPFCand()->pfTrack()->vertex().z());
    //}else{
    //    l1extra_.hpsTauZ0.push_back(-9999);
    //}
    l1extra_.nHPSTaus++;
  }
}

void L1Analysis::L1AnalysisPhaseIIStep1::SetCaloJet(const edm::Handle<l1t::JetBxCollection> jet,
                                                    unsigned maxL1Extra,
                                                    float caloJetHTT) {
  double mHT30_px = 0, mHT30_py = 0, HT30 = 0;
  double mHT30_3p5_px = 0, mHT30_3p5_py = 0, HT30_3p5 = 0;

  for (int ibx = jet->getFirstBX(); ibx <= jet->getLastBX(); ++ibx) {
    for (l1t::JetBxCollection::const_iterator it = jet->begin(ibx);
         it != jet->end(ibx) && l1extra_.nCaloJets < maxL1Extra;
         it++) {
      if (it->pt() > 0) {
        l1extra_.caloJetEt.push_back(it->et());
        l1extra_.caloJetPt.push_back(it->pt());
        l1extra_.caloJetEta.push_back(it->eta());
        l1extra_.caloJetPhi.push_back(it->phi());
        l1extra_.caloJetBx.push_back(ibx);
        l1extra_.nCaloJets++;

        if (it->pt() > 30 && fabs(it->eta()) < 2.4) {
          HT30 += it->pt();
          mHT30_px += it->px();
          mHT30_py += it->py();
        }
        if (it->pt() > 30 && fabs(it->eta()) < 3.5) {
          HT30_3p5 += it->pt();
          mHT30_3p5_px += it->px();
          mHT30_3p5_py += it->py();
        }
      }
    }
  }

  l1extra_.caloJetHT = caloJetHTT;

  l1extra_.caloJetMHTMenuEt.push_back(sqrt(mHT30_px * mHT30_px + mHT30_py * mHT30_py));
  l1extra_.caloJetMHTMenuPhi.push_back(atan(mHT30_py / mHT30_px));
  l1extra_.caloJetHTMenu.push_back(HT30);

  l1extra_.caloJetMHTMenuEt.push_back(sqrt(mHT30_3p5_px * mHT30_3p5_px + mHT30_3p5_py * mHT30_3p5_py));
  l1extra_.caloJetMHTMenuPhi.push_back(atan(mHT30_3p5_py / mHT30_3p5_px));
  l1extra_.caloJetHTMenu.push_back(HT30_3p5);

  l1extra_.nCaloJetMHTMenu = 2;
}

//EG (seeded by Phase 2 Objects )
void L1Analysis::L1AnalysisPhaseIIStep1::SetEG(const edm::Handle<l1t::EGammaBxCollection> EG,
                                               const edm::Handle<l1t::EGammaBxCollection> EGHGC,
                                               unsigned maxL1Extra) {
  for (l1t::EGammaBxCollection::const_iterator it = EG->begin(); it != EG->end() && l1extra_.nEG < maxL1Extra; it++) {
    if (it->et() > 5) {
      l1extra_.EGPt.push_back(it->pt());
      l1extra_.EGEt.push_back(it->et());
      l1extra_.EGEta.push_back(it->eta());
      l1extra_.EGPhi.push_back(it->phi());
      l1extra_.EGIso.push_back(it->isoEt());
      l1extra_.EGHwQual.push_back(it->hwQual());
      l1extra_.EGBx.push_back(0);  //it->bx());
      l1extra_.EGHGC.push_back(0);
      bool quality = ((it->hwQual() >> 1) & 1) > 0;
      l1extra_.EGPassesLooseTrackID.push_back(quality);
      quality = ((it->hwQual() >> 2) & 1) > 0;
      l1extra_.EGPassesPhotonID.push_back(quality);
      l1extra_.nEG++;
    }
  }

  for (l1t::EGammaBxCollection::const_iterator it = EGHGC->begin(); it != EGHGC->end() && l1extra_.nEG < maxL1Extra;
       it++) {
    if (it->et() > 5) {
      l1extra_.EGPt.push_back(it->pt());
      l1extra_.EGEt.push_back(it->et());
      l1extra_.EGEta.push_back(it->eta());
      l1extra_.EGPhi.push_back(it->phi());
      l1extra_.EGIso.push_back(it->isoEt());
      l1extra_.EGHwQual.push_back(it->hwQual());
      l1extra_.EGBx.push_back(0);  //it->bx());
      l1extra_.EGHGC.push_back(1);
      bool quality = (it->hwQual() == 3);
      l1extra_.EGPassesLooseTrackID.push_back(quality);
      l1extra_.EGPassesPhotonID.push_back(quality);
      l1extra_.nEG++;
    }
  }
}

// TrkEG (seeded by Phase 2 Objects)
void L1Analysis::L1AnalysisPhaseIIStep1::SetTkEG(const edm::Handle<l1t::TkElectronCollection> tkElectron,
                                                 const edm::Handle<l1t::TkElectronCollection> tkElectronHGC,
                                                 unsigned maxL1Extra) {
  for (l1t::TkElectronCollection::const_iterator it = tkElectron->begin();
       it != tkElectron->end() && l1extra_.nTkElectrons < maxL1Extra;
       it++) {
    if (it->et() > 5) {
      l1extra_.tkElectronPt.push_back(it->pt());
      l1extra_.tkElectronEt.push_back(it->et());
      l1extra_.tkElectronEta.push_back(it->eta());
      l1extra_.tkElectronPhi.push_back(it->phi());
      int chargeFromCurvature = (it->trackCurvature() > 0) ? 1 : -1;  // ThisIsACheck
      l1extra_.tkElectronChg.push_back(chargeFromCurvature);
      l1extra_.tkElectronzVtx.push_back(it->trkzVtx());
      l1extra_.tkElectronTrkIso.push_back(it->trkIsol());
      l1extra_.tkElectronPfIso.push_back(it->pfIsol());
      l1extra_.tkElectronPuppiIso.push_back(it->puppiIsol());
      l1extra_.tkElectronHwQual.push_back(it->EGRef()->hwQual());
      l1extra_.tkElectronEGRefPt.push_back(it->EGRef()->et());  //Rename  this?
      l1extra_.tkElectronEGRefEta.push_back(it->EGRef()->eta());
      l1extra_.tkElectronEGRefPhi.push_back(it->EGRef()->phi());
      l1extra_.tkElectronBx.push_back(0);  //it->bx());
      l1extra_.tkElectronHGC.push_back(0);
      bool quality = ((it->EGRef()->hwQual() >> 1) & 1) > 0;  // LooseTrackID should be the second bit
      l1extra_.tkElectronPassesLooseTrackID.push_back(quality);
      quality = ((it->EGRef()->hwQual() >> 2) & 1) > 0;  // LooseTrackID should be the second bit
      l1extra_.tkElectronPassesPhotonID.push_back(quality);
      l1extra_.nTkElectrons++;
    }
  }

  for (l1t::TkElectronCollection::const_iterator it = tkElectronHGC->begin();
       it != tkElectronHGC->end() && l1extra_.nTkElectrons < maxL1Extra;
       it++) {
    if (it->et() > 5) {
      l1extra_.tkElectronPt.push_back(it->pt());
      l1extra_.tkElectronEt.push_back(it->et());
      l1extra_.tkElectronEta.push_back(it->eta());
      l1extra_.tkElectronPhi.push_back(it->phi());
      int chargeFromCurvature = (it->trackCurvature() > 0) ? 1 : -1;  // ThisIsACheck
      l1extra_.tkElectronChg.push_back(chargeFromCurvature);
      l1extra_.tkElectronzVtx.push_back(it->trkzVtx());
      l1extra_.tkElectronTrkIso.push_back(it->trkIsol());
      l1extra_.tkElectronPfIso.push_back(it->pfIsol());
      l1extra_.tkElectronPuppiIso.push_back(it->puppiIsol());
      l1extra_.tkElectronHwQual.push_back(it->EGRef()->hwQual());
      l1extra_.tkElectronEGRefPt.push_back(it->EGRef()->et());  //Rename this?
      l1extra_.tkElectronEGRefEta.push_back(it->EGRef()->eta());
      l1extra_.tkElectronEGRefPhi.push_back(it->EGRef()->phi());
      l1extra_.tkElectronBx.push_back(0);  //it->bx());
      l1extra_.tkElectronHGC.push_back(1);
      bool quality = (it->EGRef()->hwQual() == 3);
      l1extra_.tkElectronPassesLooseTrackID.push_back(quality);
      l1extra_.tkElectronPassesPhotonID.push_back(quality);
      l1extra_.nTkElectrons++;
    }
  }
}

void L1Analysis::L1AnalysisPhaseIIStep1::SetTkEM(const edm::Handle<l1t::TkEmCollection> tkPhoton,
                                                 const edm::Handle<l1t::TkEmCollection> tkPhotonHGC,
                                                 unsigned maxL1Extra) {
  for (l1t::TkEmCollection::const_iterator it = tkPhoton->begin();
       it != tkPhoton->end() && l1extra_.nTkPhotons < maxL1Extra;
       it++) {
    if (it->et() > 5) {
      l1extra_.tkPhotonPt.push_back(it->pt());
      l1extra_.tkPhotonEt.push_back(it->et());
      l1extra_.tkPhotonEta.push_back(it->eta());
      l1extra_.tkPhotonPhi.push_back(it->phi());
      l1extra_.tkPhotonTrkIso.push_back(it->trkIsol());
      l1extra_.tkPhotonTrkIsoPV.push_back(it->trkIsolPV());
      l1extra_.tkPhotonPfIso.push_back(it->pfIsol());
      l1extra_.tkPhotonPfIsoPV.push_back(it->pfIsolPV());
      l1extra_.tkPhotonPuppiIso.push_back(it->puppiIsol());
      l1extra_.tkPhotonPuppiIsoPV.push_back(it->puppiIsolPV());
      l1extra_.tkPhotonBx.push_back(0);  //it->bx());
      l1extra_.tkPhotonHwQual.push_back(it->EGRef()->hwQual());
      l1extra_.tkPhotonEGRefPt.push_back(it->EGRef()->et());  //REname this?
      l1extra_.tkPhotonEGRefEta.push_back(it->EGRef()->eta());
      l1extra_.tkPhotonEGRefPhi.push_back(it->EGRef()->phi());
      l1extra_.tkPhotonHGC.push_back(0);
      bool quality = ((it->EGRef()->hwQual() >> 1) & 1) > 0;
      l1extra_.tkPhotonPassesLooseTrackID.push_back(quality);
      quality = ((it->EGRef()->hwQual() >> 2) & 1) > 0;  // Photon Id should be the third bit
      l1extra_.tkPhotonPassesPhotonID.push_back(quality);
      l1extra_.nTkPhotons++;
    }
  }
  for (l1t::TkEmCollection::const_iterator it = tkPhotonHGC->begin();
       it != tkPhotonHGC->end() && l1extra_.nTkPhotons < maxL1Extra;
       it++) {
    if (it->et() > 5) {
      l1extra_.tkPhotonPt.push_back(it->pt());
      l1extra_.tkPhotonEt.push_back(it->et());
      l1extra_.tkPhotonEta.push_back(it->eta());
      l1extra_.tkPhotonPhi.push_back(it->phi());
      l1extra_.tkPhotonTrkIso.push_back(it->trkIsol());
      l1extra_.tkPhotonTrkIsoPV.push_back(it->trkIsolPV());
      l1extra_.tkPhotonPfIso.push_back(it->pfIsol());
      l1extra_.tkPhotonPfIsoPV.push_back(it->pfIsolPV());
      l1extra_.tkPhotonPuppiIso.push_back(it->puppiIsol());
      l1extra_.tkPhotonPuppiIsoPV.push_back(it->puppiIsolPV());
      l1extra_.tkPhotonBx.push_back(0);  //it->bx());
      l1extra_.tkPhotonHwQual.push_back(it->EGRef()->hwQual());
      l1extra_.tkPhotonEGRefPt.push_back(it->EGRef()->et());  //rename this?
      l1extra_.tkPhotonEGRefEta.push_back(it->EGRef()->eta());
      l1extra_.tkPhotonEGRefPhi.push_back(it->EGRef()->phi());
      l1extra_.tkPhotonHGC.push_back(1);
      bool quality = (it->EGRef()->hwQual() == 3);
      l1extra_.tkPhotonPassesLooseTrackID.push_back(quality);
      l1extra_.tkPhotonPassesPhotonID.push_back(quality);
      l1extra_.nTkPhotons++;
    }
  }
}

/*
void L1Analysis::L1AnalysisPhaseIIStep1::SetMuonKF(const edm::Handle<l1t::RegionalMuonCandBxCollection> standaloneMuon,
                                                   unsigned maxL1Extra,
                                                   unsigned int muonDetector) {
  for (int ibx = standaloneMuon->getFirstBX(); ibx <= standaloneMuon->getLastBX(); ++ibx) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator it = standaloneMuon->begin(ibx);
         it != standaloneMuon->end(ibx) && l1extra_.nStandaloneMuons < maxL1Extra;
         it++) {
      if (it->hwPt() > 0) {
        //      std::cout<<"hwPt vs hwPt2?"<<it->hwPt()*0.5<<" "<<it->hwPt2()<<"   "<<it->hwSign()<<"   "<<muonDetector<<std::endl;
        l1extra_.standaloneMuonPt.push_back(it->hwPt() * 0.5);
        l1extra_.standaloneMuonPt2.push_back(it->hwPtUnconstrained());
        l1extra_.standaloneMuonDXY.push_back(it->hwDXY());
        l1extra_.standaloneMuonEta.push_back(it->hwEta() * 0.010875);
        l1extra_.standaloneMuonPhi.push_back(
            l1t::MicroGMTConfiguration::calcGlobalPhi(it->hwPhi(), it->trackFinderType(), it->processor()) * 2 * M_PI /
            576);
        l1extra_.standaloneMuonChg.push_back(pow(-1, it->hwSign()));
        l1extra_.standaloneMuonQual.push_back(it->hwQual());
        l1extra_.standaloneMuonRegion.push_back(muonDetector);
        l1extra_.standaloneMuonBx.push_back(ibx);
        l1extra_.nStandaloneMuons++;
      }
    }
  }
}

void L1Analysis::L1AnalysisPhaseIIStep1::SetMuonEMTF(const edm::Handle<l1t::EMTFTrackCollection> standaloneEMTFMuon,
                                                     unsigned maxL1Extra,
                                                     unsigned int muonDetector) {
  for (l1t::EMTFTrackCollection::const_iterator it = standaloneEMTFMuon->begin();
       it != standaloneEMTFMuon->end() && l1extra_.nStandaloneMuons < maxL1Extra;
       it++) {
    if (it->Pt() > 0) {
      l1extra_.standaloneMuonPt.push_back(it->Pt());
      l1extra_.standaloneMuonPt2.push_back(-999);
      l1extra_.standaloneMuonDXY.push_back(-999);
      l1extra_.standaloneMuonEta.push_back(it->Eta());  // * 0.010875);
      l1extra_.standaloneMuonPhi.push_back(angle_units::operators::convertDegToRad(it->Phi_glob()));
      l1extra_.standaloneMuonChg.push_back(it->Charge());
      l1extra_.standaloneMuonQual.push_back(it->Mode());
      l1extra_.standaloneMuonRegion.push_back(muonDetector);
      l1extra_.standaloneMuonBx.push_back(it->BX());
      l1extra_.nStandaloneMuons++;
    }
  }
}

void L1Analysis::L1AnalysisPhaseIIStep1::SetTkMuon(const edm::Handle<l1t::TkMuonCollection> muon, unsigned maxL1Extra) {
  for (l1t::TkMuonCollection::const_iterator it = muon->begin(); it != muon->end() && l1extra_.nTkMuons < maxL1Extra;
       it++) {
    l1extra_.tkMuonPt.push_back(it->pt());
    l1extra_.tkMuonEta.push_back(it->eta());
    l1extra_.tkMuonPhi.push_back(it->phi());
    int chargeFromCurvature = (it->trackCurvature() > 0) ? 1 : -1;  // ThisIsACheck
    l1extra_.tkMuonChg.push_back(chargeFromCurvature);
    l1extra_.tkMuonTrkIso.push_back(it->trkIsol());
    if (it->muonDetector() != 3) {
      l1extra_.tkMuonMuRefPt.push_back(it->muRef()->hwPt() * 0.5);
      l1extra_.tkMuonMuRefEta.push_back(it->muRef()->hwEta() * 0.010875);
      l1extra_.tkMuonMuRefPhi.push_back(l1t::MicroGMTConfiguration::calcGlobalPhi(it->muRef()->hwPhi(),
                                                                                  it->muRef()->trackFinderType(),
                                                                                  it->muRef()->processor()) *
                                        2 * M_PI / 576);
      l1extra_.tkMuonDRMuTrack.push_back(it->dR());
      l1extra_.tkMuonNMatchedTracks.push_back(it->nTracksMatched());
      l1extra_.tkMuonQual.push_back(it->quality());
      l1extra_.tkMuonMuRefChg.push_back(pow(-1, it->muRef()->hwSign()));
    } else {
      l1extra_.tkMuonMuRefPt.push_back(it->emtfTrk()->Pt());
      l1extra_.tkMuonMuRefEta.push_back(it->emtfTrk()->Eta());
      l1extra_.tkMuonMuRefPhi.push_back(angle_units::operators::convertDegToRad(it->emtfTrk()->Phi_glob()));
      l1extra_.tkMuonDRMuTrack.push_back(it->dR());
      l1extra_.tkMuonNMatchedTracks.push_back(it->nTracksMatched());
      l1extra_.tkMuonQual.push_back(it->quality());
      l1extra_.tkMuonMuRefChg.push_back(it->emtfTrk()->Charge());
    }
    l1extra_.tkMuonRegion.push_back(it->muonDetector());
    l1extra_.tkMuonzVtx.push_back(it->trkzVtx());
    l1extra_.tkMuonBx.push_back(0);  //it->bx());
    l1extra_.nTkMuons++;
  }
}

//global muons

//sta glb
void L1Analysis::L1AnalysisPhaseIIStep1::SetMuon(const edm::Handle<l1t::MuonBxCollection> muon, unsigned maxL1Extra) {
  for (int ibx = muon->getFirstBX(); ibx <= muon->getLastBX(); ++ibx) {
    for (l1t::MuonBxCollection::const_iterator it = muon->begin(ibx);
         it != muon->end(ibx) && l1extra_.nGlobalMuons < maxL1Extra;
         it++) {
      if (it->pt() > 0) {
        l1extra_.globalMuonPt.push_back(it->pt());  //use pT
        l1extra_.globalMuonEta.push_back(it->eta());
        l1extra_.globalMuonPhi.push_back(it->phi());
        l1extra_.globalMuonEtaAtVtx.push_back(it->etaAtVtx());
        l1extra_.globalMuonPhiAtVtx.push_back(it->phiAtVtx());
        l1extra_.globalMuonIEt.push_back(it->hwPt());  //rename?
        l1extra_.globalMuonIEta.push_back(it->hwEta());
        l1extra_.globalMuonIPhi.push_back(it->hwPhi());
        l1extra_.globalMuonIEtaAtVtx.push_back(it->hwEtaAtVtx());
        l1extra_.globalMuonIPhiAtVtx.push_back(it->hwPhiAtVtx());
        l1extra_.globalMuonIDEta.push_back(it->hwDEtaExtra());
        l1extra_.globalMuonIDPhi.push_back(it->hwDPhiExtra());
        l1extra_.globalMuonChg.push_back(it->charge());
        l1extra_.globalMuonIso.push_back(it->hwIso());
        l1extra_.globalMuonQual.push_back(it->hwQual());
        l1extra_.globalMuonTfMuonIdx.push_back(it->tfMuonIndex());
        l1extra_.globalMuonBx.push_back(ibx);
        l1extra_.nGlobalMuons++;
      }
    }
  }
}

//tkmuon global
void L1Analysis::L1AnalysisPhaseIIStep1::SetTkGlbMuon(const edm::Handle<l1t::TkGlbMuonCollection> muon,
                                                      unsigned maxL1Extra) {
  for (l1t::TkGlbMuonCollection::const_iterator it = muon->begin();
       it != muon->end() && l1extra_.nTkGlbMuons < maxL1Extra;
       it++) {
    l1extra_.tkGlbMuonPt.push_back(it->pt());
    l1extra_.tkGlbMuonEta.push_back(it->eta());
    l1extra_.tkGlbMuonPhi.push_back(it->phi());
    l1extra_.tkGlbMuonChg.push_back(it->charge());
    l1extra_.tkGlbMuonTrkIso.push_back(it->trkIsol());
    l1extra_.tkGlbMuonDRMuTrack.push_back(it->dR());
    l1extra_.tkGlbMuonNMatchedTracks.push_back(it->nTracksMatched());
    l1extra_.tkGlbMuonMuRefPt.push_back(it->muRef()->pt());
    l1extra_.tkGlbMuonMuRefEta.push_back(it->muRef()->eta());
    l1extra_.tkGlbMuonMuRefPhi.push_back(it->muRef()->phi());
    l1extra_.tkGlbMuonQual.push_back(it->muRef()->hwQual());  //What to do with this?
    l1extra_.tkGlbMuonzVtx.push_back(it->trkzVtx());
    l1extra_.tkGlbMuonBx.push_back(0);  //it->bx());
    l1extra_.nTkGlbMuons++;
  }
}
*/



void L1Analysis::L1AnalysisPhaseIIStep1::SetL1PfPhase1L1TJet(
    const edm::Handle<std::vector<reco::CaloJet> > l1L1PFPhase1L1Jet, unsigned maxL1Extra) {
  double mHT30_px = 0, mHT30_py = 0, HT30 = 0;
  double mHT30_3p5_px = 0, mHT30_3p5_py = 0, HT30_3p5 = 0;

  for (reco::CaloJetCollection::const_iterator it = l1L1PFPhase1L1Jet->begin();
       it != l1L1PFPhase1L1Jet->end() && l1extra_.nPhase1PuppiJets < maxL1Extra;
       it++) {
    if (it->pt() > 0) {
      l1extra_.phase1PuppiJetPt.push_back(it->pt());
      l1extra_.phase1PuppiJetEt.push_back(it->et());
      l1extra_.phase1PuppiJetEta.push_back(it->eta());
      l1extra_.phase1PuppiJetPhi.push_back(it->phi());
      //      l1extra_.phase1PuppiJetBx .push_back(0);
      l1extra_.nPhase1PuppiJets++;

      if (it->pt() > 30 && fabs(it->eta()) < 2.4) {  //use pT
        HT30 += it->pt();
        mHT30_px += it->px();
        mHT30_py += it->py();
      }
      if (it->pt() > 30 && fabs(it->eta()) < 3.5) {
        HT30_3p5 += it->pt();
        mHT30_3p5_px += it->px();
        mHT30_3p5_py += it->py();
      }
    }
  }

  l1extra_.nPhase1PuppiMHTMenu = 2;

  l1extra_.phase1PuppiMHTMenuEt.push_back(sqrt(mHT30_px * mHT30_px + mHT30_py * mHT30_py));
  l1extra_.phase1PuppiMHTMenuPhi.push_back(atan(mHT30_py / mHT30_px));
  l1extra_.phase1PuppiHTMenu.push_back(HT30);

  l1extra_.phase1PuppiMHTMenuEt.push_back(sqrt(mHT30_3p5_px * mHT30_3p5_px + mHT30_3p5_py * mHT30_3p5_py));
  l1extra_.phase1PuppiMHTMenuPhi.push_back(atan(mHT30_3p5_py / mHT30_3p5_px));
  l1extra_.phase1PuppiHTMenu.push_back(HT30_3p5);
}

void L1Analysis::L1AnalysisPhaseIIStep1::SetL1PfPhase1L1TJetMET(
    const edm::Handle<std::vector<l1t::EtSum> > l1L1PFPhase1L1JetMET, unsigned maxL1Extra) {
  l1t::EtSum met = l1L1PFPhase1L1JetMET->at(0);
  //cout<<met.et()<< " and " <<met.phi()<<endl;
  l1extra_.phase1PuppiMETEt = met.et();
  l1extra_.phase1PuppiMETPhi = met.phi();
}

void L1Analysis::L1AnalysisPhaseIIStep1::SetL1PfPhase1L1TJetSums(
    const edm::Handle<std::vector<l1t::EtSum> > l1L1PFPhase1L1JetSums, unsigned maxL1Extra) {
  //cout<<"testing the size of this sums vector:"<<l1L1PFPhase1L1JetSums<<endl;
  l1t::EtSum HT = l1L1PFPhase1L1JetSums->at(0);
  l1t::EtSum MHT = l1L1PFPhase1L1JetSums->at(1);
  //cout<<HT.pt()<< " and " <<MHT.pt()<<" and "<<MHT.phi()<<endl;
  l1extra_.phase1PuppiHT = HT.pt();
  l1extra_.phase1PuppiMHTEt = MHT.pt();
  l1extra_.phase1PuppiMHTPhi = MHT.phi();
}

void L1Analysis::L1AnalysisPhaseIIStep1::SetPFJet(const edm::Handle<l1t::PFJetCollection> PFJet, unsigned maxL1Extra) {
  for (l1t::PFJetCollection::const_iterator it = PFJet->begin();
       it != PFJet->end() && l1extra_.nSeededConePuppiJets < maxL1Extra;
       it++) {
    l1extra_.seededConePuppiJetPt.push_back(it->pt());
    l1extra_.seededConePuppiJetEt.push_back(it->et());
    l1extra_.seededConePuppiJetEtUnCorr.push_back(it->rawPt());  //rename?
    l1extra_.seededConePuppiJetEta.push_back(it->eta());
    l1extra_.seededConePuppiJetPhi.push_back(it->phi());
    //    l1extra_.seededConePuppiJetzVtx.push_back(it->getJetVtx());
    l1extra_.seededConePuppiJetBx.push_back(0);  //it->bx());
    l1extra_.nSeededConePuppiJets++;
  }

}

void L1Analysis::L1AnalysisPhaseIIStep1::SetL1seededConeMHT(const edm::Handle<std::vector<l1t::EtSum> > l1SeededConeMHT) {
  l1t::EtSum HT = l1SeededConeMHT->at(0);
  l1t::EtSum MHT = l1SeededConeMHT->at(1);
  l1extra_.seededConePuppiHT = HT.pt();
  l1extra_.seededConePuppiMHTEt = MHT.pt();
  l1extra_.seededConePuppiMHTPhi = MHT.phi();
}


void L1Analysis::L1AnalysisPhaseIIStep1::SetL1METPF(const edm::Handle<std::vector<l1t::EtSum> > l1MetPF) {
  l1t::EtSum met = l1MetPF->at(0);
  l1extra_.puppiMETEt = met.et();
  l1extra_.puppiMETPhi = met.phi();
}

//void L1Analysis::L1AnalysisPhaseIIStep1::SetL1METPF(const edm::Handle<std::vector<reco::PFMET> > l1MetPF) {
//  reco::PFMET met = l1MetPF->at(0);
//  l1extra_.puppiMETRecoEt = met.et();
//  l1extra_.puppiMETRecoPhi = met.phi();
//}

void L1Analysis::L1AnalysisPhaseIIStep1::SetNNTaus(const edm::Handle<vector<l1t::PFTau> > l1nnTaus,
                                                   unsigned maxL1Extra) {
  for (unsigned int i = 0; i < l1nnTaus->size() && l1extra_.nNNTaus < maxL1Extra; i++) {
    if (l1nnTaus->at(i).pt() < 1)
      continue;
    l1extra_.nnTauPt.push_back(l1nnTaus->at(i).pt());
    l1extra_.nnTauEt.push_back(l1nnTaus->at(i).et());
    l1extra_.nnTauEta.push_back(l1nnTaus->at(i).eta());
    l1extra_.nnTauPhi.push_back(l1nnTaus->at(i).phi());
    l1extra_.nnTauChg.push_back(l1nnTaus->at(i).charge());
    l1extra_.nnTauChargedIso.push_back(l1nnTaus->at(i).chargedIso());
    l1extra_.nnTauFullIso.push_back(l1nnTaus->at(i).fullIso());
    l1extra_.nnTauID.push_back(l1nnTaus->at(i).id());
    l1extra_.nnTauPassLooseNN.push_back(l1nnTaus->at(i).passLooseNN());
    l1extra_.nnTauPassLoosePF.push_back(l1nnTaus->at(i).passLoosePF());
    l1extra_.nnTauPassTightPF.push_back(l1nnTaus->at(i).passTightPF());
    l1extra_.nnTauPassTightNN.push_back(l1nnTaus->at(i).passTightNN());
    l1extra_.nnTauPassLooseNNMass.push_back(l1nnTaus->at(i).passLooseNNMass());
    l1extra_.nnTauPassTightNNMass.push_back(l1nnTaus->at(i).passTightNNMass());
    l1extra_.nnTauPassMass.push_back(l1nnTaus->at(i).passMass());
    l1extra_.nnTauDXY.push_back(l1nnTaus->at(i).dxy());
    l1extra_.nnTauZ0.push_back(l1nnTaus->at(i).z0());
    l1extra_.nNNTaus++;
  }
}

void L1Analysis::L1AnalysisPhaseIIStep1::SetNNTau2vtxs(const edm::Handle<vector<l1t::PFTau> > l1nnTau2vtxs,
                                                       unsigned maxL1Extra) {
  for (unsigned int i = 0; i < l1nnTau2vtxs->size() && l1extra_.nNNTau2vtxs < maxL1Extra; i++) {
    if (l1nnTau2vtxs->at(i).pt() < 1)
      continue;
    l1extra_.nnTau2vtxPt.push_back(l1nnTau2vtxs->at(i).pt());
    l1extra_.nnTau2vtxEt.push_back(l1nnTau2vtxs->at(i).et());
    l1extra_.nnTau2vtxEta.push_back(l1nnTau2vtxs->at(i).eta());
    l1extra_.nnTau2vtxPhi.push_back(l1nnTau2vtxs->at(i).phi());
    l1extra_.nnTau2vtxChg.push_back(l1nnTau2vtxs->at(i).charge());
    l1extra_.nnTau2vtxChargedIso.push_back(l1nnTau2vtxs->at(i).chargedIso());
    l1extra_.nnTau2vtxFullIso.push_back(l1nnTau2vtxs->at(i).fullIso());
    l1extra_.nnTau2vtxID.push_back(l1nnTau2vtxs->at(i).id());
    l1extra_.nnTau2vtxPassLooseNN.push_back(l1nnTau2vtxs->at(i).passLooseNN());
    l1extra_.nnTau2vtxPassLoosePF.push_back(l1nnTau2vtxs->at(i).passLoosePF());
    l1extra_.nnTau2vtxPassTightPF.push_back(l1nnTau2vtxs->at(i).passTightPF());
    l1extra_.nnTau2vtxPassTightNN.push_back(l1nnTau2vtxs->at(i).passTightNN());
    l1extra_.nnTau2vtxPassLooseNNMass.push_back(l1nnTau2vtxs->at(i).passLooseNNMass());
    l1extra_.nnTau2vtxPassTightNNMass.push_back(l1nnTau2vtxs->at(i).passTightNNMass());
    l1extra_.nnTau2vtxPassMass.push_back(l1nnTau2vtxs->at(i).passMass());
    l1extra_.nnTau2vtxDXY.push_back(l1nnTau2vtxs->at(i).dxy());
    l1extra_.nnTau2vtxZ0.push_back(l1nnTau2vtxs->at(i).z0());
    l1extra_.nNNTau2vtxs++;
  }
}

// TkJet
void L1Analysis::L1AnalysisPhaseIIStep1::SetTkJet(const edm::Handle<l1t::TkJetWordCollection> trackerJet,
                                                  unsigned maxL1Extra) {
  for (l1t::TkJetWordCollection::const_iterator it = trackerJet->begin();
       it != trackerJet->end() && l1extra_.nTrackerJets < maxL1Extra;
       it++) {
    l1extra_.trackerJetPt.push_back(it->pt());
    //l1extra_.trackerJetEt.push_back(it->et());
    l1extra_.trackerJetEta.push_back(it->glbeta());
    l1extra_.trackerJetPhi.push_back(it->glbphi());
    //l1extra_.trackerJetzVtx.push_back(it->jetVtx());
    l1extra_.trackerJetBx.push_back(0);  //it->bx());
    l1extra_.nTrackerJets++;
  }
}

void L1Analysis::L1AnalysisPhaseIIStep1::SetTkJetDisplaced(const edm::Handle<l1t::TkJetWordCollection> trackerJet,
                                                           unsigned maxL1Extra) {
  for (l1t::TkJetWordCollection::const_iterator it = trackerJet->begin();
       it != trackerJet->end() && l1extra_.nTrackerJets < maxL1Extra;
       it++) {
    l1extra_.trackerJetDisplacedPt.push_back(it->pt());
    //l1extra_.trackerJetDisplacedEt.push_back(it->et());
    l1extra_.trackerJetDisplacedEta.push_back(it->glbeta());
    l1extra_.trackerJetDisplacedPhi.push_back(it->glbphi());
    l1extra_.trackerJetDisplacedZ0.push_back(it->z0());
    l1extra_.trackerJetDisplacedBx.push_back(0);  //it->bx());
    l1extra_.nTrackerJetsDisplaced++;
  }
}

void L1Analysis::L1AnalysisPhaseIIStep1::SetTkMET(const edm::Handle<std::vector<l1t::EtSum> > trackerMET) {
  l1extra_.trackerMET = trackerMET->begin()->hwPt() * l1tmetemu::kStepMETwordEt;
  l1extra_.trackerMETPhi = trackerMET->begin()->hwPhi() * l1tmetemu::kStepMETwordPhi;
}

void L1Analysis::L1AnalysisPhaseIIStep1::SetTkMHT(const edm::Handle<std::vector<l1t::EtSum> > trackerMHT) {
  l1extra_.trackerMHT = trackerMHT->begin()->p4().energy() * l1tmhtemu::kStepMHT;
  l1extra_.trackerHT = trackerMHT->begin()->hwPt() * l1tmhtemu::kStepPt;
  l1extra_.trackerMHTPhi = trackerMHT->begin()->hwPhi() * l1tmhtemu::kStepMHTPhi - M_PI;
}


void L1Analysis::L1AnalysisPhaseIIStep1::SetTkMHTDisplaced(const edm::Handle<std::vector<l1t::EtSum> > trackerMHT) {
  l1extra_.trackerHTDisplaced = trackerMHT->begin()->hwPt() * l1tmhtemu::kStepPt;
  l1extra_.trackerMHTDisplaced = trackerMHT->begin()->p4().energy() * l1tmhtemu::kStepMHT;
  l1extra_.trackerMHTPhiDisplaced = trackerMHT->begin()->hwPhi() * l1tmhtemu::kStepMHTPhi - M_PI;
}

//gmt muons
void L1Analysis::L1AnalysisPhaseIIStep1::SetGmtMuon(const edm::Handle<std::vector<l1t::SAMuon> > gmtMuon,
                                                    unsigned maxL1Extra) {

  for (unsigned int i = 0; i < gmtMuon->size() && l1extra_.nGmtMuons < maxL1Extra; i++) {
    if (lsb_pt * gmtMuon->at(i).hwPt() > 0) {

      l1extra_.gmtMuonPt.push_back(gmtMuon->at(i).phPt());
      l1extra_.gmtMuonEta.push_back(gmtMuon->at(i).phEta());
      l1extra_.gmtMuonPhi.push_back(gmtMuon->at(i).phPhi());
      l1extra_.gmtMuonZ0.push_back(gmtMuon->at(i).phZ0());
      l1extra_.gmtMuonD0.push_back(gmtMuon->at(i).phD0());

      l1extra_.gmtMuonIPt.push_back(gmtMuon->at(i).hwPt());  //rename?
      l1extra_.gmtMuonIEta.push_back(gmtMuon->at(i).hwEta());
      l1extra_.gmtMuonIPhi.push_back(gmtMuon->at(i).hwPhi());
      l1extra_.gmtMuonIZ0.push_back(gmtMuon->at(i).hwZ0());
      l1extra_.gmtMuonID0.push_back(gmtMuon->at(i).hwD0());

      l1extra_.gmtMuonChg.push_back(gmtMuon->at(i).hwCharge());
      l1extra_.gmtMuonIso.push_back(gmtMuon->at(i).hwIso());
      l1extra_.gmtMuonQual.push_back(gmtMuon->at(i).hwQual());
      l1extra_.gmtMuonBeta.push_back(gmtMuon->at(i).hwBeta());

      l1extra_.gmtMuonBx.push_back(0);  //is this just 0 always?

      l1extra_.nGmtMuons++;
    }
  }
}

//tkmuon gmt
void L1Analysis::L1AnalysisPhaseIIStep1::SetGmtTkMuon(const edm::Handle<std::vector<l1t::TrackerMuon> > gmtTkMuon,
  
                                                    unsigned maxL1Extra) {

  for (unsigned int i = 0; i < gmtTkMuon->size() && l1extra_.nGmtTkMuons < maxL1Extra; i++) {
    if (lsb_pt * gmtTkMuon->at(i).hwPt() > 0) {

      l1extra_.gmtTkMuonPt.push_back(gmtTkMuon->at(i).phPt());
      l1extra_.gmtTkMuonEta.push_back(gmtTkMuon->at(i).phEta());
      l1extra_.gmtTkMuonPhi.push_back(gmtTkMuon->at(i).phPhi());
      l1extra_.gmtTkMuonZ0.push_back(gmtTkMuon->at(i).phZ0());
      l1extra_.gmtTkMuonD0.push_back(gmtTkMuon->at(i).phD0());

      l1extra_.gmtTkMuonIPt.push_back(gmtTkMuon->at(i).hwPt());  //rename?
      l1extra_.gmtTkMuonIEta.push_back(gmtTkMuon->at(i).hwEta());
      l1extra_.gmtTkMuonIPhi.push_back(gmtTkMuon->at(i).hwPhi());
      l1extra_.gmtTkMuonIZ0.push_back(gmtTkMuon->at(i).hwZ0());
      l1extra_.gmtTkMuonID0.push_back(gmtTkMuon->at(i).hwD0());

      l1extra_.gmtTkMuonChg.push_back(gmtTkMuon->at(i).hwCharge());
      l1extra_.gmtTkMuonIso.push_back(gmtTkMuon->at(i).hwIso());
      l1extra_.gmtTkMuonQual.push_back(gmtTkMuon->at(i).hwQual());
      l1extra_.gmtTkMuonBeta.push_back(gmtTkMuon->at(i).hwBeta());

      l1extra_.gmtTkMuonNStubs.push_back(gmtTkMuon->at(i).stubs().size());
      l1extra_.gmtTkMuonBx.push_back(0);  //is this just 0 always?

      l1extra_.nGmtTkMuons++;
    }
  }
}
