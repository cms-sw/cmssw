#include "RecoParticleFlow/PFProducer/interface/MLPFModel.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include <TMath.h>

namespace reco::mlpf {

  //Prepares the input array of floats for a single PFElement
  ElementFeatures getElementProperties(const reco::PFBlockElement& orig,
                                       const edm::View<reco::GsfElectron>& gsfElectrons) {
    const auto type = orig.type();

    float pt = 0.0;
    float pterror = 0.0;
    float deltap = 0.0;
    float sigmadeltap = 0.0;
    float px = 0.0;
    float py = 0.0;
    float pz = 0.0;
    float sigma_x = 0.0;
    float sigma_y = 0.0;
    float sigma_z = 0.0;
    float eta = 0.0;
    float etaerror = 0.0;
    float phi = 0.0;
    float phierror = 0.0;
    float lambda = 0.0;
    float lambdaerror = 0.0;
    float theta = 0.0;
    float thetaerror = 0.0;
    float energy = 0.0;
    float vx = 0.0;
    float vy = 0.0;
    float vz = 0.0;
    float corr_energy = 0.0;
    float corr_energy_err = 0.0;
    float trajpoint = 0.0;
    float eta_ecal = 0.0;
    float phi_ecal = 0.0;
    float eta_hcal = 0.0;
    float phi_hcal = 0.0;
    int charge = 0;
    int layer = 0;
    float depth = 0;
    float muon_dt_hits = 0.0;
    float muon_csc_hits = 0.0;
    float muon_type = 0.0;
    float cluster_flags = 0.0;
    float gsf_electronseed_trkorecal = 0.0;
    float gsf_electronseed_dnn1 = 0.0;
    float gsf_electronseed_dnn2 = 0.0;
    float gsf_electronseed_dnn3 = 0.0;
    float gsf_electronseed_dnn4 = 0.0;
    float gsf_electronseed_dnn5 = 0.0;
    float num_hits = 0.0;
    float time = 0.0;
    float timeerror = 0.0;
    float etaerror1 = 0.0;
    float phierror1 = 0.0;
    float etaerror2 = 0.0;
    float phierror2 = 0.0;
    float etaerror3 = 0.0;
    float phierror3 = 0.0;
    float etaerror4 = 0.0;
    float phierror4 = 0.0;

    if (type == reco::PFBlockElement::TRACK) {
      const auto& matched_pftrack = orig.trackRefPF();
      if (matched_pftrack.isNonnull()) {
        const auto& atECAL = matched_pftrack->extrapolatedPoint(reco::PFTrajectoryPoint::ECALShowerMax);
        const auto& atHCAL = matched_pftrack->extrapolatedPoint(reco::PFTrajectoryPoint::HCALEntrance);
        if (atHCAL.isValid()) {
          eta_hcal = atHCAL.positionREP().eta();
          phi_hcal = atHCAL.positionREP().phi();
        }
        if (atECAL.isValid()) {
          eta_ecal = atECAL.positionREP().eta();
          phi_ecal = atECAL.positionREP().phi();
        }
      }
      const auto& ref = ((const reco::PFBlockElementTrack*)&orig)->trackRef();
      pt = ref->pt();
      pterror = ref->ptError();
      px = ref->px();
      py = ref->py();
      pz = ref->pz();
      eta = ref->eta();
      etaerror = ref->etaError();
      phi = ref->phi();
      phierror = ref->phiError();
      energy = ref->p();
      charge = ref->charge();
      num_hits = ref->recHitsSize();
      lambda = ref->lambda();
      lambdaerror = ref->lambdaError();
      theta = ref->theta();
      thetaerror = ref->thetaError();
      vx = ref->vx();
      vy = ref->vy();
      vz = ref->vz();

      const reco::MuonRef& muonRef = orig.muonRef();
      if (muonRef.isNonnull()) {
        reco::TrackRef standAloneMu = muonRef->standAloneMuon();
        if (standAloneMu.isNonnull()) {
          muon_dt_hits = standAloneMu->hitPattern().numberOfValidMuonDTHits();
          muon_csc_hits = standAloneMu->hitPattern().numberOfValidMuonCSCHits();
        }
        muon_type = muonRef->type();
      }

    } else if (type == reco::PFBlockElement::BREM) {
      const auto* orig2 = (const reco::PFBlockElementBrem*)&orig;
      const auto& ref = orig2->GsftrackRef();
      trajpoint = orig2->indTrajPoint();
      if (ref.isNonnull()) {
        deltap = orig2->DeltaP();
        sigmadeltap = orig2->SigmaDeltaP();
        pt = ref->pt();
        px = ref->px();
        py = ref->py();
        pz = ref->pz();
        eta = ref->eta();
        phi = ref->phi();
        energy = ref->p();
        charge = ref->charge();
      }

      const auto& gsfextraref = ref->extra();
      if (gsfextraref.isAvailable() && gsfextraref->seedRef().isAvailable()) {
        reco::ElectronSeedRef seedref = gsfextraref->seedRef().castTo<reco::ElectronSeedRef>();
        if (seedref.isAvailable()) {
          if (seedref->isEcalDriven()) {
            gsf_electronseed_trkorecal = 1.0;
          } else if (seedref->isTrackerDriven()) {
            gsf_electronseed_trkorecal = 2.0;
          }
        }
      }

    } else if (type == reco::PFBlockElement::GSF) {
      //requires to keep GsfPFRecTracks
      const auto* orig2 = (const reco::PFBlockElementGsfTrack*)&orig;
      const auto& ref = orig2->GsftrackRef();

      pt = ref->ptMode();
      pterror = ref->ptModeError();
      px = ref->pxMode();
      py = ref->pyMode();
      pz = ref->pzMode();
      eta = ref->etaMode();
      etaerror = ref->etaModeError();
      phi = ref->phiMode();
      phierror = ref->phiModeError();
      energy = ref->pMode();

      const auto& vec = orig2->Pin();
      eta_ecal = vec.eta();
      phi_ecal = vec.phi();

      const auto& vec2 = orig2->Pout();
      eta_hcal = vec2.eta();
      phi_hcal = vec2.phi();

      if (!orig2->GsftrackRefPF().isNull()) {
        charge = orig2->GsftrackRefPF()->charge();
        num_hits = orig2->GsftrackRefPF()->PFRecBrem().size();
      }

      lambda = ref->lambdaMode();
      lambdaerror = ref->lambdaModeError();
      theta = ref->thetaMode();
      thetaerror = ref->thetaModeError();
      vx = ref->vx();
      vy = ref->vy();
      vz = ref->vz();

      //Find the GSF electron that corresponds to this GSF track
      for (const auto& gsfEle : gsfElectrons) {
        if (ref == gsfEle.gsfTrack()) {
          gsf_electronseed_dnn1 = gsfEle.dnn_signal_Isolated();
          gsf_electronseed_dnn2 = gsfEle.dnn_signal_nonIsolated();
          gsf_electronseed_dnn3 = gsfEle.dnn_bkg_nonIsolated();
          gsf_electronseed_dnn4 = gsfEle.dnn_bkg_Tau();
          gsf_electronseed_dnn5 = gsfEle.dnn_bkg_Photon();
          break;
        }
      }

      const auto& gsfextraref = ref->extra();
      if (gsfextraref.isAvailable() && gsfextraref->seedRef().isAvailable()) {
        reco::ElectronSeedRef seedref = gsfextraref->seedRef().castTo<reco::ElectronSeedRef>();
        if (seedref.isAvailable()) {
          if (seedref->isEcalDriven()) {
            gsf_electronseed_trkorecal = 1.0;
          } else if (seedref->isTrackerDriven()) {
            gsf_electronseed_trkorecal = 2.0;
          }
        }
      };

    } else if (type == reco::PFBlockElement::ECAL || type == reco::PFBlockElement::PS1 ||
               type == reco::PFBlockElement::PS2 || type == reco::PFBlockElement::HCAL ||
               type == reco::PFBlockElement::HO || type == reco::PFBlockElement::HFHAD ||
               type == reco::PFBlockElement::HFEM) {
      const auto& ref = ((const reco::PFBlockElementCluster*)&orig)->clusterRef();
      if (ref.isNonnull()) {
        cluster_flags = ref->flags();
        eta = ref->eta();
        phi = ref->phi();
        pt = ref->pt();
        px = ref->position().x();
        py = ref->position().y();
        pz = ref->position().z();
        energy = ref->energy();
        corr_energy = ref->correctedEnergy();
        corr_energy_err = ref->correctedEnergyUncertainty();
        layer = ref->layer();
        depth = ref->depth();
        num_hits = ref->recHitFractions().size();
        vx = ref->vx();
        vy = ref->vy();
        vz = ref->vz();

        time = ref->time();
        timeerror = ref->timeError();

        std::vector<double> hitE(ref->recHitFractions().size(), 0.0);
        std::vector<double> posEta(ref->recHitFractions().size(), 0.0);
        std::vector<double> posPhi(ref->recHitFractions().size(), 0.0);
        std::vector<double> posX(ref->recHitFractions().size(), 0.0);
        std::vector<double> posY(ref->recHitFractions().size(), 0.0);
        std::vector<double> posZ(ref->recHitFractions().size(), 0.0);
        std::vector<double> depths(ref->recHitFractions().size(), 0.0);

        std::vector<double> depth1_hitE;
        std::vector<double> depth1_posEta;
        std::vector<double> depth1_posPhi;
        std::vector<double> depth2_hitE;
        std::vector<double> depth2_posEta;
        std::vector<double> depth2_posPhi;
        std::vector<double> depth3_hitE;
        std::vector<double> depth3_posEta;
        std::vector<double> depth3_posPhi;
        std::vector<double> depth4_hitE;
        std::vector<double> depth4_posEta;
        std::vector<double> depth4_posPhi;

        const std::vector<reco::PFRecHitFraction>& PFRecHits = ref->recHitFractions();
        unsigned int ihit = 0;
        for (std::vector<reco::PFRecHitFraction>::const_iterator it = PFRecHits.begin(); it != PFRecHits.end(); ++it) {
          const PFRecHitRef& RefPFRecHit = it->recHitRef();
          double energyHit = RefPFRecHit->energy() * it->fraction();
          hitE[ihit] = energyHit;
          posX[ihit] = RefPFRecHit->position().x();
          posY[ihit] = RefPFRecHit->position().y();
          posZ[ihit] = RefPFRecHit->position().z();
          posEta[ihit] = RefPFRecHit->position().eta();
          posPhi[ihit] = deltaPhi(RefPFRecHit->position().phi(), ref->phi());
          depths[ihit] = RefPFRecHit->depth();

          if (depths[ihit] == 1) {
            depth1_hitE.push_back(hitE[ihit]);
            depth1_posEta.push_back(posEta[ihit]);
            depth1_posPhi.push_back(posPhi[ihit]);
          } else if (depths[ihit] == 2) {
            depth2_hitE.push_back(hitE[ihit]);
            depth2_posEta.push_back(posEta[ihit]);
            depth2_posPhi.push_back(posPhi[ihit]);
          } else if (depths[ihit] == 3) {
            depth3_hitE.push_back(hitE[ihit]);
            depth3_posEta.push_back(posEta[ihit]);
            depth3_posPhi.push_back(posPhi[ihit]);
          } else {
            depth4_hitE.push_back(hitE[ihit]);
            depth4_posEta.push_back(posEta[ihit]);
            depth4_posPhi.push_back(posPhi[ihit]);
          }

          ihit++;
        }
        if (ref->recHitFractions().size() > 1) {
          sigma_x = TMath::StdDev(posX.begin(), posX.end(), hitE.begin());
          sigma_y = TMath::StdDev(posY.begin(), posY.end(), hitE.begin());
          sigma_z = TMath::StdDev(posZ.begin(), posZ.end(), hitE.begin());
          pterror = TMath::StdDev(hitE.begin(), hitE.end());
          etaerror = TMath::StdDev(posEta.begin(), posEta.end());
          phierror = TMath::StdDev(posPhi.begin(), posPhi.end());
        }
        if (depth1_hitE.size() > 1) {
          etaerror1 = TMath::StdDev(depth1_posEta.begin(), depth1_posEta.end(), depth1_hitE.begin());
          phierror1 = TMath::StdDev(depth1_posPhi.begin(), depth1_posPhi.end(), depth1_hitE.begin());
        }
        if (depth2_hitE.size() > 1) {
          etaerror2 = TMath::StdDev(depth2_posEta.begin(), depth2_posEta.end(), depth2_hitE.begin());
          phierror2 = TMath::StdDev(depth2_posPhi.begin(), depth2_posPhi.end(), depth2_hitE.begin());
        }
        if (depth3_hitE.size() > 1) {
          etaerror3 = TMath::StdDev(depth3_posEta.begin(), depth3_posEta.end(), depth3_hitE.begin());
          phierror3 = TMath::StdDev(depth3_posPhi.begin(), depth3_posPhi.end(), depth3_hitE.begin());
        }
        if (depth4_hitE.size() > 1) {
          etaerror4 = TMath::StdDev(depth4_posEta.begin(), depth4_posEta.end(), depth4_hitE.begin());
          phierror4 = TMath::StdDev(depth4_posPhi.begin(), depth4_posPhi.end(), depth4_hitE.begin());
        }
      }
    } else if (type == reco::PFBlockElement::SC) {
      const auto& clref = ((const reco::PFBlockElementSuperCluster*)&orig)->superClusterRef();
      if (clref.isNonnull()) {
        //Find the GSF electron that corresponds to this SC
        for (const auto& gsfEle : gsfElectrons) {
          if (clref == gsfEle.superCluster()) {
            gsf_electronseed_dnn1 = gsfEle.dnn_signal_Isolated();
            gsf_electronseed_dnn2 = gsfEle.dnn_signal_nonIsolated();
            gsf_electronseed_dnn3 = gsfEle.dnn_bkg_nonIsolated();
            gsf_electronseed_dnn4 = gsfEle.dnn_bkg_Tau();
            gsf_electronseed_dnn5 = gsfEle.dnn_bkg_Photon();
            break;
          }
        }
        cluster_flags = clref->flags();
        eta = clref->eta();
        phi = clref->phi();
        px = clref->position().x();
        py = clref->position().y();
        pz = clref->position().z();
        energy = clref->energy();
        corr_energy = clref->preshowerEnergy();
        num_hits = clref->clustersSize();
        etaerror = clref->etaWidth();
        phierror = clref->phiWidth();
      }
    }

    ElementFeatures ret;
    ret.type = static_cast<float>(orig.type());
    ret.pt = pt;
    ret.eta = eta;
    ret.phi = phi;
    ret.energy = energy;
    ret.layer = layer;
    ret.depth = depth;
    ret.charge = charge;
    ret.trajpoint = trajpoint;
    ret.eta_ecal = eta_ecal;
    ret.phi_ecal = phi_ecal;
    ret.eta_hcal = eta_hcal;
    ret.phi_hcal = phi_hcal;
    ret.muon_dt_hits = muon_dt_hits;
    ret.muon_csc_hits = muon_csc_hits;
    ret.muon_type = muon_type;
    ret.px = px;
    ret.py = py;
    ret.pz = pz;
    ret.sigma_x = sigma_x;
    ret.sigma_y = sigma_y;
    ret.sigma_z = sigma_z;
    ret.deltap = deltap;
    ret.sigmadeltap = sigmadeltap;
    ret.gsf_electronseed_trkorecal = gsf_electronseed_trkorecal;
    ret.gsf_electronseed_dnn1 = gsf_electronseed_dnn1;
    ret.gsf_electronseed_dnn2 = gsf_electronseed_dnn2;
    ret.gsf_electronseed_dnn3 = gsf_electronseed_dnn3;
    ret.gsf_electronseed_dnn4 = gsf_electronseed_dnn4;
    ret.gsf_electronseed_dnn5 = gsf_electronseed_dnn5;
    ret.num_hits = num_hits;
    ret.cluster_flags = cluster_flags;
    ret.corr_energy = corr_energy;
    ret.corr_energy_err = corr_energy_err;
    ret.vx = vx;
    ret.vy = vy;
    ret.vz = vz;
    ret.pterror = pterror;
    ret.etaerror = etaerror;
    ret.phierror = phierror;
    ret.lambda = lambda;
    ret.lambdaerror = lambdaerror;
    ret.theta = theta;
    ret.thetaerror = thetaerror;
    ret.time = time;
    ret.timeerror = timeerror;
    ret.etaerror1 = etaerror1;
    ret.phierror1 = phierror1;
    ret.etaerror2 = etaerror2;
    ret.phierror2 = phierror2;
    ret.etaerror3 = etaerror3;
    ret.phierror3 = phierror3;
    ret.etaerror4 = etaerror4;
    ret.phierror4 = phierror4;

    return ret;
  }

  //to make sure DNN inputs are within numerical bounds, use the same in training
  float normalize(float in) { return in; }

  int argMax(std::vector<float> const& vec) {
    return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
  }

  reco::PFCandidate makeCandidate(int pred_pid,
                                  int pred_charge,
                                  float pred_pt,
                                  float pred_eta,
                                  float pred_sin_phi,
                                  float pred_cos_phi,
                                  float pred_e) {
    float pred_phi = std::atan2(pred_sin_phi, pred_cos_phi);

    //set the charge to +1 or -1 for PFCandidates that are charged, according to the sign of the predicted charge
    reco::PFCandidate::Charge charge = 0;
    if (pred_pid == 11 || pred_pid == 13 || pred_pid == 211) {
      charge = pred_charge > 0 ? +1 : -1;
    }

    math::PtEtaPhiELorentzVectorD p4(pred_pt, pred_eta, pred_phi, pred_e);

    reco::PFCandidate::ParticleType particleType(reco::PFCandidate::X);
    if (pred_pid == 211)
      particleType = reco::PFCandidate::h;
    else if (pred_pid == 130)
      particleType = reco::PFCandidate::h0;
    else if (pred_pid == 22)
      particleType = reco::PFCandidate::gamma;
    else if (pred_pid == 11)
      particleType = reco::PFCandidate::e;
    else if (pred_pid == 13)
      particleType = reco::PFCandidate::mu;
    else if (pred_pid == 1)
      particleType = reco::PFCandidate::h_HF;
    else if (pred_pid == 2)
      particleType = reco::PFCandidate::egamma_HF;

    reco::PFCandidate cand(charge, math::XYZTLorentzVector(p4.X(), p4.Y(), p4.Z(), p4.E()), particleType);
    cand.setMass(0.0);
    if (pred_pid == 211)
      cand.setMass(PI_MASS);

    //cand.setPdgId(pred_pid);
    //cand.setCharge(charge);

    return cand;
  }

  const std::vector<const reco::PFBlockElement*> getPFElements(const reco::PFBlockCollection& blocks) {
    std::vector<reco::PFCandidate> pOutputCandidateCollection;

    std::vector<const reco::PFBlockElement*> all_elements;
    for (const auto& block : blocks) {
      const auto& elems = block.elements();
      for (const auto& elem : elems) {
        all_elements.push_back(&elem);
      }
    }
    return all_elements;
  }

  //   [4] Calling method for module JetTracksAssociatorExplicit/'ak4JetTracksAssociatorExplicitAll' -> Ref is inconsistent with RefVectorid = (3:3546) should be (3:3559)
  //   [6] Calling method for module MuonProducer/'muons' -> muon::isTightMuon
  void setCandidateRefs(reco::PFCandidate& cand,
                        const std::vector<const reco::PFBlockElement*> elems,
                        size_t ielem_originator) {
    const reco::PFBlockElement* elem = elems[ielem_originator];

    //set the track ref in case the originating element was a track
    if (std::abs(cand.pdgId()) == 211 && elem->type() == reco::PFBlockElement::TRACK && elem->trackRef().isNonnull()) {
      const auto* eltTrack = dynamic_cast<const reco::PFBlockElementTrack*>(elem);
      cand.setTrackRef(eltTrack->trackRef());
      cand.setVertex(eltTrack->trackRef()->vertex());
      cand.setPositionAtECALEntrance(eltTrack->positionAtECALEntrance());
    }

    //set the muon ref
    if (std::abs(cand.pdgId()) == 13) {
      const auto* eltTrack = dynamic_cast<const reco::PFBlockElementTrack*>(elem);
      const auto& muonRef = eltTrack->muonRef();
      cand.setTrackRef(muonRef->track());
      cand.setMuonTrackType(muonRef->muonBestTrackType());
      cand.setVertex(muonRef->track()->vertex());
      cand.setMuonRef(muonRef);
    }

    if (std::abs(cand.pdgId()) == 11) {
      if (elem->type() == reco::PFBlockElement::GSF) {
        const auto* eltTrack = dynamic_cast<const reco::PFBlockElementGsfTrack*>(elem);
        const auto& ref = eltTrack->GsftrackRef();
        cand.setGsfTrackRef(ref);
        cand.setVertex(ref->vertex());
      }
    }
  }

};  // namespace reco::mlpf
