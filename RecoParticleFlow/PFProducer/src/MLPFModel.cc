#include "RecoParticleFlow/PFProducer/interface/MLPFModel.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"

namespace reco::mlpf {

  //Prepares the input array of floats for a single PFElement
  std::array<float, NUM_ELEMENT_FEATURES> getElementProperties(const reco::PFBlockElement& orig) {
    const auto type = orig.type();
    float pt = 0.0;
    //these are placeholders for the the future
    [[maybe_unused]] float deltap = 0.0;
    [[maybe_unused]] float sigmadeltap = 0.0;
    [[maybe_unused]] float px = 0.0;
    [[maybe_unused]] float py = 0.0;
    [[maybe_unused]] float pz = 0.0;
    float eta = 0.0;
    float phi = 0.0;
    float energy = 0.0;
    float trajpoint = 0.0;
    float eta_ecal = 0.0;
    float phi_ecal = 0.0;
    float eta_hcal = 0.0;
    float phi_hcal = 0.0;
    float charge = 0;
    float layer = 0;
    float depth = 0;
    float muon_dt_hits = 0.0;
    float muon_csc_hits = 0.0;

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
      px = ref->px();
      py = ref->py();
      pz = ref->pz();
      eta = ref->eta();
      phi = ref->phi();
      energy = ref->p();
      charge = ref->charge();

      reco::MuonRef muonRef = orig.muonRef();
      if (muonRef.isNonnull()) {
        reco::TrackRef standAloneMu = muonRef->standAloneMuon();
        if (standAloneMu.isNonnull()) {
          muon_dt_hits = standAloneMu->hitPattern().numberOfValidMuonDTHits();
          muon_csc_hits = standAloneMu->hitPattern().numberOfValidMuonCSCHits();
        }
      }

    } else if (type == reco::PFBlockElement::BREM) {
      const auto* orig2 = (const reco::PFBlockElementBrem*)&orig;
      const auto& ref = orig2->GsftrackRef();
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
        trajpoint = orig2->indTrajPoint();
        charge = ref->charge();
      }
    } else if (type == reco::PFBlockElement::GSF) {
      //requires to keep GsfPFRecTracks
      const auto* orig2 = (const reco::PFBlockElementGsfTrack*)&orig;
      const auto& vec = orig2->Pin();
      pt = vec.pt();
      px = vec.px();
      py = vec.py();
      pz = vec.pz();
      eta = vec.eta();
      phi = vec.phi();
      energy = vec.energy();
      if (!orig2->GsftrackRefPF().isNull()) {
        charge = orig2->GsftrackRefPF()->charge();
      }
    } else if (type == reco::PFBlockElement::ECAL || type == reco::PFBlockElement::PS1 ||
               type == reco::PFBlockElement::PS2 || type == reco::PFBlockElement::HCAL ||
               type == reco::PFBlockElement::HO || type == reco::PFBlockElement::HFHAD ||
               type == reco::PFBlockElement::HFEM) {
      const auto& ref = ((const reco::PFBlockElementCluster*)&orig)->clusterRef();
      if (ref.isNonnull()) {
        eta = ref->eta();
        phi = ref->phi();
        px = ref->position().x();
        py = ref->position().y();
        pz = ref->position().z();
        energy = ref->energy();
        layer = ref->layer();
        depth = ref->depth();
      }
    } else if (type == reco::PFBlockElement::SC) {
      const auto& clref = ((const reco::PFBlockElementSuperCluster*)&orig)->superClusterRef();
      if (clref.isNonnull()) {
        eta = clref->eta();
        phi = clref->phi();
        px = clref->position().x();
        py = clref->position().y();
        pz = clref->position().z();
        energy = clref->energy();
      }
    }

    float typ_idx = static_cast<float>(elem_type_encoding.at(orig.type()));

    //Must be the same order as in tf_model.py
    return std::array<float, NUM_ELEMENT_FEATURES>({{typ_idx,
                                                     pt,
                                                     eta,
                                                     phi,
                                                     energy,
                                                     layer,
                                                     depth,
                                                     charge,
                                                     trajpoint,
                                                     eta_ecal,
                                                     phi_ecal,
                                                     eta_hcal,
                                                     phi_hcal,
                                                     muon_dt_hits,
                                                     muon_csc_hits}});
  }

  //to make sure DNN inputs are within numerical bounds, use the same in training
  float normalize(float in) {
    if (std::abs(in) > 1e4f) {
      return 0.0;
    } else if (std::isnan(in)) {
      return 0.0;
    }
    return in;
  }

  int argMax(std::vector<float> const& vec) {
    return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
  }

  reco::PFCandidate makeCandidate(int pred_pid, int pred_charge, float pred_e, float pred_eta, float pred_phi) {
    pred_phi = angle0to2pi::make0To2pi(pred_phi);

    //currently, set the pT from a massless approximation.
    //later versions of the model may predict predict both the energy and pT of the particle
    float pred_pt = pred_e / cosh(pred_eta);

    //set the charge to +1 or -1 for PFCandidates that are charged, according to the sign of the predicted charge
    reco::PFCandidate::Charge charge = 0;
    if (pred_pid == 11 || pred_pid == 13 || pred_pid == 211) {
      charge = pred_charge > 0 ? +1 : -1;
    }

    math::PtEtaPhiELorentzVectorD p4(pred_pt, pred_eta, pred_phi, pred_e);

    reco::PFCandidate cand(
        0, math::XYZTLorentzVector(p4.X(), p4.Y(), p4.Z(), p4.E()), reco::PFCandidate::ParticleType(0));
    cand.setPdgId(pred_pid);
    cand.setCharge(charge);

    return cand;
  }

  const std::vector<const reco::PFBlockElement*> getPFElements(const reco::PFBlockCollection& blocks) {
    std::vector<reco::PFCandidate> pOutputCandidateCollection;

    std::vector<const reco::PFBlockElement*> all_elements;
    for (const auto& block : blocks) {
      const auto& elems = block.elements();
      for (const auto& elem : elems) {
        if (all_elements.size() < NUM_MAX_ELEMENTS_BATCH) {
          all_elements.push_back(&elem);
        } else {
          //model needs to be created with a bigger LSH codebook size
          edm::LogError("MLPFProducer") << "too many input PFElements for predefined model size: " << elems.size();
          break;
        }
      }
    }
    return all_elements;
  }

  void setCandidateRefs(reco::PFCandidate& cand,
                        const std::vector<const reco::PFBlockElement*> elems,
                        size_t ielem_originator) {
    const reco::PFBlockElement* elem = elems[ielem_originator];
    //set the track ref in case the originating element was a track
    if (elem->type() == reco::PFBlockElement::TRACK && cand.charge() != 0 && elem->trackRef().isNonnull()) {
      cand.setTrackRef(elem->trackRef());

      //set the muon ref in case the originator was a muon
      const auto& muonref = elem->muonRef();
      if (muonref.isNonnull()) {
        cand.setMuonRef(muonref);
      }
    }
  }

};  // namespace reco::mlpf