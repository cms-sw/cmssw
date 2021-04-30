#ifndef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_H
#define L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_H

#if defined(__GXX_EXPERIMENTAL_CXX0X__) or defined(CMSSW)
#include <cstdint>
#include <limits>
#define L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
#else
#include <stdint.h>
#endif

namespace l1t {
  class PFTrack;
  class PFCluster;
  class PFCandidate;
  class Muon;
}  // namespace l1t

// the serialization may be hidden if needed
#include <cmath>
#include <vector>

namespace l1tpf_impl {

  struct CaloCluster {
    int16_t hwPt;
    int16_t hwEmPt;
    int16_t hwPtErr;
    int16_t hwEta;
    int16_t hwPhi;
    uint16_t hwFlags;
    bool isEM, used;
    const l1t::PFCluster *src;

    // sorting
    bool operator<(const CaloCluster &other) const { return hwPt > other.hwPt; }

#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
    static constexpr float PT_SCALE = 4.0;     // quantize in units of 0.25 GeV (can be changed)
    static constexpr float ETAPHI_FACTOR = 4;  // size of an ecal crystal in phi in integer units (our choice)
    static constexpr float ETAPHI_SCALE =
        ETAPHI_FACTOR *
        (180. / M_PI);  // M_PI/180 is the size of an ECal crystal; we make a grid that is 4 times that size
    static constexpr int16_t PHI_WRAP = 360 * ETAPHI_FACTOR;  // what is 3.14 in integer

    static int16_t ptToInt16(float pt) {  // avoid overflows
      return std::min<float>(round(pt * CaloCluster::PT_SCALE), std::numeric_limits<int16_t>::max());
    }

    // filling from floating point
    void fill(float pt,
              float emPt,
              float ptErr,
              float eta,
              float phi,
              bool em,
              unsigned int flags,
              const l1t::PFCluster *source = nullptr) {
      hwPt = CaloCluster::ptToInt16(pt);
      hwEmPt = CaloCluster::ptToInt16(emPt);
      hwPtErr = CaloCluster::ptToInt16(ptErr);
      hwEta = round(eta * CaloCluster::ETAPHI_SCALE);
      hwPhi = int16_t(round(phi * CaloCluster::ETAPHI_SCALE)) % CaloCluster::PHI_WRAP;
      isEM = em;
      used = false;
      hwFlags = flags;
      src = source;
    }

    float floatPt() const { return float(hwPt) / CaloCluster::PT_SCALE; }
    float floatEmPt() const { return float(hwEmPt) / CaloCluster::PT_SCALE; }
    float floatPtErr() const { return float(hwPtErr) / CaloCluster::PT_SCALE; }
    static float minFloatPt() { return float(1.0) / CaloCluster::PT_SCALE; }
    float floatEta() const { return float(hwEta) / CaloCluster::ETAPHI_SCALE; }
    float floatPhi() const { return float(hwPhi) / CaloCluster::ETAPHI_SCALE; }
    void setFloatPt(float pt) { hwPt = round(pt * CaloCluster::PT_SCALE); }
    void setFloatEmPt(float emPt) { hwEmPt = round(emPt * CaloCluster::PT_SCALE); }
#endif
  };

  // https://twiki.cern.ch/twiki/bin/view/CMS/L1TriggerPhase2InterfaceSpecifications
  struct InputTrack {
    uint16_t hwInvpt;
    int32_t hwVtxEta;
    int32_t hwVtxPhi;
    bool hwCharge;
    int16_t hwZ0;
    uint16_t hwChi2, hwStubs;
    uint16_t hwFlags;
    const l1t::PFTrack *src;

    enum QualityFlags { PFLOOSE = 1, PFTIGHT = 2, TKEG = 4 };
    bool quality(QualityFlags q) const { return hwFlags & q; }

#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
    static constexpr float INVPT_SCALE = 2E4;           // 1%/pt @ 100 GeV is 2 bits
    static constexpr float VTX_PHI_SCALE = 1 / 1.6E-3;  // 5 micro rad is 2 bits
    static constexpr float VTX_ETA_SCALE = 1 / 1E-4;    // no idea, but assume it's somewhat worse than phi
    static constexpr float Z0_SCALE = 20;               // 1mm is 2 bits
    static constexpr int32_t VTX_ETA_1p3 = 1.3 * InputTrack::VTX_ETA_SCALE;

    // filling from floating point
    void fillInput(
        float pt, float eta, float phi, int charge, float dz, unsigned int flags, const l1t::PFTrack *source = nullptr) {
      hwInvpt = std::min<double>(round(1 / pt * InputTrack::INVPT_SCALE), std::numeric_limits<uint16_t>::max());
      hwVtxEta = round(eta * InputTrack::VTX_ETA_SCALE);
      hwVtxPhi = round(phi * InputTrack::VTX_PHI_SCALE);
      hwCharge = (charge > 0);
      hwZ0 = round(dz * InputTrack::Z0_SCALE);
      hwFlags = flags;
      src = source;
    }

    float floatVtxPt() const { return 1 / (float(hwInvpt) / InputTrack::INVPT_SCALE); }
    float floatVtxEta() const { return float(hwVtxEta) / InputTrack::VTX_ETA_SCALE; }
    float floatVtxPhi() const { return float(hwVtxPhi) / InputTrack::VTX_PHI_SCALE; }
    float floatDZ() const { return float(hwZ0) / InputTrack::Z0_SCALE; }
    int intCharge() const { return hwCharge ? +1 : -1; }
#endif
  };

  struct PropagatedTrack : public InputTrack {
    int16_t hwPt;
    int16_t hwPtErr;
    int16_t hwCaloPtErr;
    int16_t hwEta;  // at calo
    int16_t hwPhi;  // at calo
    bool muonLink;
    bool used;  // note: this flag is not used in the default PF, but is used in alternative algos
    bool fromPV;

    // sorting
    bool operator<(const PropagatedTrack &other) const { return hwPt > other.hwPt; }

#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
    void fillPropagated(
        float pt, float ptErr, float caloPtErr, float caloEta, float caloPhi, unsigned int quality, bool isMuon) {
      hwPt = CaloCluster::ptToInt16(pt);
      hwPtErr = CaloCluster::ptToInt16(ptErr);
      hwCaloPtErr = CaloCluster::ptToInt16(caloPtErr);
      // saturation protection
      if (hwPt == std::numeric_limits<int16_t>::max()) {
        hwCaloPtErr = hwPt / 4;
      }
      hwEta = round(caloEta * CaloCluster::ETAPHI_SCALE);
      hwPhi = int16_t(round(caloPhi * CaloCluster::ETAPHI_SCALE)) % CaloCluster::PHI_WRAP;
      muonLink = isMuon;
      used = false;
    }

    float floatPt() const { return float(hwPt) / CaloCluster::PT_SCALE; }
    float floatPtErr() const { return float(hwPtErr) / CaloCluster::PT_SCALE; }
    float floatCaloPtErr() const { return float(hwCaloPtErr) / CaloCluster::PT_SCALE; }
    float floatEta() const { return float(hwEta) / CaloCluster::ETAPHI_SCALE; }
    float floatPhi() const { return float(hwPhi) / CaloCluster::ETAPHI_SCALE; }
#endif
  };

  struct Muon {
    int16_t hwPt;
    int16_t hwEta;  // at calo
    int16_t hwPhi;  // at calo
    uint16_t hwFlags;
    bool hwCharge;
    const l1t::Muon *src;

    // sorting
    bool operator<(const Muon &other) const { return hwPt > other.hwPt; }

#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
    void fill(float pt, float eta, float phi, int charge, unsigned int flags, const l1t::Muon *source = nullptr) {
      // we assume we use the same discrete ieta, iphi grid for all particles
      hwPt = round(pt * CaloCluster::PT_SCALE);
      hwEta = round(eta * CaloCluster::ETAPHI_SCALE);
      hwPhi = int16_t(round(phi * CaloCluster::ETAPHI_SCALE)) % CaloCluster::PHI_WRAP;
      hwCharge = (charge > 0);
      hwFlags = flags;
      src = source;
    }
    float floatPt() const { return float(hwPt) / CaloCluster::PT_SCALE; }
    float floatEta() const { return float(hwEta) / CaloCluster::ETAPHI_SCALE; }
    float floatPhi() const { return float(hwPhi) / CaloCluster::ETAPHI_SCALE; }
    int intCharge() const { return hwCharge ? +1 : -1; }
#endif
  };

  struct PFParticle {
    int16_t hwPt;
    int16_t hwEta;  // at calo face
    int16_t hwPhi;
    uint8_t hwId;      // CH=0, EL=1, NH=2, GAMMA=3, MU=4
    int16_t hwVtxEta;  // propagate back to Vtx for charged particles (if useful?)
    int16_t hwVtxPhi;
    uint16_t hwFlags;
    CaloCluster cluster;
    PropagatedTrack track;
    bool chargedPV;
    uint16_t hwPuppiWeight;
    uint16_t hwStatus;  // for debugging
    const l1t::Muon *muonsrc;
    const l1t::PFCandidate *src;

    // sorting
    bool operator<(const PFParticle &other) const { return hwPt > other.hwPt; }

#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
    static constexpr float PUPPI_SCALE = 100;

    float floatPt() const { return float(hwPt) / CaloCluster::PT_SCALE; }
    float floatEta() const { return float(hwEta) / CaloCluster::ETAPHI_SCALE; }
    float floatPhi() const { return float(hwPhi) / CaloCluster::ETAPHI_SCALE; }
    float floatVtxEta() const {
      return (track.hwPt > 0 ? track.floatVtxEta() : float(hwVtxEta) / CaloCluster::ETAPHI_SCALE);
    }
    float floatVtxPhi() const {
      return (track.hwPt > 0 ? track.floatVtxPhi() : float(hwVtxPhi) / CaloCluster::ETAPHI_SCALE);
    }
    float floatDZ() const { return float(track.hwZ0) / InputTrack::Z0_SCALE; }
    float floatPuppiW() const { return float(hwPuppiWeight) / PUPPI_SCALE; }
    int intCharge() const { return (track.hwPt > 0 ? track.intCharge() : 0); }
    void setPuppiW(float w) { hwPuppiWeight = std::round(w * PUPPI_SCALE); }
    void setFloatPt(float pt) { hwPt = round(pt * CaloCluster::PT_SCALE); }
#endif
  };

  struct EGParticle {
    int16_t hwPt;
    int16_t hwEta;  // at calo face
    int16_t hwPhi;
    uint16_t hwQual;

    // FIXME: an index would also do...
    CaloCluster cluster;

    // sorting
    bool operator<(const EGParticle &other) const { return hwPt > other.hwPt; }

#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
    void setFloatPt(float pt) { hwPt = round(pt * CaloCluster::PT_SCALE); }
    float floatPt() const { return float(hwPt) / CaloCluster::PT_SCALE; }
    float floatEta() const { return float(hwEta) / CaloCluster::ETAPHI_SCALE; }
    float floatPhi() const { return float(hwPhi) / CaloCluster::ETAPHI_SCALE; }
#endif
  };

  struct EGIso {
    // FIXME: eventually only one iso will be saved
    uint16_t hwIso;
    uint16_t hwPFIso;

#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
    static constexpr float ISO_SCALE = 100;
    void setIso(float iso, uint16_t &hwIso) { hwIso = round(iso * EGIso::ISO_SCALE); }
    void setIso(float iso) { setIso(iso, hwIso); }
    void setPFIso(float iso) { setIso(iso, hwPFIso); }

    float getFloatIso(uint16_t hwIso) const { return float(hwIso) / EGIso::ISO_SCALE; }
    float floatIso() const { return getFloatIso(hwIso); }
    float floatPFIso() const { return getFloatIso(hwPFIso); }
#endif
  };

  struct EGIsoPV : public EGIso {
    uint16_t hwIsoPV;
    uint16_t hwPFIsoPV;

#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
    void setIsoPV(float iso) { setIso(iso, hwIsoPV); }
    void setPFIsoPV(float iso) { setIso(iso, hwPFIsoPV); }

    float floatIsoPV() const { return getFloatIso(hwIsoPV); }
    float floatPFIsoPV() const { return getFloatIso(hwPFIsoPV); }
#endif
  };

  struct EGIsoEleParticle : public EGParticle, public EGIso {
    // track parameters for electrons
    int16_t hwVtxEta;
    int16_t hwVtxPhi;
    int16_t hwZ0;
    bool hwCharge;
    PropagatedTrack track;

#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE

    float floatVtxEta() const { return float(hwVtxEta) / InputTrack::VTX_ETA_SCALE; }
    float floatVtxPhi() const { return float(hwVtxPhi) / InputTrack::VTX_PHI_SCALE; }
    float floatDZ() const { return float(track.hwZ0) / InputTrack::Z0_SCALE; }
    int intCharge() const { return hwCharge ? +1 : -1; }

#endif
  };

  struct EGIsoParticle : public EGParticle, public EGIsoPV {
#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
    // NOTE: this is needed because of CMSSW requirements
    // i.e. we need to put the EG object and the TkEm and TkEle ones at the same time to have a valid ref
    int ele_idx;
#endif
  };

  struct InputRegion {
    float etaCenter, etaMin, etaMax, phiCenter, phiHalfWidth;
    float etaExtra, phiExtra;
    std::vector<CaloCluster> calo;
    std::vector<CaloCluster> emcalo;
    std::vector<PropagatedTrack> track;
    std::vector<Muon> muon;

    InputRegion()
        : etaCenter(),
          etaMin(),
          etaMax(),
          phiCenter(),
          phiHalfWidth(),
          etaExtra(),
          phiExtra(),
          calo(),
          emcalo(),
          track(),
          muon() {}
    InputRegion(
        float etacenter, float etamin, float etamax, float phicenter, float phihalfwidth, float etaextra, float phiextra)
        : etaCenter(etacenter),
          etaMin(etamin),
          etaMax(etamax),
          phiCenter(phicenter),
          phiHalfWidth(phihalfwidth),
          etaExtra(etaextra),
          phiExtra(phiextra),
          calo(),
          emcalo(),
          track(),
          muon() {}
  };

}  // namespace l1tpf_impl
#endif
