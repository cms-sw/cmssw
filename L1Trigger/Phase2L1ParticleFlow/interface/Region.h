#ifndef L1Trigger_Phase2L1ParticleFlow_Region_h
#define L1Trigger_Phase2L1ParticleFlow_Region_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/DiscretePFInputs.h"
#include "DataFormats/Math/interface/deltaPhi.h"

namespace l1tpf_impl {

  struct Region : public InputRegion {
    std::vector<PFParticle> pf;
    std::vector<PFParticle> puppi;
    std::vector<EGIsoEleParticle> egeles;
    std::vector<EGIsoParticle> egphotons;

    unsigned int caloOverflow, emcaloOverflow, trackOverflow, muonOverflow, pfOverflow, puppiOverflow;

    const bool relativeCoordinates;  // whether the eta,phi in each region are global or relative to the region center
    const unsigned int ncaloMax, nemcaloMax, ntrackMax, nmuonMax, npfMax, npuppiMax;
    Region(float etamin,
           float etamax,
           float phicenter,
           float phiwidth,
           float etaextra,
           float phiextra,
           bool useRelativeCoordinates,
           unsigned int ncalomax,
           unsigned int nemcalomax,
           unsigned int ntrackmax,
           unsigned int nmuonmax,
           unsigned int npfmax,
           unsigned int npuppimax)
        : InputRegion(0.5 * (etamin + etamax), etamin, etamax, phicenter, 0.5 * phiwidth, etaextra, phiextra),
          pf(),
          puppi(),
          egeles(),
          egphotons(),
          caloOverflow(),
          emcaloOverflow(),
          trackOverflow(),
          muonOverflow(),
          pfOverflow(),
          puppiOverflow(),
          relativeCoordinates(useRelativeCoordinates),
          ncaloMax(ncalomax),
          nemcaloMax(nemcalomax),
          ntrackMax(ntrackmax),
          nmuonMax(nmuonmax),
          npfMax(npfmax),
          npuppiMax(npuppimax) {}

    enum InputType { calo_type = 0, emcalo_type = 1, track_type = 2, l1mu_type = 3, n_input_types = 4 };
    static const char* inputTypeName(int inputType);

    enum OutputType {
      any_type = 0,
      charged_type = 1,
      neutral_type = 2,
      electron_type = 3,
      pfmuon_type = 4,
      charged_hadron_type = 5,
      neutral_hadron_type = 6,
      photon_type = 7,
      n_output_types = 8
    };
    static const char* outputTypeName(int outputType);

    unsigned int nInput(InputType type) const;
    unsigned int nOutput(OutputType type, bool puppi, bool fiducial = true) const;

    // global coordinates
    bool contains(float eta, float phi) const {
      float dphi = deltaPhi(phiCenter, phi);
      return (etaMin - etaExtra < eta && eta <= etaMax + etaExtra && -phiHalfWidth - phiExtra < dphi &&
              dphi <= phiHalfWidth + phiExtra);
    }
    // global coordinates
    bool fiducial(float eta, float phi) const {
      float dphi = deltaPhi(phiCenter, phi);
      return (etaMin < eta && eta <= etaMax && -phiHalfWidth < dphi && dphi <= phiHalfWidth);
    }
    // possibly local coordinates
    bool fiducialLocal(float localEta, float localPhi) const {
      if (relativeCoordinates) {
        float dphi = deltaPhi(0.f, localPhi);
        return (etaMin < localEta + etaCenter && localEta + etaCenter <= etaMax && -phiHalfWidth < dphi &&
                dphi <= phiHalfWidth);
      }
      float dphi = deltaPhi(phiCenter, localPhi);
      return (etaMin < localEta && localEta <= etaMax && -phiHalfWidth < dphi && dphi <= phiHalfWidth);
    }
    float regionAbsEta() const { return std::abs(etaCenter); }
    float globalAbsEta(float localEta) const { return std::abs(relativeCoordinates ? localEta + etaCenter : localEta); }
    float globalEta(float localEta) const { return relativeCoordinates ? localEta + etaCenter : localEta; }
    float globalPhi(float localPhi) const { return relativeCoordinates ? localPhi + phiCenter : localPhi; }
    float localEta(float globalEta) const { return relativeCoordinates ? globalEta - etaCenter : globalEta; }
    float localPhi(float globalPhi) const { return relativeCoordinates ? deltaPhi(globalPhi, phiCenter) : globalPhi; }

    void zero() {
      calo.clear();
      emcalo.clear();
      track.clear();
      muon.clear();
      pf.clear();
      puppi.clear();
      egeles.clear();
      egphotons.clear();
      caloOverflow = 0;
      emcaloOverflow = 0;
      trackOverflow = 0;
      muonOverflow = 0;
      pfOverflow = 0;
      puppiOverflow = 0;
    }

    void inputCrop(bool doSort);
    void outputCrop(bool doSort);
  };

}  // namespace l1tpf_impl

#endif
