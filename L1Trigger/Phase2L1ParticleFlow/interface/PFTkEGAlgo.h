#ifndef L1Trigger_Phase2L1ParticleFlow_PFTkEGAlgo_h
#define L1Trigger_Phase2L1ParticleFlow_PFTkEGAlgo_h

#include <algorithm>
#include <vector>

#include "L1Trigger/Phase2L1ParticleFlow/interface/Region.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace l1tpf_impl {

  class PFTkEGAlgo {
  public:
    PFTkEGAlgo(const edm::ParameterSet &);
    virtual ~PFTkEGAlgo();
    void runTkEG(Region &r) const;
    void runTkIso(Region &r, const float z0) const;
    void runPFIso(Region &r, const float z0) const;

    bool writeEgSta() const { return writeEgSta_; }

  protected:
    struct IsoParameters {
      IsoParameters(const edm::ParameterSet &);
      float tkQualityPtMin;
      float dZ;
      float dRMin;
      float dRMax;
      float tkQualityChi2Max;
      float dRMin2;
      float dRMax2;
      ;
    };

    int debug_;
    bool doBremRecovery_;
    bool doTkIsolation_;
    bool filterHwQuality_;
    int caloHwQual_;
    float dEtaMaxBrem_;
    float dPhiMaxBrem_;
    std::vector<double> absEtaBoundaries_;
    std::vector<double> dEtaValues_;
    std::vector<double> dPhiValues_;
    float caloEtMin_;
    float trkQualityPtMin_;
    float trkQualityChi2_;
    bool writeEgSta_;
    IsoParameters tkIsoParametersTkEm_;
    IsoParameters tkIsoParametersTkEle_;
    IsoParameters pfIsoParametersTkEm_;
    IsoParameters pfIsoParametersTkEle_;

    void initRegion(Region &r) const;
    void link_emCalo2emCalo(const Region &r, std::vector<int> &emCalo2emCalo) const;
    void link_emCalo2tk(const Region &r, std::vector<int> &emCalo2tk) const;

    template <typename T>
    void compute_isolation_tkEm(
        Region &r, const std::vector<T> &objects, const IsoParameters &params, const float z0, bool isPF) const {
      for (int ic = 0, nc = r.egphotons.size(); ic < nc; ++ic) {
        auto &egphoton = r.egphotons[ic];

        float sumPt = 0.;
        float sumPtPV = 0.;

        for (int itk = 0, ntk = objects.size(); itk < ntk; ++itk) {
          const auto &tk = objects[itk];

          if (tk.floatPt() < params.tkQualityPtMin)
            continue;

          // FIXME: we compare Tk at vertex against the calo variable....shall we correct for the PV position ?
          float d_phi = deltaPhi(tk.floatVtxPhi(), egphoton.floatPhi());
          float d_eta = tk.floatVtxEta() - egphoton.floatEta();
          float dR2 = d_phi * d_phi + d_eta * d_eta;

          if (dR2 > params.dRMin2 && dR2 < params.dRMax2) {
            sumPt += tk.floatPt();
            // PF neutrals are not constrained by PV (since their Z0 is 0 by design)
            if (tk.intCharge() == 0 || std::abs(tk.floatDZ() - z0) < params.dZ)
              sumPtPV += tk.floatPt();
          }
        }
        if (isPF) {
          egphoton.setPFIso(sumPt / egphoton.floatPt());
          egphoton.setPFIsoPV(sumPtPV / egphoton.floatPt());
        } else {
          egphoton.setIso(sumPt / egphoton.floatPt());
          egphoton.setIsoPV(sumPtPV / egphoton.floatPt());
        }
      }
    }

    template <typename T>
    void compute_isolation_tkEle(
        Region &r, const std::vector<T> &objects, const IsoParameters &params, const float z0, bool isPF) const {
      for (int ic = 0, nc = r.egeles.size(); ic < nc; ++ic) {
        auto &egele = r.egeles[ic];

        float sumPt = 0.;

        for (int itk = 0, ntk = objects.size(); itk < ntk; ++itk) {
          const auto &tk = objects[itk];

          if (tk.floatPt() < params.tkQualityPtMin)
            continue;

          // we check the DZ only for charged PFParticles for which Z0 is assigned to (0,0,0)
          if (tk.intCharge() != 0 && std::abs(tk.floatDZ() - egele.floatDZ()) > params.dZ)
            continue;

          float d_phi = deltaPhi(tk.floatVtxPhi(), egele.floatVtxPhi());
          float d_eta = tk.floatVtxEta() - egele.floatVtxEta();
          float dR2 = d_phi * d_phi + d_eta * d_eta;

          if (dR2 > params.dRMin2 && dR2 < params.dRMax2) {
            sumPt += tk.floatPt();
          }
        }
        if (isPF) {
          egele.setPFIso(sumPt / egele.floatPt());
        } else {
          egele.setIso(sumPt / egele.floatPt());
        }
      }
    }

    void eg_algo(Region &r, const std::vector<int> &emCalo2emCalo, const std::vector<int> &emCalo2tk) const;

    void addEgObjsToPF(Region &r, const int calo_idx, const int hwQual, const float ptCorr, const int tk_idx = -1) const;

    EGIsoParticle &addEGIsoToPF(std::vector<EGIsoParticle> &egobjs,
                                const CaloCluster &calo,
                                const int hwQual,
                                const float ptCorr) const;

    EGIsoEleParticle &addEGIsoEleToPF(std::vector<EGIsoEleParticle> &egobjs,
                                      const CaloCluster &calo,
                                      const PropagatedTrack &track,
                                      const int hwQual,
                                      const float ptCorr) const;
  };

}  // namespace l1tpf_impl

#endif
