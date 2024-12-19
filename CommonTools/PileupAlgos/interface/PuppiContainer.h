#ifndef COMMONTOOLS_PUPPI_PUPPICONTAINER_H_
#define COMMONTOOLS_PUPPI_PUPPICONTAINER_H_

#include "CommonTools/PileupAlgos/interface/PuppiAlgo.h"
#include "CommonTools/PileupAlgos/interface/PuppiCandidate.h"
#include "CommonTools/PileupAlgos/interface/RecoObj.h"

class PuppiContainer {
public:
  PuppiContainer(const edm::ParameterSet &iConfig);

  struct Weights {
    std::vector<double> weights;
    std::vector<double> puppiRawAlphas;
    std::vector<double> puppiAlphas;
    std::vector<double> puppiAlphasMed;
    std::vector<double> puppiAlphasRMS;
  };

  Weights calculatePuppiWeights(const std::vector<RecoObj> &iRecoObjects, double iPUProxy);

  int puppiNAlgos() const { return fPuppiAlgo.size(); }

private:
  void initialize(const std::vector<RecoObj> &iRecoObjects,
                  std::vector<PuppiCandidate> &fPFParticles,
                  std::vector<PuppiCandidate> &fPFParticlesForVar,
                  std::vector<PuppiCandidate> &fPFParticlesForVarChargedPV) const;

  double goodVar(PuppiCandidate const &iPart,
                 std::vector<PuppiCandidate> const &iParts,
                 int iOpt,
                 const double iRCone) const;
  void getRMSAvg(int iOpt,
                 std::vector<PuppiCandidate> const &iConstits,
                 std::vector<PuppiCandidate> const &iParticles,
                 std::vector<PuppiCandidate> const &iChargeParticles,
                 std::vector<double> &oVals);
  std::vector<double> getRawAlphas(int iOpt,
                                   std::vector<PuppiCandidate> const &iConstits,
                                   std::vector<PuppiCandidate> const &iParticles,
                                   std::vector<PuppiCandidate> const &iChargeParticles) const;
  double getChi2FromdZ(double iDZ) const;
  int getPuppiId(float iPt, float iEta);
  double var_within_R(int iId,
                      const std::vector<PuppiCandidate> &particles,
                      const PuppiCandidate &centre,
                      const double R) const;

  double fNeutralMinPt;
  double fNeutralSlope;
  double fPuppiWeightCut;
  double fPtMaxPhotons;
  double fEtaMaxPhotons;
  double fPtMaxNeutrals;
  double fPtMaxNeutralsStartSlope;
  std::vector<PuppiAlgo> fPuppiAlgo;

  bool fPuppiDiagnostics;
  bool fApplyCHS;
  bool fInvert;
  bool fUseExp;
};
#endif
