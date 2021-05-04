#ifndef CommonTools_PileupAlgos_PuppiAlgo_h
#define CommonTools_PileupAlgos_PuppiAlgo_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CommonTools/PileupAlgos/interface/PuppiCandidate.h"
#include <vector>

class PuppiAlgo {
public:
  PuppiAlgo(edm::ParameterSet &iConfig);
  ~PuppiAlgo();
  static void fillDescriptionsPuppiAlgo(edm::ParameterSetDescription &desc);
  //Computing Mean and RMS
  void reset();
  void fixAlgoEtaBin(int i_eta);
  void add(const PuppiCandidate &iParticle, const double &iVal, const unsigned int iAlgo);
  void computeMedRMS(const unsigned int &iAlgo);
  //Get the Weight
  double compute(std::vector<double> const &iVals, double iChi2) const;
  const std::vector<float> &alphas() { return fPups; }
  //Helpers
  inline int etaBins() const { return fEtaMin.size(); }
  inline double etaMin(int i) const { return fEtaMin[i]; }
  inline double etaMax(int i) const { return fEtaMax[i]; }
  inline double ptMin() const { return cur_PtMin; }

  inline int numAlgos() const { return fNAlgos; }
  inline int algoId(unsigned int iAlgo) const { return fAlgoId.at(iAlgo); }
  inline bool isCharged(unsigned int iAlgo) const { return fCharged.at(iAlgo); }
  inline double coneSize(unsigned int iAlgo) const { return fConeSize.at(iAlgo); }
  inline double neutralPt(double const iPUProxy) const { return cur_NeutralPtMin + iPUProxy * cur_NeutralPtSlope; }

  inline double rms() const { return cur_RMS; }
  inline double median() const { return cur_Med; }

  inline double etaMaxExtrap() const { return fEtaMaxExtrap; }

private:
  unsigned int fNAlgos;
  std::vector<double> fEtaMax;
  std::vector<double> fEtaMin;
  std::vector<double> fPtMin;
  std::vector<double> fNeutralPtMin;
  std::vector<double> fNeutralPtSlope;

  std::vector<double> fRMSEtaSF;
  std::vector<double> fMedEtaSF;
  double fEtaMaxExtrap;

  double cur_PtMin;
  double cur_NeutralPtMin;
  double cur_NeutralPtSlope;
  double cur_RMS;
  double cur_Med;

  std::vector<double> fRMS;                          // this is the raw RMS per algo
  std::vector<double> fMedian;                       // this is the raw Median per algo
  std::vector<std::vector<double> > fRMS_perEta;     // this is the final RMS used after eta corrections
  std::vector<std::vector<double> > fMedian_perEta;  // this is the final Med used after eta corrections

  std::vector<float> fPups;
  std::vector<float> fPupsPV;
  std::vector<int> fAlgoId;
  std::vector<bool> fCharged;
  std::vector<bool> fAdjust;
  std::vector<int> fCombId;
  std::vector<double> fConeSize;
  std::vector<double> fRMSPtMin;
  std::vector<double> fRMSScaleFactor;
  std::vector<double> fMean;
  std::vector<int> fNCount;
};

#endif
