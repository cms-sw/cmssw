#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "fastjet/PseudoJet.hh"
#include <vector>

class PuppiAlgo{ 
public:
  PuppiAlgo(edm::ParameterSet &iConfig);
  ~PuppiAlgo();
  //Computing Mean and RMS
  void   reset();
  void   add(const fastjet::PseudoJet &iParticle,const double &iVal,const unsigned int iAlgo);
  void   computeMedRMS(const unsigned int &iAlgo,const double &iPVFrac);
  //Get the Weight
  double compute(std::vector<double> &iVals,double iChi2);
  //Helpers
  double ptMin();
  double etaMin();
  double etaMax();
  int    numAlgos ();
  int    algoId   (const unsigned int &iAlgo);
  bool   isCharged(const unsigned int &iAlgo);
  double coneSize (const unsigned int &iAlgo);
  double neutralPt(int iNPV);

private:  
  unsigned int   fNAlgos;
  float  fEtaMax;
  float  fEtaMin;
  float  fPtMin ;
  double fNeutralPtMin;
  double fNeutralPtSlope;
  std::vector<float>  fPups;
  std::vector<float>  fPupsPV;
  std::vector<int>    fAlgoId;
  std::vector<bool>   fCharged;
  std::vector<bool>   fAdjust;
  std::vector<int>    fCombId;
  std::vector<double> fConeSize;
  std::vector<double> fRMSPtMin;
  std::vector<double> fRMSScaleFactor;
  std::vector<double> fRMS;
  std::vector<double> fMedian;
  std::vector<double> fMean;
  std::vector<int>    fNCount;
};
