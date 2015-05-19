#ifndef CommonTools_PileupAlgos_PuppiAlgo_h
#define CommonTools_PileupAlgos_PuppiAlgo_h

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
  double compute(std::vector<double> const &iVals,double iChi2) const;
  const std::vector<float> & alphas(){ return fPups; }
  //Helpers
  inline double ptMin() const { return fPtMin; }
  inline double etaMin() const { return fEtaMin; }
  inline double etaMax() const { return fEtaMax; }
  inline int    numAlgos () const { return fNAlgos;}
  inline int    algoId  ( unsigned int iAlgo) const { return fAlgoId.at(iAlgo); }
  inline bool   isCharged  ( unsigned int iAlgo) const { return fCharged.at(iAlgo); }
  inline double coneSize  ( unsigned int iAlgo) const { return fConeSize.at(iAlgo); }
  inline double neutralPt  (int iNPV) const { return fNeutralPtMin + iNPV * fNeutralPtSlope; }

  inline double rms( unsigned int i ) const {return fRMS[i];}
  inline double median( unsigned int i ) const {return fMedian[i];}

private:  
  unsigned int   fNAlgos;
  float  fEtaMax;
  float  fEtaMin;
  float  fPtMin ;
  double fNeutralPtMin;
  double fNeutralPtSlope;

  double fRMSEtaSF;
  double fMedEtaSF;
  double fEtaMaxExtrap;

  std::vector<double> fRMS;
  std::vector<double> fMedian;

  std::vector<float>  fPups;
  std::vector<float>  fPupsPV;
  std::vector<int>    fAlgoId;
  std::vector<bool>   fCharged;
  std::vector<bool>   fAdjust;
  std::vector<int>    fCombId;
  std::vector<double> fConeSize;
  std::vector<double> fRMSPtMin;
  std::vector<double> fRMSScaleFactor;
  std::vector<double> fMean;
  std::vector<int>    fNCount;
};


#endif
