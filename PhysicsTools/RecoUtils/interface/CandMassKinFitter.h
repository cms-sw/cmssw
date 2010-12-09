#ifndef PhysicsTools_RecoUtils_CandMassKinFitter_h
#define PhysicsTools_RecoUtils_CandMassKinFitter_h
#include "DataFormats/RecoCandidate/interface/FitQuality.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

struct CandMassKinFitter {
  explicit CandMassKinFitter(double m) : mass_(m) { }
  reco::FitQuality set(reco::Candidate &) const;
private:
  virtual double errEt(double et, double eta) const { return 0.1; }
  virtual double errEta(double et, double eta) const { return 0.1; }
  virtual double errPhi(double et, double eta) const { return 0.1; }

  double mass_;
};

#endif
