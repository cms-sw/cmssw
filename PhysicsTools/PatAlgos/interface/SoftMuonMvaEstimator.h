#ifndef __PhysicsTools_PatAlgos_SoftMuonMvaEstimator__
#define __PhysicsTools_PatAlgos_SoftMuonMvaEstimator__
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "TMVA/Reader.h"

namespace pat {
class SoftMuonMvaEstimator{
public:
  SoftMuonMvaEstimator();
  void initialize(std::string weightsfile);
  void computeMva(const pat::Muon& imuon);
  float mva() const {return mva_;}
  private:
  TMVA::Reader tmvaReader_;
  bool initialized_;
  float mva_;

  // MVA Spectator
  float pID_;
  float pt_;
  float eta_;
  float momID_;
  float dummy_;

  // MVA VAriables
  float segmentCompatibility_;
  float chi2LocalMomentum_;
  float chi2LocalPosition_;
  float glbTrackProbability_;
  float iValidFraction_;
  float layersWithMeasurement_;
  float trkKink_;
  float log2PlusGlbKink_;
  float timeAtIpInOutErr_;
  float outerChi2_;
  float innerChi2_;
  float trkRelChi2_;
  float vMuonHitComb_;
  float qProd_;

  };
}
#endif
