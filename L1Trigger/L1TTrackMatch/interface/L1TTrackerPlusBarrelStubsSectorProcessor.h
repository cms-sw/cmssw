#ifndef L1TTRACKERPLUSBARRELSTUBSSECTORPROCESSOR_H
#define L1TTRACKERPLUSBARRELSTUBSSECTORPROCESSOR_H

#include "DataFormats/L1TMuon/interface/L1MuKBMTCombinedStub.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class L1TTrackerPlusBarrelStubsSectorProcessor {
 public:
  typedef std::vector<edm::Ptr< l1t::L1TkMuonParticle::L1TTTrackType > > TrackPtrVector;

  L1TTrackerPlusBarrelStubsSectorProcessor(const edm::ParameterSet&,int );
  ~L1TTrackerPlusBarrelStubsSectorProcessor();

  std::vector<l1t::L1TkMuonParticle> process(const TrackPtrVector& ,const L1MuKBMTCombinedStubRefVector& stub);
 private:
  int verbose_;
  int sector_;
  std::vector<int> station_;
  int tol_;

  //here is an example configurable that is read from cfg file
  std::vector<double> phi1_;
  std::vector<double> phi2_;
  std::vector<double> propagation_;
  std::vector<double> etaHighm2_;
  std::vector<double> etaHighm1_;
  std::vector<double> etaHigh0_;
  std::vector<double> etaHigh1_;
  std::vector<double> etaHigh2_;
  std::vector<double> etaLowm2_;
  std::vector<double> etaLowm1_;
  std::vector<double> etaLow0_;
  std::vector<double> etaLow1_;
  std::vector<double> etaLow2_;
  std::vector<double> alpha_;
  std::vector<double> beta_;

  int previousSector_;
  int nextSector_;

  int deltaPhi(double p1,double p2);
  int phiProp(double muPhi,int k,int sc,int st);
  double pull(int k,int dphi,int st);
  int stubPhi(const L1MuKBMTCombinedStubRef& stub);
  L1MuKBMTCombinedStubRefVector select(const L1MuKBMTCombinedStubRefVector& stubsPass,const l1t::L1TkMuonParticle& muon,int k);
};

#endif
