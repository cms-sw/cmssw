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

  //here is an example configurable that is read from cfg file
  std::vector<double> propagation_;

  int previousSector_;
  int nextSector_;

};



#endif
