#ifndef RecoMuon_MuonSeedGenerator_MuonSeedVPtExtractor_H
#define RecoMuon_MuonSeedGenerator_MuonSeedVPtExtractor_H

/** \class MuonSeedVPtExtractor
 */

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h" 


class MuonSeedVPtExtractor {

public:

  MuonSeedVPtExtractor();

  /// Destructor
  virtual ~MuonSeedVPtExtractor() {}


  virtual std::vector<double> pT_extract(MuonTransientTrackingRecHit::ConstMuonRecHitPointer firstHit,
                                 MuonTransientTrackingRecHit::ConstMuonRecHitPointer secondHit) const = 0;

};
#endif
