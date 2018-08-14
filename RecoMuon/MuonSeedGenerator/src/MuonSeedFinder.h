#ifndef RecoMuon_MuonSeedGenerator_MuonSeedFinder_H
#define RecoMuon_MuonSeedGenerator_MuonSeedFinder_H

/** \class MuonSeedFinder
 *  
 *  Uses SteppingHelixPropagator
 *
 *  \author A. Vitelli - INFN Torino
 *  \author porting R. Bellan - INFN Torino
 *
 *  
 */

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedVFinder.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedPtExtractor.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonCSCSeedFromRecHits.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonDTSeedFromRecHits.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonOverlapSeedFromRecHits.h"

#include <vector>

class MuonSeedFinder: public MuonSeedVFinder
{
public:
  /// Constructor
  MuonSeedFinder(const edm::ParameterSet & pset);

  /// Destructor
  ~MuonSeedFinder() override{delete thePtExtractor;}

  // Operations

  void setBField(const MagneticField * field) override;

  void seeds(const MuonTransientTrackingRecHit::MuonRecHitContainer & hits,
             std::vector<TrajectorySeed> & result) override;

private:
  
  float computePt(MuonTransientTrackingRecHit::ConstMuonRecHitPointer muon, const MagneticField *field) const;

  void analyze() const;
  // put a parameterSet instead of
  // static SimpleConfigurable<float> theMinMomentum;
  float theMinMomentum;
  const MagneticField * theField;

  MuonDTSeedFromRecHits theBarrel;
  MuonOverlapSeedFromRecHits theOverlap;
  MuonCSCSeedFromRecHits theEndcap;

};
#endif
