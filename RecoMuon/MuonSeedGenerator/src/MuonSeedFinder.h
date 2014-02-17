#ifndef RecoMuon_MuonSeedGenerator_MuonSeedFinder_H
#define RecoMuon_MuonSeedGenerator_MuonSeedFinder_H

/** \class MuonSeedFinder
 *  
 *  Uses SteppingHelixPropagator
 *
 *  \author A. Vitelli - INFN Torino
 *  \author porting R. Bellan - INFN Torino
 *
 *  $Date: 2010/08/10 20:08:06 $
 *  $Revision: 1.12 $
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
  virtual ~MuonSeedFinder(){delete thePtExtractor;}

  // Operations

  virtual void setBField(const MagneticField * field);

  void seeds(const MuonTransientTrackingRecHit::MuonRecHitContainer & hits,
             std::vector<TrajectorySeed> & result);

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
