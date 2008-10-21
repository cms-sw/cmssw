#ifndef RecoMuon_MuonSeedGenerator_MuonSeedFromRecHits_H
#define RecoMuon_MuonSeedGenerator_MuonSeedFromRecHits_H

/**  \class MuonSeedFromRecHits
 *
 *  \author A. Vitelli - INFN Torino
 *  
 *  \author porting R.Bellan - INFN Torino
 *
 *   Generate a seed starting from a list of RecHits 
 *   make use of TrajectorySeed from CommonDet
 *
 */

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include <vector>
class MuonSeedPtExtractor;


namespace edm {class EventSetup;}

class MuonSeedFromRecHits 
{
public:
  MuonSeedFromRecHits();
  virtual ~MuonSeedFromRecHits() {}

  void setBField(const MagneticField * field) {theField = field;}
  void setPtExtractor(const MuonSeedPtExtractor * extractor) {thePtExtractor = extractor;}

  void add(MuonTransientTrackingRecHit::MuonRecHitPointer hit) { theRhits.push_back(hit); }
  MuonTransientTrackingRecHit::ConstMuonRecHitPointer firstRecHit() const { return theRhits.front(); }
  unsigned int nrhit() const { return  theRhits.size(); }
  void clear() {theRhits.clear();}

  TrajectorySeed createSeed(float ptmean, float sptmean,
			    MuonTransientTrackingRecHit::ConstMuonRecHitPointer last) const;
  
  protected:
  friend class MuonSeedFinder;
  typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
  typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
  typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;

  protected:
  MuonTransientTrackingRecHit::MuonRecHitContainer theRhits;
  const MagneticField * theField;
  const MuonSeedPtExtractor * thePtExtractor;

};

#endif
