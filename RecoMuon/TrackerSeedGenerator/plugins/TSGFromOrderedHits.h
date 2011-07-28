#ifndef RecoMuon_TrackerSeedGenerator_TSGFromOrderedHits_H
#define RecoMuon_TrackerSeedGenerator_TSGFromOrderedHits_H

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/RunID.h"

class SeedGeneratorFromRegionHits;
class TrackingRegion;


class TSGFromOrderedHits : public TrackerSeedGenerator {

public:
  TSGFromOrderedHits(const edm::ParameterSet &pset);

  virtual ~TSGFromOrderedHits();

private:
  virtual void run(TrajectorySeedCollection &seeds, 
      const edm::Event &ev, const edm::EventSetup &es, const TrackingRegion& region);

private:
  void init();
  edm::RunNumber_t theLastRun;
  edm::ParameterSet theConfig;
  SeedGeneratorFromRegionHits * theGenerator; 
};


#endif 
