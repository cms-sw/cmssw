#ifndef RecoMuon_TrackerSeedGenerator_TSGSmart_H
#define RecoMuon_TrackerSeedGenerator_TSGSmart_H

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SeedGeneratorFromRegionHits;
class TrackingRegion;


class TSGSmart : public TrackerSeedGenerator {

public:
  TSGSmart(const edm::ParameterSet &pset, edm::ConsumesCollector& iC);

  ~TSGSmart() override;

private:
  void run(TrajectorySeedCollection &seeds, 
      const edm::Event &ev, const edm::EventSetup &es, const TrackingRegion& region) override;

private:
  edm::ParameterSet theConfig;
  SeedGeneratorFromRegionHits * thePairGenerator; 
  SeedGeneratorFromRegionHits * theTripletGenerator; 
  SeedGeneratorFromRegionHits * theMixedGenerator; 

  double theEtaBound;
};


#endif 
