#ifndef MuonSeedGenerator_MuonSeedOrcaPatternRecognition_h
#define MuonSeedGenerator_MuonSeedOrcaPatternRecognition_h

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedVPatternRecognition.h"

class MuonSeedOrcaPatternRecognition : public MuonSeedVPatternRecognition
{
public:
  explicit MuonSeedOrcaPatternRecognition(const edm::ParameterSet & pset); 

  void produce(edm::Event& event, const edm::EventSetup& eSetup,
               std::vector<MuonRecHitContainer> & result);

private:

  // aalocates a zeroed array of a given size
  bool * zero(unsigned listSize);

  void complete(MuonRecHitContainer& seedSegments,
                const MuonRecHitContainer &recHits, bool* used=0) const;


};

#endif

