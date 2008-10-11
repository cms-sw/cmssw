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

  void endcapPatterns(
    const MuonRecHitContainer & me11, const MuonRecHitContainer & me12,
    const MuonRecHitContainer & me2,  const MuonRecHitContainer & me3,
    const MuonRecHitContainer & me4,  const  MuonRecHitContainer & mb1,
    const MuonRecHitContainer & mb2,  const  MuonRecHitContainer & mb3,
    bool * MB1, bool * MB2, bool * MB3,
    std::vector<MuonRecHitContainer> & result);

  void complete(MuonRecHitContainer& seedSegments,
                const MuonRecHitContainer &recHits, bool* used=0) const;

  // see if it's OK to add
  bool check(const MuonRecHitContainer & segments);
  void rememberCrackSegments(const MuonRecHitContainer & segments,
                             MuonRecHitContainer & crackSegments) const;

  std::vector<double> theCrackEtas;
  double theCrackWindow;
};

#endif

