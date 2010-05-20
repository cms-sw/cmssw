#ifndef MuonSeedGenerator_MuonSeedOrcaPatternRecognition_h
#define MuonSeedGenerator_MuonSeedOrcaPatternRecognition_h

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedVPatternRecognition.h"


class MuonSeedOrcaPatternRecognition : public MuonSeedVPatternRecognition
{
public:
  explicit MuonSeedOrcaPatternRecognition(const edm::ParameterSet & pset); 

  void produce(const edm::Event& event, const edm::EventSetup& eSetup,
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

  MuonRecHitPointer
  bestMatch(const ConstMuonRecHitPointer & first,  MuonRecHitContainer & good_rhit) const;
  // some score to measure how well the two hits match
  double discriminator(const ConstMuonRecHitPointer & first, 
                       MuonRecHitPointer & other) const;
  // see if it's OK to add
  bool check(const MuonRecHitContainer & segments);
  bool isCrack(const ConstMuonRecHitPointer & segment) const;
  void rememberCrackSegments(const MuonRecHitContainer & segments,
                             MuonRecHitContainer & crackSegments) const;

  void dumpLayer(const char * name, const MuonRecHitContainer & segments) const;

  /// apply some cuts to segments before using them
  MuonRecHitContainer filterSegments(const MuonRecHitContainer & segments, double dThetaCut) const;
  void filterOverlappingChambers(MuonRecHitContainer & segments) const;
  bool isME1A(const ConstMuonRecHitPointer & segment) const;
  int countHits(const MuonRecHitPointer & segment) const;
  // can mark other ME1A as used if one is.  
  void markAsUsed(int nr, const MuonRecHitContainer &recHits, bool* used) const;
  std::vector<double> theCrackEtas;
  double theCrackWindow;
};

#endif

