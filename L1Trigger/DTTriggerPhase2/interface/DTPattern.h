#ifndef L1Trigger_DTTriggerPhase2_DTPattern_h
#define L1Trigger_DTTriggerPhase2_DTPattern_h

#include <tuple>
#include <vector>
#include <iostream>

// Typedef for refHits organized as [SL, Cell, Laterality]. Using integers is an
// overkill for something that only needs 3, 7 and 2 bits.
typedef std::tuple<int, int, int> RefDTPatternHit;
// Another overkill typing for the pattern identifier
// [SLUp, SLDown, ChambedUp-ChamberDown], only need 3, 3 and 5 bits
typedef std::tuple<int, int, int> DTPatternIdentifier;

class DTPattern {
  // A pattern is a seed plus a set of hits. Translational simmetry is used to
  // translate it across all posible recoHits in the lower (upper layer) and
  // check for pattern hit matches of recohits.
public:
  //Constructors and destructors
  DTPattern();
  DTPattern(RefDTPatternHit seedUp, RefDTPatternHit seedDown);
  DTPattern(int SL1, int SL2, int diff);
  virtual ~DTPattern();

  //Adding hits to the pattern
  void addHit(RefDTPatternHit hit);
  // Given the up and down seeding hits check if a given hit is in the pattern.
  // Returns -1 for left laterality, +1 for right laterality, 0 if undecided
  // and -999 if not in the pattern
  int latHitIn(int slId, int chId, int allowedVariance) const;

  // When comparing with a given set of hits we need to set up at least one of
  // those two to compute the translation
  void setHitUp(int chIdUp) { recoseedUp_ = chIdUp; }
  void setHitDown(int chIdDown) { recoseedDown_ = chIdDown; }

  //Get methods
  DTPatternIdentifier id() const { return id_; }
  int sl1() const { return std::get<0>(id_); }
  int sl2() const { return std::get<1>(id_); }
  int diff() const { return std::get<2>(id_); }
  const std::vector<RefDTPatternHit> &genHits() const { return genHits_; }

  //Printing
  friend std::ostream &operator<<(std::ostream &out, DTPattern const &p);

private:
  //Generated seeds
  RefDTPatternHit seedUp_;
  RefDTPatternHit seedDown_;
  // Generated hits
  std::vector<RefDTPatternHit> genHits_;
  // Pattern is classified in terms of SL + chamber differences to profit from
  // translational invariance
  DTPatternIdentifier id_;
  //Generated seeds + hits translated to a given seed pair
  int recoseedUp_;
  int recoseedDown_;
  const bool debug_ = false;
};

#endif
