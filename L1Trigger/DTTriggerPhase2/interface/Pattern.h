#ifndef L1Trigger_DTTriggerPhase2_Pattern_h
#define L1Trigger_DTTriggerPhase2_Pattern_h

#include <tuple>
#include <vector>
#include <iostream>

//Typedef for refHits organized as [SL, Cell, Laterality]. Using integers is an overkill for something that only needs 3, 7 and 2 bits.
typedef std::tuple<int, int, int> RefPatternHit;
//Another overkill typing for the pattern identifier [SLUp, SLDown, ChambedUp-ChamberDown], only need 3, 3 and 5 bits
typedef std::tuple<int, int, int> PatternIdentifier;

class Pattern {
  //A pattern is a seed plus a set of hits. Translational simmetry is used to translate it across all posible recoHits in the lower (upper layer) and check for pattern hit matches of recohits.
public:
  //Constructors and destructors
  Pattern();
  Pattern(RefPatternHit seedUp, RefPatternHit seedDown);
  Pattern(int SL1, int SL2, int diff);
  virtual ~Pattern();

  //Adding hits to the pattern
  void addHit(RefPatternHit hit);
  //Given the up and down seeding hits check if a given hit is in the pattern. Returns -1 for left laterality, +1 for right laterality, 0 if undecided and -999 if not in the pattern
  int latHitIn(int slId, int chId, int allowedVariance);

  //When comparing with a given set of hits we need to set up at least one of those two to compute the translation
  void setHitUp(int chIdUp) { recoseedUp_ = chIdUp; }
  void setHitDown(int chIdDown) { recoseedDown_ = chIdDown; }

  //Get methods
  PatternIdentifier id() { return id_; }
  int SL1() { return std::get<0>(id_); }
  int SL2() { return std::get<1>(id_); }
  int Diff() { return std::get<2>(id_); }
  std::vector<RefPatternHit> genHits() { return genHits_; }

  //Printing
  friend std::ostream &operator<<(std::ostream &out, Pattern &p);

private:
  //Generated seeds
  RefPatternHit seedUp_;
  RefPatternHit seedDown_;
  //Generated hits
  std::vector<RefPatternHit> genHits_;
  //Pattern is classified in terms of SL + chamber differences to profit from translational invariance
  PatternIdentifier id_;
  //Generated seeds + hits translated to a given seed pair
  int recoseedUp_;
  int recoseedDown_;
  bool debug_ = false;
};

#endif
