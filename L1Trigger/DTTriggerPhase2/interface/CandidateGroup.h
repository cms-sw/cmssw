#ifndef L1Trigger_DTTriggerPhase2_CandidateGroup_h
#define L1Trigger_DTTriggerPhase2_CandidateGroup_h

#include <tuple>
#include <vector>
#include <bitset>
#include <iostream>
//Note that this being in src is probably wrong but I don't want to crash everone's code for being picky
#include "L1Trigger/DTTriggerPhase2/interface/DTprimitive.h"
#include "L1Trigger/DTTriggerPhase2/interface/DTPattern.h"

class CandidateGroup {
public:
  //Constructors and destructors
  CandidateGroup();
  CandidateGroup(DTPattern* p);
  ~CandidateGroup();

  //Hit operation procedures
  void addHit(DTPrimitive dthit, int lay, bool isGood);
  void removeHit(DTPrimitive dthit);

  //Get Methods
  int candId() const { return candId_; };
  int nhits() const { return nhits_; };
  int nisGood() const { return nisGood_; };
  int nLayerhits() const { return nLayerhits_; };
  int nLayerUp() const { return nLayerUp_; };
  int nLayerDown() const { return nLayerDown_; };
  std::vector<DTPrimitive> candHits() { return candHits_; }; // WHAT TO DO WITH THIS!!! 
  std::bitset<8> quality() const { return quality_; };
  const DTPattern* pattern() const { return pattern_; }; // WHAT TO DO WITH THIS!!!

  //Set Methods
  void setCandId(int cId) { candId_ = cId; };

  //Pattern rankers
  bool operator>(const CandidateGroup& cOther) const;
  bool operator==(const CandidateGroup& cOther) const;

  //Just so we don't need need std::pow for powers
  int power(int a, int n) {
    int res = 1;
    while (n) {
      if (n & 1)
        res *= a;
      a *= a;
      n >>= 1;
    }
    return res;
  }

private:
  std::vector<DTPrimitive> candHits_;
  std::bitset<8> quality_;
  int nhits_;
  int nLayerhits_;
  int nLayerUp_;
  int nLayerDown_;
  int nisGood_;
  DTPattern* pattern_;
  int candId_;
};

#endif
