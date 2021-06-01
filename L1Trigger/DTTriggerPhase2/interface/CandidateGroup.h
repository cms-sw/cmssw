#ifndef L1Trigger_DTTriggerPhase2_CandidateGroup_h
#define L1Trigger_DTTriggerPhase2_CandidateGroup_h

#include <tuple>
#include <vector>
#include <bitset>
#include <iostream>
#include <complex>

#include "L1Trigger/DTTriggerPhase2/interface/DTprimitive.h"
#include "L1Trigger/DTTriggerPhase2/interface/DTPattern.h"

namespace dtbayesam {

  typedef std::bitset<8> qualitybits;

  typedef std::shared_ptr<DTPattern> DTPatternPtr;
  typedef std::vector<DTPatternPtr> DTPatternPtrs;

  class CandidateGroup {
  public:
    //Constructors and destructors
    CandidateGroup();
    CandidateGroup(DTPatternPtr p);
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
    DTPrimitivePtrs candHits() const { return candHits_; };
    qualitybits quality() const { return quality_; };
    const DTPatternPtr pattern() const { return pattern_; };

    //Set Methods
    void setCandId(int cId) { candId_ = cId; };

    //Pattern rankers
    bool operator>(const CandidateGroup& cOther) const;
    bool operator==(const CandidateGroup& cOther) const;

  private:
    DTPrimitivePtrs candHits_;
    qualitybits quality_;
    int nhits_;
    int nLayerhits_;
    int nLayerUp_;
    int nLayerDown_;
    int nisGood_;
    DTPatternPtr pattern_;
    int candId_;
  };

  typedef std::shared_ptr<CandidateGroup> CandidateGroupPtr;
  typedef std::vector<CandidateGroupPtr> CandidateGroupPtrs;
};  // namespace dtbayesam

#endif
