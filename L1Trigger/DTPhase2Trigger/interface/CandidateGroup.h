#ifndef CandidateGroup_H
#define CandidateGroup_H

#include <tuple>
#include <vector>
#include <bitset>
#include <iostream>
//Note that this being in src is probably wrong but I don't want to crash everone's code for being picky
#include "L1Trigger/DTPhase2Trigger/interface/dtprimitive.h"
#include "L1Trigger/DTPhase2Trigger/interface/Pattern.h"

class CandidateGroup {  
  public:
    //Constructors and destructors
    CandidateGroup();
    CandidateGroup(Pattern* p);
    virtual ~CandidateGroup();

    //Hit operation procedures
    void AddHit(DTPrimitive dthit, int lay, bool isGood);
    void RemoveHit(DTPrimitive dthit);

    //Get Methods
    int getCandId() const {return candId;};
    int getNhits() const {return nhits;};
    int getNisGood() const {return nisGood;};
    int getNLayerhits() const {return nLayerhits;};
    int getNLayerUp() const   {return nLayerUp;};
    int getNLayerDown() const {return nLayerDown;};
    std::vector<DTPrimitive> getcandHits() const {return candHits;};
    void setCandId(int cId) {candId = cId;};
    Pattern* getPattern() const {return pattern;};

    //Set Methods
    std::bitset<8> getQuality() {return quality;};

    //Pattern rankers
    bool operator> (const CandidateGroup& cOther) const;
    bool operator== (const CandidateGroup& cOther) const;

    //Just so we don't need need std::pow for powers
    int power(int a, int n) {
      int res = 1;
      while (n) {
        if (n & 1) res *= a;
        a *= a;
        n >>= 1;}
    return res;}

  private:
    std::vector<DTPrimitive> candHits;
    std::bitset<8> quality;
    int nhits;
    int nLayerhits;
    int nLayerUp;
    int nLayerDown;
    int nisGood;
    Pattern* pattern;
    int candId;
};

#endif
