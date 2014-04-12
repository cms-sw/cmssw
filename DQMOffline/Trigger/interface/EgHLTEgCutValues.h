#ifndef DQMOFFLINE_TRIGGER_EGHLTEGCUTVALUES
#define DQMOFFLINE_TRIGGER_EGHLTEGCUTVALUES


//This is a simple struct to hold the values of a particular set of cuts


#include <iostream>
#include <string>

namespace edm{
  class ParameterSet;
}
namespace egHLT {
  struct EgCutValues  {
  public:
    int cutMask;  
    //kinematic and fiduicual cuts
    double minEt;
    double minEta;
    double maxEta;
    //track cuts
    double maxDEtaIn;
    double maxDPhiIn;
    double maxInvEInvP;
    //super cluster cuts
    double maxHadem; //h/e
    double maxHadEnergy; //max h of h/e
    double maxSigmaIEtaIEta;   
    double maxSigmaEtaEta;
    double minR9;
    //--Morse----
    double maxR9;
    //--------
     //std isolation cuts
    double isolEmConstTerm;
    double isolEmGradTerm;
    double isolEmGradStart;
    double isolHadConstTerm;
    double isolHadGradTerm;
    double isolHadGradStart;
    double isolPtTrksConstTerm;
    double isolPtTrksGradTerm;
    double isolPtTrksGradStart;
    int isolNrTrksConstTerm;
    //hlt isolation cuts
    double maxHLTIsolTrksEle; 
    double maxHLTIsolTrksEleOverPt; 
    double maxHLTIsolTrksEleOverPt2;
    double maxHLTIsolTrksPho; 
    double maxHLTIsolTrksPhoOverPt; 
    double maxHLTIsolTrksPhoOverPt2;
    double maxHLTIsolHad;
    double maxHLTIsolHadOverEt;
    double maxHLTIsolHadOverEt2;
    double maxHLTIsolEm;
    double maxHLTIsolEmOverEt;
    double maxHLTIsolEmOverEt2;
    //ctf track quality cuts
    double minCTFTrkOuterRadius;
    double maxCTFTrkInnerRadius;
    double minNrCTFTrkHits;
    double maxNrCTFTrkHitsLost;
    double maxCTFTrkChi2NDof;
    bool requirePixelHitsIfOuterInOuter;
    //hlt track variable cuts
    double maxHLTDEtaIn;
    double maxHLTDPhiIn;
    double maxHLTInvEInvP;

    EgCutValues(){}
    explicit EgCutValues(const edm::ParameterSet& iConfig){setup(iConfig);}
    void setup(const edm::ParameterSet& iConfig);

  };
}

#endif
