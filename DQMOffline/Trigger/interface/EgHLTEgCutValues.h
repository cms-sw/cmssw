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
    double maxHadem;
    double maxSigmaIEtaIEta;
    double minR9;
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
    double maxHLTIsolTrksPho;
    double maxHLTIsolHad;
    double maxHLTIsolHadOverEt;
    double maxHLTIsolHadOverEt2;
    
    EgCutValues(){}
    explicit EgCutValues(const edm::ParameterSet& iConfig){setup(iConfig);}
    void setup(const edm::ParameterSet& iConfig);

  };
}

#endif
