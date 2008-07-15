#ifndef DQMOFFLINE_TRIGGER_CUTVALUES
#define DQMOFFLINE_TRIGGER_CUTVALUES


//This is a simple struct to hold the values of a particular set of cuts
//may end up being promoted to a class


#include <iostream>
#include <string>

struct CutValues  {
  //identifier
  std::string id; //id in a human readable format (unused)
  int idWord; //allows fast comparisions, really this tells you all about the cuts (unused)
  int validEleTypes; //what electrons types are valid (eg barrel, endcap)
  int cutMask; //allows cuts to be turned off/on

  //cut values
  //kinmatic cuts
  float minEtCut;
  float minEtaCut;
  float maxEtaCut;
  bool rejectCracks;

  //id cuts
  float minEpInCut;
  float maxEpInCut;
  float epInReleaseEtCut;
  float maxDEtaInCut;
  float maxDPhiInCut;
  float maxHademCut; 
  float maxHadCell2ConstCut; 
  float maxHadCell2GradCut;
  float minEpOutCut;
  float maxEpOutCut;
  float maxDPhiOutCut;
  float minInvEInvPCut;
  float maxInvEInvPCut;
  float minBremFracCut;
  float minE9E25Cut;
  float minSigmaEtaEtaCut;
  float maxSigmaEtaEtaCut;
  float minSigmaPhiPhiCut;
  float maxSigmaPhiPhiCut;
  
  //isol cut values
  float minIsolEmConstCut;
  float isolEmGradCut; 
  float minIsolEmRecHitConstCut;
  float isolEmRecHitGradCut;
  float minIsolHadConstCut;
  float isolHadGradCut;
  float minIsolHadDepth2ConstCut;
  float isolHadDepth2GradCut;
  float minIsolPtTrksConstCut;
  float isolPtTrksGradCut;
  int minIsolNrTrksConstCut;

  float minIsolEmHadDepth1ConstCut;
  float isolEmHadDepth1GradCut;
  



  CutValues(){}
  virtual ~CutValues(){}

  void setEBHighNrgy(int inputCutMask=~0x0);
  void setEEHighNrgy(int inputCutMask=~0x0);
  void setEBPreSel(int inputCutMask=~0x0);
  void setEEPreSel(int inputCutMask=~0x0);
 
 

};


#endif
