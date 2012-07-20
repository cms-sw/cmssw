#include "DQMOffline/Trigger/interface/EgHLTEgCutValues.h"
#include "DQMOffline/Trigger/interface/EgHLTEgCutCodes.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace egHLT;

void EgCutValues::setup(const edm::ParameterSet& iConfig)
{
  cutMask = EgCutCodes::getCode(iConfig.getParameter<std::string>("cuts"));  
  //kinematic and fiduicual cuts
  minEt = iConfig.getParameter<double>("minEt");
  minEta = iConfig.getParameter<double>("minEta");
  maxEta = iConfig.getParameter<double>("maxEta");
  //track cuts
  maxDEtaIn = iConfig.getParameter<double>("maxDEtaIn");
  maxDPhiIn = iConfig.getParameter<double>("maxDPhiIn"); 
  maxInvEInvP = iConfig.getParameter<double>("maxInvEInvP");
  //super cluster cuts
  maxHadem = iConfig.getParameter<double>("maxHadem");
  maxHadEnergy = iConfig.getParameter<double>("maxHadEnergy");
  maxSigmaIEtaIEta = iConfig.getParameter<double>("maxSigmaIEtaIEta"); 
  maxSigmaEtaEta = iConfig.getParameter<double>("maxSigmaEtaEta"); 
  //----Morse------
  minR9 = iConfig.getParameter<double>("minR9");
  maxR9 = iConfig.getParameter<double>("maxR9");
  //---------------

  //std isolation cuts
  isolEmConstTerm = iConfig.getParameter<double>("isolEmConstTerm");
  isolEmGradTerm = iConfig.getParameter<double>("isolEmGradTerm");
  isolEmGradStart = iConfig.getParameter<double>("isolEmGradStart");
  
  isolHadConstTerm = iConfig.getParameter<double>("isolHadConstTerm");
  isolHadGradTerm = iConfig.getParameter<double>("isolHadGradTerm");
  isolHadGradStart = iConfig.getParameter<double>("isolHadGradStart");
  
  isolPtTrksConstTerm = iConfig.getParameter<double>("isolPtTrksConstTerm");
  isolPtTrksGradTerm = iConfig.getParameter<double>("isolPtTrksGradTerm");
  isolPtTrksGradStart = iConfig.getParameter<double>("isolPtTrksGradStart");
  
  isolNrTrksConstTerm = iConfig.getParameter<int>("isolNrTrksConstTerm");

  //hlt isolation cuts
  maxHLTIsolTrksEle = iConfig.getParameter<double>("maxHLTIsolTrksEle");  
  maxHLTIsolTrksEleOverPt = iConfig.getParameter<double>("maxHLTIsolTrksEleOverPt");  
  maxHLTIsolTrksEleOverPt2 = iConfig.getParameter<double>("maxHLTIsolTrksEleOverPt2"); 
  maxHLTIsolTrksPho = iConfig.getParameter<double>("maxHLTIsolTrksPho"); 
  maxHLTIsolTrksPhoOverPt = iConfig.getParameter<double>("maxHLTIsolTrksPhoOverPt");  
  maxHLTIsolTrksPhoOverPt2 = iConfig.getParameter<double>("maxHLTIsolTrksPhoOverPt2"); 
  maxHLTIsolHad = iConfig.getParameter<double>("maxHLTIsolHad");
  maxHLTIsolHadOverEt = iConfig.getParameter<double>("maxHLTIsolHadOverEt");
  maxHLTIsolHadOverEt2 = iConfig.getParameter<double>("maxHLTIsolHadOverEt2");  

  maxHLTIsolEm = iConfig.getParameter<double>("maxHLTIsolEm");
  maxHLTIsolEmOverEt = iConfig.getParameter<double>("maxHLTIsolEmOverEt");
  maxHLTIsolEmOverEt2 = iConfig.getParameter<double>("maxHLTIsolEmOverEt2");  

  minCTFTrkOuterRadius = iConfig.getParameter<double>("minCTFTrkOuterRadius");
  maxCTFTrkInnerRadius = iConfig.getParameter<double>("maxCTFTrkInnerRadius");
  minNrCTFTrkHits = iConfig.getParameter<int>("minNrCTFTrkHits");
  maxNrCTFTrkHitsLost = iConfig.getParameter<int>("maxNrCTFTrkHitsLost");
  maxCTFTrkChi2NDof = iConfig.getParameter<double>("maxCTFTrkChi2NDof");
  requirePixelHitsIfOuterInOuter = iConfig.getParameter<bool>("requirePixelHitsIfOuterInOuter");
  
  maxHLTDEtaIn = iConfig.getParameter<double>("maxHLTDEtaIn");
  maxHLTDPhiIn = iConfig.getParameter<double>("maxHLTDPhiIn"); 
  maxHLTInvEInvP = iConfig.getParameter<double>("maxHLTInvEInvP");
  
}

