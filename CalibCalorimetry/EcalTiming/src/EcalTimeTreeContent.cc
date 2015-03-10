#include "CalibCalorimetry/EcalTiming/interface/EcalTimeTreeContent.h"

bool EcalTimeTreeContent::trgVariables = false;
bool EcalTimeTreeContent::muonVariables = false;
bool EcalTimeTreeContent::ecalVariables = true;
bool EcalTimeTreeContent::ecalShapeVariables = false;
bool EcalTimeTreeContent::hcalVariables = false;
bool EcalTimeTreeContent::tkAssVariables = false;
bool EcalTimeTreeContent::tpgVariables = false;
bool EcalTimeTreeContent::l1Variables = true;



void setBranchAddresses(TTree* chain, EcalTimeTreeContent& treeVars)
{
  chain -> SetBranchAddress("runId",       &treeVars.runId);
  chain -> SetBranchAddress("lumiSection", &treeVars.lumiSection);
  chain -> SetBranchAddress("unixTime",    &treeVars.unixTime);
  chain -> SetBranchAddress("orbit",       &treeVars.orbit);
  chain -> SetBranchAddress("bx",          &treeVars.bx);

  chain -> SetBranchAddress("eventId",       &treeVars.eventId);
  chain -> SetBranchAddress("eventNaiveId",  &treeVars.eventNaiveId);
  chain -> SetBranchAddress("timeStampLow",  &treeVars.timeStampLow);
  chain -> SetBranchAddress("timeStampHigh", &treeVars.timeStampHigh);
  
  

  ///*  
  // TRG VARIABLES  
  if(EcalTimeTreeContent::trgVariables)
  {    
    // trigger variables
    chain -> SetBranchAddress("isRPCL1",  &treeVars.isRPCL1);
    chain -> SetBranchAddress("isDTL1",   &treeVars.isDTL1);
    chain -> SetBranchAddress("isCSCL1",  &treeVars.isCSCL1);
    chain -> SetBranchAddress("isECALL1", &treeVars.isECALL1);
    chain -> SetBranchAddress("isHCALL1", &treeVars.isHCALL1);

    chain -> SetBranchAddress("isRPCL1Bx",  treeVars.isRPCL1Bx); 
    chain -> SetBranchAddress("isDTL1Bx",   treeVars.isDTL1Bx); 
    chain -> SetBranchAddress("isCSCL1Bx",  treeVars.isCSCL1Bx); 
    chain -> SetBranchAddress("isECALL1Bx", treeVars.isECALL1Bx); 
    chain -> SetBranchAddress("isHCALL1Bx", treeVars.isHCALL1Bx); 
  } // TRG VARIABLES
  //*/  


  
  ///*
  // ECAL VARIABLES  
  if(EcalTimeTreeContent::ecalVariables)
  {    
    // supercluster variables
    chain -> SetBranchAddress("nSuperClusters",       &treeVars.nSuperClusters);
    chain -> SetBranchAddress("nBarrelSuperClusters", &treeVars.nBarrelSuperClusters);
    chain -> SetBranchAddress("nEndcapSuperClusters", &treeVars.nEndcapSuperClusters);
    chain -> SetBranchAddress("superClusterRawEnergy", treeVars.superClusterRawEnergy);
    chain -> SetBranchAddress("superClusterPhiWidth",  treeVars.superClusterPhiWidth);
    chain -> SetBranchAddress("superClusterEtaWidth",  treeVars.superClusterEtaWidth);
    chain -> SetBranchAddress("superClusterPhi",       treeVars.superClusterPhi);
    chain -> SetBranchAddress("superClusterEta",       treeVars.superClusterEta);
    chain -> SetBranchAddress("superClusterX",         treeVars.superClusterX);
    chain -> SetBranchAddress("superClusterY",         treeVars.superClusterY);
    chain -> SetBranchAddress("superClusterZ",         treeVars.superClusterZ);
    chain -> SetBranchAddress("superClusterVertexX",         treeVars.superClusterVertexX);
    chain -> SetBranchAddress("superClusterVertexY",         treeVars.superClusterVertexY);
    chain -> SetBranchAddress("superClusterVertexZ",         treeVars.superClusterVertexZ);

    chain -> SetBranchAddress("nClustersInSuperCluster",    treeVars.nClustersInSuperCluster);
  
  
    // basic cluster variables
    chain -> SetBranchAddress("nClusters",              &treeVars.nClusters);
    chain -> SetBranchAddress("clusterEnergy",           treeVars.clusterEnergy);
    chain -> SetBranchAddress("clusterTransverseEnergy", treeVars.clusterTransverseEnergy);
    chain -> SetBranchAddress("clusterE1",               treeVars.clusterE1);
    chain -> SetBranchAddress("clusterE2",               treeVars.clusterE2);
    chain -> SetBranchAddress("clusterTime",             treeVars.clusterTime);
    chain -> SetBranchAddress("clusterPhi",              treeVars.clusterPhi);
    chain -> SetBranchAddress("clusterEta",              treeVars.clusterEta);
    chain -> SetBranchAddress("clusterXtals",            treeVars.clusterXtals);
    chain -> SetBranchAddress("clusterXtalsAbove3Sigma", treeVars.clusterXtalsAbove3Sigma);
    chain -> SetBranchAddress("clusterMaxId",            treeVars.clusterMaxId);
    chain -> SetBranchAddress("cluster2ndId",            treeVars.cluster2ndId);
    chain -> SetBranchAddress("nXtalsInCluster",    treeVars.nXtalsInCluster);
  
  
    // vertex variables
    chain -> SetBranchAddress("nVertices",         &treeVars.nVertices);
    chain -> SetBranchAddress("vtxNTracks",       treeVars.vtxNTracks);
    chain -> SetBranchAddress("vtxChi2",          treeVars.vtxChi2);
    chain -> SetBranchAddress("vtxNdof",          treeVars.vtxNdof);
    chain -> SetBranchAddress("vtxX",             treeVars.vtxX);
    chain -> SetBranchAddress("vtxDx",            treeVars.vtxDx);
    chain -> SetBranchAddress("vtxY",             treeVars.vtxY);
    chain -> SetBranchAddress("vtxDy",            treeVars.vtxDy);
    chain -> SetBranchAddress("vtxZ",             treeVars.vtxZ);
    chain -> SetBranchAddress("vtxDz",            treeVars.vtxDz);


    // xtal variables inside a cluster
    chain -> SetBranchAddress("xtalInBCHashedIndex", treeVars.xtalInBCHashedIndex);
    chain -> SetBranchAddress("xtalInBCIEta", treeVars.xtalInBCIEta);
    chain -> SetBranchAddress("xtalInBCIPhi", treeVars.xtalInBCIPhi);
    chain -> SetBranchAddress("xtalInBCEta", treeVars.xtalInBCEta);
    chain -> SetBranchAddress("xtalInBCPhi", treeVars.xtalInBCPhi);
    chain -> SetBranchAddress("xtalInBCIx", treeVars.xtalInBCIx);
    chain -> SetBranchAddress("xtalInBCIy", treeVars.xtalInBCIy);
    chain -> SetBranchAddress("xtalInBCFlag", treeVars.xtalInBCFlag);
    chain -> SetBranchAddress("xtalInBCEnergy", treeVars.xtalInBCEnergy);
    chain -> SetBranchAddress("xtalInBCTime", treeVars.xtalInBCTime);
    chain -> SetBranchAddress("xtalInBCTimeErr", treeVars.xtalInBCTimeErr);
    chain -> SetBranchAddress("xtalInBCAmplitudeADC", treeVars.xtalInBCAmplitudeADC);
    chain -> SetBranchAddress("xtalInBCChi2", treeVars.xtalInBCChi2);
    chain -> SetBranchAddress("xtalInBCOutOfTimeChi2", treeVars.xtalInBCOutOfTimeChi2);
    chain -> SetBranchAddress("xtalInBCSwissCross", treeVars.xtalInBCSwissCross);

  } // ECAL VARIABLES
  //*/  
  

  ///*
  // ECAL VARIABLES  
  if(EcalTimeTreeContent::ecalShapeVariables)
  {    
    // clustershape variables  
    chain -> SetBranchAddress("clusterE2x2",       treeVars.clusterE2x2);
    chain -> SetBranchAddress("clusterE3x2",       treeVars.clusterE3x2);
    chain -> SetBranchAddress("clusterE3x3",       treeVars.clusterE3x3);
    chain -> SetBranchAddress("clusterE4x4",       treeVars.clusterE4x4);
    chain -> SetBranchAddress("clusterE5x5",       treeVars.clusterE5x5);
    chain -> SetBranchAddress("clusterE2x5Right",  treeVars.clusterE2x5Right);
    chain -> SetBranchAddress("clusterE2x5Left",   treeVars.clusterE2x5Left);
    chain -> SetBranchAddress("clusterE2x5Top",    treeVars.clusterE2x5Top);
    chain -> SetBranchAddress("clusterE2x5Bottom", treeVars.clusterE2x5Bottom);
    chain -> SetBranchAddress("clusterE3x2Ratio",  treeVars.clusterE3x2Ratio);
    chain -> SetBranchAddress("clusterCovPhiPhi",  treeVars.clusterCovPhiPhi);
    chain -> SetBranchAddress("clusterCovEtaEta",  treeVars.clusterCovEtaEta);
    chain -> SetBranchAddress("clusterCovEtaPhi",  treeVars.clusterCovEtaPhi);
    chain -> SetBranchAddress("clusterLat",        treeVars.clusterLat);
    chain -> SetBranchAddress("clusterPhiLat",     treeVars.clusterPhiLat);
    chain -> SetBranchAddress("clusterEtaLat",     treeVars.clusterEtaLat);
    chain -> SetBranchAddress("clusterZernike20",  treeVars.clusterZernike20);
    chain -> SetBranchAddress("clusterZernike42",  treeVars.clusterZernike42);
  } // ECAL VARIABLES
  //*/  



  ///*
  //HCAL variables
  if( EcalTimeTreeContent::hcalVariables ) 
  {  
    chain -> SetBranchAddress("hbNRecHits",   &treeVars.hbNRecHits);
    chain -> SetBranchAddress("hbRecHitDetId", treeVars.hbRecHitDetId);
    chain -> SetBranchAddress("hbRecHitEta",   treeVars.hbRecHitEta);
    chain -> SetBranchAddress("hbRecHitPhi",   treeVars.hbRecHitPhi);
    chain -> SetBranchAddress("hbRecHitE",     treeVars.hbRecHitE);
    chain -> SetBranchAddress("hbRecHitTime",  treeVars.hbRecHitTime);
    
    chain -> SetBranchAddress("nCaloTowers",         &treeVars.nCaloTowers);
    chain -> SetBranchAddress("caloTowerEmEnergy",    treeVars.caloTowerEmEnergy);
    chain -> SetBranchAddress("caloTowerHadEnergy",   treeVars.caloTowerHadEnergy);
    chain -> SetBranchAddress("caloTowerOuterEnergy", treeVars.caloTowerOuterEnergy);
    chain -> SetBranchAddress("caloTowerEmEta",       treeVars.caloTowerEmEta);
    chain -> SetBranchAddress("caloTowerEmPhi",       treeVars.caloTowerEmPhi);
    chain -> SetBranchAddress("caloTowerHadEta",      treeVars.caloTowerHadEta);
    chain -> SetBranchAddress("caloTowerHadPhi",      treeVars.caloTowerHadPhi);
  } //HCAL variables
  //*/
  

  
  ///*
  // MUON VARIABLES
  if(EcalTimeTreeContent::muonVariables)
  {  
    // muon variables
    chain -> SetBranchAddress("nRecoMuons",      &treeVars.nRecoMuons);
    chain -> SetBranchAddress("muonX",            treeVars.muonX);
    chain -> SetBranchAddress("muonY",            treeVars.muonY);
    chain -> SetBranchAddress("muonZ",            treeVars.muonZ);
    chain -> SetBranchAddress("muonPhi",          treeVars.muonPhi);
    chain -> SetBranchAddress("muonEta",          treeVars.muonEta);
    chain -> SetBranchAddress("muond0",           treeVars.muond0);
    chain -> SetBranchAddress("muondz",           treeVars.muondz);
    chain -> SetBranchAddress("muonPx",           treeVars.muonPx);
    chain -> SetBranchAddress("muonPy",           treeVars.muonPy);
    chain -> SetBranchAddress("muonPz",           treeVars.muonPz);
    chain -> SetBranchAddress("muonP",            treeVars.muonP);
    chain -> SetBranchAddress("muonPt",           treeVars.muonPt);
    chain -> SetBranchAddress("muonPtError",      treeVars.muonPtError);
    chain -> SetBranchAddress("muonCharge",       treeVars.muonCharge);
    chain -> SetBranchAddress("muonQOverP",       treeVars.muonQOverP);
    chain -> SetBranchAddress("muonQOverPError",  treeVars.muonQOverPError);
    chain -> SetBranchAddress("muonNChi2",        treeVars.muonNChi2);
    chain -> SetBranchAddress("muonNDof",         treeVars.muonNDof);
    chain -> SetBranchAddress("muonNHits",        treeVars.muonNHits);
  
    chain -> SetBranchAddress("muonInnerHitX",   treeVars.muonInnerHitX);
    chain -> SetBranchAddress("muonInnerHitY",   treeVars.muonInnerHitY);
    chain -> SetBranchAddress("muonInnerHitZ",   treeVars.muonInnerHitZ);
    chain -> SetBranchAddress("muonInnerHitPhi", treeVars.muonInnerHitPhi);
    chain -> SetBranchAddress("muonInnerHitEta", treeVars.muonInnerHitEta);
    chain -> SetBranchAddress("muonInnerHitPx",  treeVars.muonInnerHitPx);
    chain -> SetBranchAddress("muonInnerHitPy",  treeVars.muonInnerHitPy);
    chain -> SetBranchAddress("muonInnerHitPz",  treeVars.muonInnerHitPz);
    chain -> SetBranchAddress("muonInnerHitP",   treeVars.muonInnerHitP);
    chain -> SetBranchAddress("muonInnerHitPt",  treeVars.muonInnerHitPt);
  
    chain -> SetBranchAddress("muonOuterHitX",   treeVars.muonOuterHitX);
    chain -> SetBranchAddress("muonOuterHitY",   treeVars.muonOuterHitY);
    chain -> SetBranchAddress("muonOuterHitZ",   treeVars.muonOuterHitZ);
    chain -> SetBranchAddress("muonOuterHitPhi", treeVars.muonOuterHitPhi);
    chain -> SetBranchAddress("muonOuterHitEta", treeVars.muonOuterHitEta);
    chain -> SetBranchAddress("muonOuterHitPx",  treeVars.muonOuterHitPx);
    chain -> SetBranchAddress("muonOuterHitPy",  treeVars.muonOuterHitPy);
    chain -> SetBranchAddress("muonOuterHitPz",  treeVars.muonOuterHitPz);
    chain -> SetBranchAddress("muonOuterHitP",   treeVars.muonOuterHitP);
    chain -> SetBranchAddress("muonOuterHitPt",  treeVars.muonOuterHitPt);
  
    chain -> SetBranchAddress("muonInnTkInnerHitX",   treeVars.muonInnTkInnerHitX);
    chain -> SetBranchAddress("muonInnTkInnerHitY",   treeVars.muonInnTkInnerHitY);
    chain -> SetBranchAddress("muonInnTkInnerHitZ",   treeVars.muonInnTkInnerHitZ);
    chain -> SetBranchAddress("muonInnTkInnerHitPhi", treeVars.muonInnTkInnerHitPhi);
    chain -> SetBranchAddress("muonInnTkInnerHitEta", treeVars.muonInnTkInnerHitEta);
    chain -> SetBranchAddress("muonInnTkInnerHitPx",  treeVars.muonInnTkInnerHitPx);
    chain -> SetBranchAddress("muonInnTkInnerHitPy",  treeVars.muonInnTkInnerHitPy);
    chain -> SetBranchAddress("muonInnTkInnerHitPz",  treeVars.muonInnTkInnerHitPz);
    chain -> SetBranchAddress("muonInnTkInnerHitP",   treeVars.muonInnTkInnerHitP);
    chain -> SetBranchAddress("muonInnTkInnerHitPt",  treeVars.muonInnTkInnerHitPt);

    chain -> SetBranchAddress("muonInnTkOuterHitX",   treeVars.muonInnTkOuterHitX);
    chain -> SetBranchAddress("muonInnTkOuterHitY",   treeVars.muonInnTkOuterHitY);
    chain -> SetBranchAddress("muonInnTkOuterHitZ",   treeVars.muonInnTkOuterHitZ);
    chain -> SetBranchAddress("muonInnTkOuterHitPhi", treeVars.muonInnTkOuterHitPhi);
    chain -> SetBranchAddress("muonInnTkOuterHitEta", treeVars.muonInnTkOuterHitEta);
    chain -> SetBranchAddress("muonInnTkOuterHitPx",  treeVars.muonInnTkOuterHitPx);
    chain -> SetBranchAddress("muonInnTkOuterHitPy",  treeVars.muonInnTkOuterHitPy);
    chain -> SetBranchAddress("muonInnTkOuterHitPz",  treeVars.muonInnTkOuterHitPz);
    chain -> SetBranchAddress("muonInnTkOuterHitP",   treeVars.muonInnTkOuterHitP); 
    chain -> SetBranchAddress("muonInnTkOuterHitPt",  treeVars.muonInnTkOuterHitPt);
  
    chain -> SetBranchAddress("muonOutTkInnerHitX",   treeVars.muonOutTkInnerHitX);
    chain -> SetBranchAddress("muonOutTkInnerHitY",   treeVars.muonOutTkInnerHitY);
    chain -> SetBranchAddress("muonOutTkInnerHitZ",   treeVars.muonOutTkInnerHitZ);
    chain -> SetBranchAddress("muonOutTkInnerHitPhi", treeVars.muonOutTkInnerHitPhi);
    chain -> SetBranchAddress("muonOutTkInnerHitEta", treeVars.muonOutTkInnerHitEta);
    chain -> SetBranchAddress("muonOutTkInnerHitPx",  treeVars.muonOutTkInnerHitPx);
    chain -> SetBranchAddress("muonOutTkInnerHitPy",  treeVars.muonOutTkInnerHitPy);
    chain -> SetBranchAddress("muonOutTkInnerHitPz",  treeVars.muonOutTkInnerHitPz);
    chain -> SetBranchAddress("muonOutTkInnerHitP",   treeVars.muonOutTkInnerHitP);
    chain -> SetBranchAddress("muonOutTkInnerHitPt",  treeVars.muonOutTkInnerHitPt);

    chain -> SetBranchAddress("muonOutTkOuterHitX",   treeVars.muonOutTkOuterHitX);
    chain -> SetBranchAddress("muonOutTkOuterHitY",   treeVars.muonOutTkOuterHitY);
    chain -> SetBranchAddress("muonOutTkOuterHitZ",   treeVars.muonOutTkOuterHitZ);
    chain -> SetBranchAddress("muonOutTkOuterHitPhi", treeVars.muonOutTkOuterHitPhi);
    chain -> SetBranchAddress("muonOutTkOuterHitEta", treeVars.muonOutTkOuterHitEta);
    chain -> SetBranchAddress("muonOutTkOuterHitPx",  treeVars.muonOutTkOuterHitPx);
    chain -> SetBranchAddress("muonOutTkOuterHitPy",  treeVars.muonOutTkOuterHitPy);
    chain -> SetBranchAddress("muonOutTkOuterHitPz",  treeVars.muonOutTkOuterHitPz);
    chain -> SetBranchAddress("muonOutTkOuterHitP",   treeVars.muonOutTkOuterHitP); 
    chain -> SetBranchAddress("muonOutTkOuterHitPt",  treeVars.muonOutTkOuterHitPt);
  
    chain -> SetBranchAddress("muonLeg", treeVars.muonLeg);

    chain -> SetBranchAddress("muonTkLengthInEcalApprox",            treeVars.muonTkLengthInEcalApprox);
    chain -> SetBranchAddress("muonTkLengthInEcalDetail",            treeVars.muonTkLengthInEcalDetail);
    chain -> SetBranchAddress("muonTkLengthInEcalDetailCurved",      treeVars.muonTkLengthInEcalDetailCurved);
    chain -> SetBranchAddress("muonTkLengthInEcalDetailCurved_high", treeVars.muonTkLengthInEcalDetailCurved_high);
    chain -> SetBranchAddress("muonTkLengthInEcalDetailCurved_low",  treeVars.muonTkLengthInEcalDetailCurved_low);
    
    chain -> SetBranchAddress("muonTkInternalPointInEcalX", treeVars.muonTkInternalPointInEcalX);
    chain -> SetBranchAddress("muonTkInternalPointInEcalY", treeVars.muonTkInternalPointInEcalY);
    chain -> SetBranchAddress("muonTkInternalPointInEcalZ", treeVars.muonTkInternalPointInEcalZ);
    chain -> SetBranchAddress("muonTkExternalPointInEcalX", treeVars.muonTkExternalPointInEcalX);
    chain -> SetBranchAddress("muonTkExternalPointInEcalY", treeVars.muonTkExternalPointInEcalY);
    chain -> SetBranchAddress("muonTkExternalPointInEcalZ", treeVars.muonTkExternalPointInEcalZ);
    chain -> SetBranchAddress("muonTkInternalPointInEcalCurvedX", treeVars.muonTkInternalPointInEcalCurvedX);
    chain -> SetBranchAddress("muonTkInternalPointInEcalCurvedY", treeVars.muonTkInternalPointInEcalCurvedY);
    chain -> SetBranchAddress("muonTkInternalPointInEcalCurvedZ", treeVars.muonTkInternalPointInEcalCurvedZ);
    chain -> SetBranchAddress("muonTkExternalPointInEcalCurvedX", treeVars.muonTkExternalPointInEcalCurvedX);
    chain -> SetBranchAddress("muonTkExternalPointInEcalCurvedY", treeVars.muonTkExternalPointInEcalCurvedY);
    chain -> SetBranchAddress("muonTkExternalPointInEcalCurvedZ", treeVars.muonTkExternalPointInEcalCurvedZ);
    chain -> SetBranchAddress("muonTkInternalPointInEcalCurvedPx", treeVars.muonTkInternalPointInEcalCurvedPx);
    chain -> SetBranchAddress("muonTkInternalPointInEcalCurvedPy", treeVars.muonTkInternalPointInEcalCurvedPy);
    chain -> SetBranchAddress("muonTkInternalPointInEcalCurvedPz", treeVars.muonTkInternalPointInEcalCurvedPz);
    chain -> SetBranchAddress("muonTkExternalPointInEcalCurvedPx", treeVars.muonTkExternalPointInEcalCurvedPx);
    chain -> SetBranchAddress("muonTkExternalPointInEcalCurvedPy", treeVars.muonTkExternalPointInEcalCurvedPy);
    chain -> SetBranchAddress("muonTkExternalPointInEcalCurvedPz", treeVars.muonTkExternalPointInEcalCurvedPz);

    chain -> SetBranchAddress("nMuonCrossedXtals",                treeVars.nMuonCrossedXtals);
    chain -> SetBranchAddress("nMuonCrossedXtalsCurved",          treeVars.nMuonCrossedXtalsCurved);
    chain -> SetBranchAddress("muonCrossedXtalHashedIndex",       treeVars.muonCrossedXtalHashedIndex);
    chain -> SetBranchAddress("muonCrossedXtalHashedIndexCurved", treeVars.muonCrossedXtalHashedIndexCurved);
    chain -> SetBranchAddress("muonCrossedXtalTkLength",          treeVars.muonCrossedXtalTkLength);
    chain -> SetBranchAddress("muonCrossedXtalTkLengthCurved",    treeVars.muonCrossedXtalTkLengthCurved);
  } // MUON VARIABLES
  //*/  
 
  ///* 
  // TPG VARIABLES
  if(EcalTimeTreeContent::tpgVariables)
  {
    chain -> SetBranchAddress("tpgNTowers",         &treeVars.tpgNTowers);
    chain -> SetBranchAddress("tpgIEta",             treeVars.tpgIEta);
    chain -> SetBranchAddress("tpgIPhi",             treeVars.tpgIPhi);
    chain -> SetBranchAddress("tpgNbOfXtals",        treeVars.tpgNbOfXtals);
    chain -> SetBranchAddress("tpgEnRec",            treeVars.tpgEnRec);
    chain -> SetBranchAddress("tpgADC",              treeVars.tpgADC);
    chain -> SetBranchAddress("tpgNActiveTriggers", &treeVars.tpgNActiveTriggers);
    chain -> SetBranchAddress("tpgActiveTriggers",   treeVars.tpgActiveTriggers);
    
    chain -> SetBranchAddress("tpEmulNTowers", &treeVars.tpEmulNTowers);
    chain -> SetBranchAddress("tpEmulIEta",     treeVars.tpEmulIEta);
    chain -> SetBranchAddress("tpEmulIPhi",     treeVars.tpEmulIPhi);
    chain -> SetBranchAddress("tpEmulADC1",     treeVars.tpEmulADC1);
    chain -> SetBranchAddress("tpEmulADC2",     treeVars.tpEmulADC2);
    chain -> SetBranchAddress("tpEmulADC3",     treeVars.tpEmulADC3);
    chain -> SetBranchAddress("tpEmulADC4",     treeVars.tpEmulADC4);
    chain -> SetBranchAddress("tpEmulADC5",     treeVars.tpEmulADC5);
  } // TPG VARIABLES
  //*/



  ///*
  // L1 VARIABLES
  if(EcalTimeTreeContent::l1Variables)
  {
    chain -> SetBranchAddress("l1NActiveTriggers", &treeVars.l1NActiveTriggers);
    chain -> SetBranchAddress("l1ActiveTriggers",   treeVars.l1ActiveTriggers);
    chain -> SetBranchAddress("l1NActiveTechTriggers", &treeVars.l1NActiveTechTriggers);
    chain -> SetBranchAddress("l1ActiveTechTriggers", treeVars.l1ActiveTechTriggers);
    
    chain->SetBranchAddress("l1GtNEm",    &treeVars.l1GtNEm);
    chain->SetBranchAddress("l1GtEmBx",   treeVars.l1GtEmBx);
    chain->SetBranchAddress("l1GtEmIEta", treeVars.l1GtEmIEta);
    chain->SetBranchAddress("l1GtEmIPhi", treeVars.l1GtEmIPhi);
    chain->SetBranchAddress("l1GtEmEta", treeVars.l1GtEmEta);
    chain->SetBranchAddress("l1GtEmPhi", treeVars.l1GtEmPhi);
    chain->SetBranchAddress("l1GtEmRank", treeVars.l1GtEmRank);
    chain->SetBranchAddress("l1GtEmEt",   treeVars.l1GtEmEt);

    chain -> SetBranchAddress("l1NEmPartIso",           &treeVars.l1NEmPartIso);
    chain -> SetBranchAddress("l1EmPartIsoX",            treeVars.l1EmPartIsoX);
    chain -> SetBranchAddress("l1EmPartIsoY",            treeVars.l1EmPartIsoY);
    chain -> SetBranchAddress("l1EmPartIsoZ",            treeVars.l1EmPartIsoZ);
    chain -> SetBranchAddress("l1EmPartIsoEta",          treeVars.l1EmPartIsoEta);
    chain -> SetBranchAddress("l1EmPartIsoPhi",          treeVars.l1EmPartIsoPhi);
    chain -> SetBranchAddress("l1EmPartIsoPx",           treeVars.l1EmPartIsoPx);
    chain -> SetBranchAddress("l1EmPartIsoPy",           treeVars.l1EmPartIsoPy);
    chain -> SetBranchAddress("l1EmPartIsoPz",           treeVars.l1EmPartIsoPz);
    chain -> SetBranchAddress("l1EmPartIsoE",            treeVars.l1EmPartIsoE);	
    chain -> SetBranchAddress("l1EmPartIsoPt",           treeVars.l1EmPartIsoPt);
    chain -> SetBranchAddress("l1EmPartIsoBx",           treeVars.l1EmPartIsoBx);
    chain -> SetBranchAddress("l1EmPartIsoType",         treeVars.l1EmPartIsoType);
    chain -> SetBranchAddress("l1EmPartIsoCandIEta",     treeVars.l1EmPartIsoCandIEta);
    chain -> SetBranchAddress("l1EmPartIsoCandIPhi",     treeVars.l1EmPartIsoCandIPhi);
    chain -> SetBranchAddress("l1EmPartIsoGctCandIndex", treeVars.l1EmPartIsoGctCandIndex);
    
    chain -> SetBranchAddress("l1NEmPartNonIso",           &treeVars.l1NEmPartNonIso);
    chain -> SetBranchAddress("l1EmPartNonIsoX",            treeVars.l1EmPartNonIsoX);
    chain -> SetBranchAddress("l1EmPartNonIsoY",            treeVars.l1EmPartNonIsoY);
    chain -> SetBranchAddress("l1EmPartNonIsoZ",            treeVars.l1EmPartNonIsoZ);
    chain -> SetBranchAddress("l1EmPartNonIsoEta",          treeVars.l1EmPartNonIsoEta);
    chain -> SetBranchAddress("l1EmPartNonIsoPhi",          treeVars.l1EmPartNonIsoPhi);
    chain -> SetBranchAddress("l1EmPartNonIsoPx",           treeVars.l1EmPartNonIsoPx);
    chain -> SetBranchAddress("l1EmPartNonIsoPy",           treeVars.l1EmPartNonIsoPy);
    chain -> SetBranchAddress("l1EmPartNonIsoPz",           treeVars.l1EmPartNonIsoPz);
    chain -> SetBranchAddress("l1EmPartNonIsoE",            treeVars.l1EmPartNonIsoE);
    chain -> SetBranchAddress("l1EmPartNonIsoPt",           treeVars.l1EmPartNonIsoPt);
    chain -> SetBranchAddress("l1EmPartNonIsoBx",           treeVars.l1EmPartNonIsoBx);
    chain -> SetBranchAddress("l1EmPartNonIsoType",         treeVars.l1EmPartNonIsoType);
    chain -> SetBranchAddress("l1EmPartNonIsoCandIEta",     treeVars.l1EmPartNonIsoCandIEta);
    chain -> SetBranchAddress("l1EmPartNonIsoCandIPhi",     treeVars.l1EmPartNonIsoCandIPhi);
    chain -> SetBranchAddress("l1EmPartNonIsoGctCandIndex", treeVars.l1EmPartNonIsoGctCandIndex);
    
    chain -> SetBranchAddress("l1NGctCandIso",           &treeVars.l1NGctCandIso);
    chain -> SetBranchAddress("l1GctCandIsoIEta",         treeVars.l1GctCandIsoIEta);
    chain -> SetBranchAddress("l1GctCandIsoIPhi",         treeVars.l1GctCandIsoIPhi);
    chain -> SetBranchAddress("l1GctCandIsoRank",         treeVars.l1GctCandIsoRank);
    chain -> SetBranchAddress("l1GctCandIsoIsIsolated",   treeVars.l1GctCandIsoIsIsolated);
    chain -> SetBranchAddress("l1GctCandIsoCapBlock",     treeVars.l1GctCandIsoCapBlock);
    chain -> SetBranchAddress("l1GctCandIsoCapIndex",     treeVars.l1GctCandIsoCapIndex);
    chain -> SetBranchAddress("l1GctCandIsoBx",           treeVars.l1GctCandIsoBx);
    chain -> SetBranchAddress("l1GctCandIsoCaloRegIndex", treeVars.l1GctCandIsoCaloRegIndex);
    
    chain -> SetBranchAddress("l1NGctCandNonIso",           &treeVars.l1NGctCandNonIso);
    chain -> SetBranchAddress("l1GctCandNonIsoIEta",         treeVars.l1GctCandNonIsoIEta);
    chain -> SetBranchAddress("l1GctCandNonIsoIPhi",         treeVars.l1GctCandNonIsoIPhi);
    chain -> SetBranchAddress("l1GctCandNonIsoRank",         treeVars.l1GctCandNonIsoRank);
    chain -> SetBranchAddress("l1GctCandNonIsoIsIsolated",   treeVars.l1GctCandNonIsoIsIsolated);
    chain -> SetBranchAddress("l1GctCandNonIsoCapBlock",     treeVars.l1GctCandNonIsoCapBlock);
    chain -> SetBranchAddress("l1GctCandNonIsoCapIndex",     treeVars.l1GctCandNonIsoCapIndex);
    chain -> SetBranchAddress("l1GctCandNonIsoBx",           treeVars.l1GctCandNonIsoBx);
    chain -> SetBranchAddress("l1GctCandNonIsoCaloRegIndex", treeVars.l1GctCandNonIsoCaloRegIndex);
    												
    chain -> SetBranchAddress("l1NCaloCand",	     &treeVars.l1NCaloCand);
    chain -> SetBranchAddress("l1CaloCandIEta",       treeVars.l1CaloCandIEta);
    chain -> SetBranchAddress("l1CaloCandIPhi",       treeVars.l1CaloCandIPhi);
    chain -> SetBranchAddress("l1CaloCandRctCard",    treeVars.l1CaloCandRctCard);
    chain -> SetBranchAddress("l1CaloCandRctReg",     treeVars.l1CaloCandRctReg);
    chain -> SetBranchAddress("l1CaloCandRctCrate",   treeVars.l1CaloCandRctCrate);
    chain -> SetBranchAddress("l1CaloCandRank",       treeVars.l1CaloCandRank);
    chain -> SetBranchAddress("l1CaloCandIsIsolated", treeVars.l1CaloCandIsIsolated);
    chain -> SetBranchAddress("l1CaloCandBx",         treeVars.l1CaloCandBx);
  												
    chain -> SetBranchAddress("l1NCaloReg",         &treeVars.l1NCaloReg);
    chain -> SetBranchAddress("l1CaloRegIEta",       treeVars.l1CaloRegIEta);
    chain -> SetBranchAddress("l1CaloRegIPhi",       treeVars.l1CaloRegIPhi);
    chain -> SetBranchAddress("l1CaloRegCapBlock",   treeVars.l1CaloRegCapBlock);
    chain -> SetBranchAddress("l1CaloRegCapIndex",   treeVars.l1CaloRegCapIndex);
    chain -> SetBranchAddress("l1CaloRegRctCard",    treeVars.l1CaloRegRctCard);
    chain -> SetBranchAddress("l1CaloRegRctReg",     treeVars.l1CaloRegRctReg);
    chain -> SetBranchAddress("l1CaloRegRctCrate",   treeVars.l1CaloRegRctCrate);
    chain -> SetBranchAddress("l1CaloRegRank",       treeVars.l1CaloRegRank);
    chain -> SetBranchAddress("l1CaloRegIsIsolated", treeVars.l1CaloRegIsIsolated);
    chain -> SetBranchAddress("l1CaloRegBx",         treeVars.l1CaloRegBx);
    chain -> SetBranchAddress("l1CaloCandIndex",     treeVars.l1CaloCandIndex);
    
    
    //l1 extra particles - muons
    chain -> SetBranchAddress("l1NMuons",        &treeVars.l1NMuons);
    chain -> SetBranchAddress("l1MuonX",          treeVars.l1MuonX);
    chain -> SetBranchAddress("l1MuonY",          treeVars.l1MuonY);
    chain -> SetBranchAddress("l1MuonZ",          treeVars.l1MuonZ);
    chain -> SetBranchAddress("l1MuonEta",        treeVars.l1MuonEta);
    chain -> SetBranchAddress("l1MuonPhi",        treeVars.l1MuonPhi);
    chain -> SetBranchAddress("l1MuonPx",         treeVars.l1MuonPx);
    chain -> SetBranchAddress("l1MuonPy",         treeVars.l1MuonPy);
    chain -> SetBranchAddress("l1MuonPz",         treeVars.l1MuonPz);
    chain -> SetBranchAddress("l1MuonE",          treeVars.l1MuonE);
    chain -> SetBranchAddress("l1MuonPt",         treeVars.l1MuonPt);
    chain -> SetBranchAddress("l1MuonBx",         treeVars.l1MuonBx);
    chain -> SetBranchAddress("l1MuonIsIsolated", treeVars.l1MuonIsIsolated);
    chain -> SetBranchAddress("l1MuonIsMip",      treeVars.l1MuonIsMip);
    chain -> SetBranchAddress("l1MuonIsForward",  treeVars.l1MuonIsForward);
    chain -> SetBranchAddress("l1MuonIsRPC",      treeVars.l1MuonIsRPC);

    //gmt data
    chain -> SetBranchAddress("l1NGmtCand",                 &treeVars.l1NGmtCand);
    chain -> SetBranchAddress("l1GmtCandIEta",               treeVars.l1GmtCandIEta);
    chain -> SetBranchAddress("l1GmtCandIPhi",               treeVars.l1GmtCandIPhi);
    chain -> SetBranchAddress("l1GmtCandIPt",                treeVars.l1GmtCandIPt);
    chain -> SetBranchAddress("l1GmtCandEta",                treeVars.l1GmtCandEta);
    chain -> SetBranchAddress("l1GmtCandPhi",                treeVars.l1GmtCandPhi);
    chain -> SetBranchAddress("l1GmtCandPt",                 treeVars.l1GmtCandPt);
    chain -> SetBranchAddress("l1GmtCandCharge",             treeVars.l1GmtCandCharge);
    chain -> SetBranchAddress("l1GmtCandUseInSingleMuonTrg", treeVars.l1GmtCandUseInSingleMuonTrg);
    chain -> SetBranchAddress("l1GmtCandUseInDiMuonTrg",     treeVars.l1GmtCandUseInDiMuonTrg);
    chain -> SetBranchAddress("l1GmtCandIsMatchedCand",      treeVars.l1GmtCandIsMatchedCand);
    chain -> SetBranchAddress("l1GmtCandIsHaloCand",         treeVars.l1GmtCandIsHaloCand);
    chain -> SetBranchAddress("l1GmtCandIsol",               treeVars.l1GmtCandIsol);
    chain -> SetBranchAddress("l1GmtCandMip",                treeVars.l1GmtCandMip);
    chain -> SetBranchAddress("l1GmtCandQuality",            treeVars.l1GmtCandQuality);
    chain -> SetBranchAddress("l1GmtCandBx",                 treeVars.l1GmtCandBx);
  } // L1 VARIABLES
  //*/
}


 



void setBranches(TTree* chain, EcalTimeTreeContent& treeVars)
{
  chain -> Branch("runId",         &treeVars.runId,                "runId/i");
  chain -> Branch("lumiSection",   &treeVars.lumiSection,    "lumiSection/i");
  chain -> Branch("unixTime",      &treeVars.unixTime,          "unixTime/i");
  chain -> Branch("orbit",         &treeVars.orbit,                "orbit/i");
  chain -> Branch("bx",            &treeVars.bx,                      "bx/i");
  chain -> Branch("eventId",       &treeVars.eventId,            "eventId/i");
  chain -> Branch("eventNaiveId",  &treeVars.eventNaiveId,  "eventNaiveId/i");
  chain -> Branch("timeStampLow",  &treeVars.timeStampLow,  "timeStampLow/i");
  chain -> Branch("timeStampHigh", &treeVars.timeStampHigh, "timeStampHigh/i");
  
  
 
  ///*  
  // TRG VARIABLES  
  if(EcalTimeTreeContent::trgVariables)
  {    
    // trigger variables
    chain -> Branch("isRPCL1",  &treeVars.isRPCL1,   "isRPCL1/O");
    chain -> Branch("isDTL1",   &treeVars.isDTL1,     "isDTL1/O");
    chain -> Branch("isCSCL1",  &treeVars.isCSCL1,   "isCSCL1/O");
    chain -> Branch("isECALL1", &treeVars.isECALL1, "isECALL1/O");
    chain -> Branch("isHCALL1", &treeVars.isHCALL1, "isHCALL1/O");
  
    chain -> Branch("isRPCL1Bx",  treeVars.isRPCL1Bx,   "isRPCL1Bx[3]/O");
    chain -> Branch("isDTL1Bx",   treeVars.isDTL1Bx,     "isDTL1Bx[3]/O");
    chain -> Branch("isCSCL1Bx",  treeVars.isCSCL1Bx,   "isCSCL1Bx[3]/O");
    chain -> Branch("isECALL1Bx", treeVars.isECALL1Bx, "isECALL1Bx[3]/O");
    chain -> Branch("isHCALL1Bx", treeVars.isHCALL1Bx, "isHCALL1Bx[3]/O");
  }
  //*/  
  


  ///*
  // ECAL VARIABLES  
  if(EcalTimeTreeContent::ecalVariables)
  {    
    // supercluster variables
    chain -> Branch("nSuperClusters",       &treeVars.nSuperClusters,                               "nSuperClusters/I");
    chain -> Branch("nBarrelSuperClusters", &treeVars.nBarrelSuperClusters,                   "nBarrelSuperClusters/I");
    chain -> Branch("nEndcapSuperClusters", &treeVars.nEndcapSuperClusters,                   "nEndcapSuperClusters/I");
    chain -> Branch("superClusterRawEnergy", treeVars.superClusterRawEnergy, "superClusterRawEnergy[nSuperClusters]/F");
    chain -> Branch("superClusterPhiWidth",  treeVars.superClusterPhiWidth,   "superClusterPhiWidth[nSuperClusters]/F");
    chain -> Branch("superClusterEtaWidth",  treeVars.superClusterEtaWidth,   "superClusterEtaWidth[nSuperClusters]/F");
    chain -> Branch("superClusterPhi",       treeVars.superClusterPhi,             "superClusterPhi[nSuperClusters]/F");
    chain -> Branch("superClusterEta",       treeVars.superClusterEta,             "superClusterEta[nSuperClusters]/F");
    chain -> Branch("superClusterX",         treeVars.superClusterX,                 "superClusterX[nSuperClusters]/F");
    chain -> Branch("superClusterY",         treeVars.superClusterY,                 "superClusterY[nSuperClusters]/F");
    chain -> Branch("superClusterZ",         treeVars.superClusterZ,                 "superClusterZ[nSuperClusters]/F");
    chain -> Branch("superClusterVertexX",         treeVars.superClusterVertexX,                 "superClusterVertexX[nSuperClusters]/F");
    chain -> Branch("superClusterVertexY",         treeVars.superClusterVertexY,                 "superClusterVertexY[nSuperClusters]/F");
    chain -> Branch("superClusterVertexZ",         treeVars.superClusterVertexZ,                 "superClusterVertexZ[nSuperClusters]/F");

    chain -> Branch("nClustersInSuperCluster",    treeVars.nClustersInSuperCluster,       "nClustersInSuperCluster[nSuperClusters]/I");
    
    // basic cluster variables
    chain -> Branch("nClusters",              &treeVars.nClusters,                                        "nClusters/I");
    chain -> Branch("clusterEnergy",           treeVars.clusterEnergy,                     "clusterEnergy[nClusters]/F");
    chain -> Branch("clusterTransverseEnergy", treeVars.clusterTransverseEnergy, "clusterTransverseEnergy[nClusters]/F");
    chain -> Branch("clusterE1",               treeVars.clusterE1,                             "clusterE1[nClusters]/F");
    chain -> Branch("clusterE2",               treeVars.clusterE2,                             "clusterE2[nClusters]/F");
    chain -> Branch("clusterTime",             treeVars.clusterTime,                         "clusterTime[nClusters]/F");
    chain -> Branch("clusterPhi",              treeVars.clusterPhi,                           "clusterPhi[nClusters]/F");
    chain -> Branch("clusterEta",              treeVars.clusterEta,                           "clusterEta[nClusters]/F");
    chain -> Branch("clusterXtals",            treeVars.clusterXtals,                       "clusterXtals[nClusters]/I");
    chain -> Branch("clusterXtalsAbove3Sigma", treeVars.clusterXtalsAbove3Sigma, "clusterXtalsAbove3Sigma[nClusters]/I");
    chain -> Branch("clusterMaxId",            treeVars.clusterMaxId,                       "clusterMaxId[nClusters]/i");
    chain -> Branch("cluster2ndId",            treeVars.cluster2ndId,                       "cluster2ndId[nClusters]/i");
    
    chain -> Branch("nXtalsInCluster",    treeVars.nXtalsInCluster,       "nXtalsInCluster[nClusters]/I");
        
    // clustershape variables
    chain -> Branch("clusterE2x2",       treeVars.clusterE2x2,             "clusterE2x2[nClusters]/F");
    chain -> Branch("clusterE3x2",       treeVars.clusterE3x2,             "clusterE3x2[nClusters]/F");
    chain -> Branch("clusterE3x3",       treeVars.clusterE3x3,             "clusterE3x3[nClusters]/F");
    chain -> Branch("clusterE4x4",       treeVars.clusterE4x4,             "clusterE4x4[nClusters]/F");
    chain -> Branch("clusterE5x5",       treeVars.clusterE5x5,             "clusterE5x5[nClusters]/F");
    chain -> Branch("clusterE2x5Right",  treeVars.clusterE2x5Right,   "clusterE2x5Right[nClusters]/F");
    chain -> Branch("clusterE2x5Left",   treeVars.clusterE2x5Left,     "clusterE2x5Left[nClusters]/F");
    chain -> Branch("clusterE2x5Top",    treeVars.clusterE2x5Top,       "clusterE2x5Top[nClusters]/F");
    chain -> Branch("clusterE2x5Bottom", treeVars.clusterE2x5Bottom, "clusterE2x5Bottom[nClusters]/F");
    chain -> Branch("clusterE3x2Ratio",  treeVars.clusterE3x2Ratio,   "clusterE3x2Ratio[nClusters]/F");
    chain -> Branch("clusterCovPhiPhi",  treeVars.clusterCovPhiPhi,   "clusterCovPhiPhi[nClusters]/F");
    chain -> Branch("clusterCovEtaEta",  treeVars.clusterCovEtaEta,   "clusterCovEtaEta[nClusters]/F");
    chain -> Branch("clusterCovEtaPhi",  treeVars.clusterCovEtaPhi,   "clusterCovEtaPhi[nClusters]/F");
    chain -> Branch("clusterLat",        treeVars.clusterLat,               "clusterLat[nClusters]/F");
    chain -> Branch("clusterPhiLat",     treeVars.clusterPhiLat,         "clusterPhiLat[nClusters]/F");
    chain -> Branch("clusterEtaLat",     treeVars.clusterEtaLat,         "clusterEtaLat[nClusters]/F");
    chain -> Branch("clusterZernike20",  treeVars.clusterZernike20,   "clusterZernike20[nClusters]/F");
    chain -> Branch("clusterZernike42",  treeVars.clusterZernike42,   "clusterZernike42[nClusters]/F");
    
    
    // vertex variables
    chain -> Branch("nVertices",         &treeVars.nVertices,   "nVertices/I");
    chain -> Branch("vtxNTracks",       treeVars.vtxNTracks,   "vtxNTracks[nVertices]/I");
    chain -> Branch("vtxChi2",          treeVars.vtxChi2,      "vtxChi2[nVertices]/F");
    chain -> Branch("vtxNdof",          treeVars.vtxNdof,      "vtxNdof[nVertices]/F");
    chain -> Branch("vtxX",             treeVars.vtxX,         "vtxX[nVertices]/F");
    chain -> Branch("vtxDx",            treeVars.vtxDx,        "vtxDx[nVertices]/F");
    chain -> Branch("vtxY",             treeVars.vtxY,         "vtxY[nVertices]/F");
    chain -> Branch("vtxDy",            treeVars.vtxDy,        "vtxDy[nVertices]/F");
    chain -> Branch("vtxZ",             treeVars.vtxZ,         "vtxZ[nVertices]/F");
    chain -> Branch("vtxDz",            treeVars.vtxDz,        "vtxDz[nVertices]/F");


    // xtal variables inside a cluster // strange: MAXXTALINC needs be replaced by explicit "9"; not understood gf
    // using MAXXTALINC, compilation is sucesful BUT the concerned branches don't get filled (Wed Jul 13 20:42:37 CEST 2011)  
    chain -> Branch("xtalInBCHashedIndex",  treeVars.xtalInBCHashedIndex,  "xtalInBCHashedIndex[nClusters][25]/I");
    chain -> Branch("xtalInBCIEta",         treeVars.xtalInBCIEta,         "xtalInBCIEta[nClusters][25]/I");
    chain -> Branch("xtalInBCIPhi",         treeVars.xtalInBCIPhi,         "xtalInBCIPhi[nClusters][25]/I");
    chain -> Branch("xtalInBCEta",          treeVars.xtalInBCEta,          "xtalInBCEta[nClusters][25]/F");
    chain -> Branch("xtalInBCPhi",          treeVars.xtalInBCPhi,          "xtalInBCPhi[nClusters][25]/F");
    chain -> Branch("xtalInBCIx",           treeVars.xtalInBCIx,           "xtalInBCIx[nClusters][25]/I");
    chain -> Branch("xtalInBCIy",           treeVars.xtalInBCIy,           "xtalInBCIy[nClusters][25]/I");
    chain -> Branch("xtalInBCFlag",         treeVars.xtalInBCFlag,         "xtalInBCFlag[nClusters][25]/I");
    chain -> Branch("xtalInBCEnergy",       treeVars.xtalInBCEnergy,       "xtalInBCEnergy[nClusters][25]/F");
    chain -> Branch("xtalInBCTime",         treeVars.xtalInBCTime,         "xtalInBCTime[nClusters][25]/F");
    chain -> Branch("xtalInBCTimeErr",      treeVars.xtalInBCTimeErr,      "xtalInBCTimeErr[nClusters][25]/F");
    chain -> Branch("xtalInBCAmplitudeADC", treeVars.xtalInBCAmplitudeADC, "xtalInBCAmplitudeADC[nClusters][25]/F");
    chain -> Branch("xtalInBCChi2", treeVars.xtalInBCChi2, "xtalInBCChi2[nClusters][25]/F");
    chain -> Branch("xtalInBCOutOfTimeChi2", treeVars.xtalInBCOutOfTimeChi2, "xtalInBCOutOfTimeChi2[nClusters][25]/F");
    chain -> Branch("xtalInBCSwissCross", treeVars.xtalInBCSwissCross, "xtalInBCSwissCross[nClusters][25]/F");

  } // ECAL VARIABLES  
  //*/  
  
  
  
  ///*
  //HCAL VARIABLES
  if( EcalTimeTreeContent::hcalVariables ) 
  {  
    chain -> Branch("hbNRecHits",   &treeVars.hbNRecHits,                   "hbNRecHits/I");
    chain -> Branch("hbRecHitDetId", treeVars.hbRecHitDetId, "hbRecHitDetId[hbNRecHits]/I");
    chain -> Branch("hbRecHitEta",   treeVars.hbRecHitEta,     "hbRecHitEta[hbNRecHits]/F");
    chain -> Branch("hbRecHitPhi",   treeVars.hbRecHitPhi,     "hbRecHitPhi[hbNRecHits]/F");
    chain -> Branch("hbRecHitE",     treeVars.hbRecHitE,         "hbRecHitE[hbNRecHits]/F");
    chain -> Branch("hbRecHitTime",  treeVars.hbRecHitTime,   "hbRecHitTime[hbNRecHits]/F");
    
    chain -> Branch("nCaloTowers",         &treeVars.nCaloTowers,                                "nCaloTowers/I");     
    chain -> Branch("caloTowerEmEnergy",    treeVars.caloTowerEmEnergy,       "caloTowerEmEnergy[nCaloTowers]/F");
    chain -> Branch("caloTowerHadEnergy",   treeVars.caloTowerHadEnergy,     "caloTowerHadEnergy[nCaloTowers]/F");
    chain -> Branch("caloTowerOuterEnergy", treeVars.caloTowerOuterEnergy, "caloTowerOuterEnergy[nCaloTowers]/F");
    chain -> Branch("caloTowerEmEta",       treeVars.caloTowerEmEta,             "caloTowerEmEta[nCaloTowers]/F");
    chain -> Branch("caloTowerEmPhi",       treeVars.caloTowerEmPhi,             "caloTowerEmPhi[nCaloTowers]/F");
    chain -> Branch("caloTowerHadEta",      treeVars.caloTowerHadEta,           "caloTowerHadEta[nCaloTowers]/F");
    chain -> Branch("caloTowerHadPhi",      treeVars.caloTowerHadPhi,           "caloTowerHadPhi[nCaloTowers]/F");
  } // HCAL VARIABLES
  //*/  
  
  
  
  ///*
  // MUON VARIABLES  
  if(EcalTimeTreeContent::muonVariables)
  {    
    // muon variables
    chain -> Branch("nRecoMuons",     &treeVars.nRecoMuons,               "nRecoMuons/I");
    chain -> Branch("muonX",           treeVars.muonX,                     "muonX[nRecoMuons]/F");
    chain -> Branch("muonY",           treeVars.muonY,                     "muonY[nRecoMuons]/F");
    chain -> Branch("muonZ",           treeVars.muonZ,                     "muonZ[nRecoMuons]/F");
    chain -> Branch("muonPhi",         treeVars.muonPhi,                 "muonPhi[nRecoMuons]/F");
    chain -> Branch("muonEta",         treeVars.muonEta,                 "muonEta[nRecoMuons]/F");
    chain -> Branch("muond0",          treeVars.muond0,                   "muond0[nRecoMuons]/F");
    chain -> Branch("muondz",          treeVars.muondz,                   "muondz[nRecoMuons]/F");
    chain -> Branch("muonPx",          treeVars.muonPx,                   "muonPx[nRecoMuons]/F");
    chain -> Branch("muonPy",          treeVars.muonPy,                   "muonPy[nRecoMuons]/F");
    chain -> Branch("muonPz",          treeVars.muonPz,                   "muonPz[nRecoMuons]/F");
    chain -> Branch("muonP",           treeVars.muonP,                     "muonP[nRecoMuons]/F");
    chain -> Branch("muonPt",          treeVars.muonPt,                   "muonPt[nRecoMuons]/F");
    chain -> Branch("muonPtError",     treeVars.muonPtError,         "muonPtError[nRecoMuons]/F");
    chain -> Branch("muonCharge",      treeVars.muonCharge,           "muonCharge[nRecoMuons]/F");
    chain -> Branch("muonQOverP",      treeVars.muonQOverP,           "muonQOverP[nRecoMuons]/F");
    chain -> Branch("muonQOverPError", treeVars.muonQOverPError, "muonQOverPError[nRecoMuons]/F");
    chain -> Branch("muonNChi2",       treeVars.muonNChi2,             "muonNChi2[nRecoMuons]/F");
    chain -> Branch("muonNDof",        treeVars.muonNDof,               "muonNDof[nRecoMuons]/F");
    chain -> Branch("muonNHits",       treeVars.muonNHits,             "muonNHits[nRecoMuons]/F");
    
    chain -> Branch("muonInnerHitX",   treeVars.muonInnerHitX,     "muonInnerHitX[nRecoMuons]/F");
    chain -> Branch("muonInnerHitY",   treeVars.muonInnerHitY,     "muonInnerHitY[nRecoMuons]/F");
    chain -> Branch("muonInnerHitZ",   treeVars.muonInnerHitZ,     "muonInnerHitZ[nRecoMuons]/F");
    chain -> Branch("muonInnerHitPhi", treeVars.muonInnerHitPhi, "muonInnerHitPhi[nRecoMuons]/F");
    chain -> Branch("muonInnerHitEta", treeVars.muonInnerHitEta, "muonInnerHitEta[nRecoMuons]/F");
    chain -> Branch("muonInnerHitPx",  treeVars.muonInnerHitPx,   "muonInnerHitPx[nRecoMuons]/F");
    chain -> Branch("muonInnerHitPy",  treeVars.muonInnerHitPy,   "muonInnerHitPy[nRecoMuons]/F");
    chain -> Branch("muonInnerHitPz",  treeVars.muonInnerHitPz,   "muonInnerHitPz[nRecoMuons]/F");
    chain -> Branch("muonInnerHitP",   treeVars.muonInnerHitP,     "muonInnerHitP[nRecoMuons]/F");
    chain -> Branch("muonInnerHitPt",  treeVars.muonInnerHitPt,   "muonInnerHitPt[nRecoMuons]/F");
    
    chain -> Branch("muonOuterHitX",   treeVars.muonOuterHitX,     "muonOuterHitX[nRecoMuons]/F");
    chain -> Branch("muonOuterHitY",   treeVars.muonOuterHitY,     "muonOuterHitY[nRecoMuons]/F");
    chain -> Branch("muonOuterHitZ",   treeVars.muonOuterHitZ,     "muonOuterHitZ[nRecoMuons]/F");
    chain -> Branch("muonOuterHitPhi", treeVars.muonOuterHitPhi, "muonOuterHitPhi[nRecoMuons]/F");
    chain -> Branch("muonOuterHitEta", treeVars.muonOuterHitEta, "muonOuterHitEta[nRecoMuons]/F");
    chain -> Branch("muonOuterHitPx",  treeVars.muonOuterHitPx,   "muonOuterHitPx[nRecoMuons]/F");
    chain -> Branch("muonOuterHitPy",  treeVars.muonOuterHitPy,   "muonOuterHitPy[nRecoMuons]/F");
    chain -> Branch("muonOuterHitPz",  treeVars.muonOuterHitPz,   "muonOuterHitPz[nRecoMuons]/F");
    chain -> Branch("muonOuterHitP",   treeVars.muonOuterHitP,     "muonOuterHitP[nRecoMuons]/F");
    chain -> Branch("muonOuterHitPt",  treeVars.muonOuterHitPt,   "muonOuterHitPt[nRecoMuons]/F");
    
    chain -> Branch("muonInnTkInnerHitX",   treeVars.muonInnTkInnerHitX,     "muonInnTkInnerHitX[nRecoMuons]/F");
    chain -> Branch("muonInnTkInnerHitY",   treeVars.muonInnTkInnerHitY,     "muonInnTkInnerHitY[nRecoMuons]/F");
    chain -> Branch("muonInnTkInnerHitZ",   treeVars.muonInnTkInnerHitZ,     "muonInnTkInnerHitZ[nRecoMuons]/F");
    chain -> Branch("muonInnTkInnerHitPhi", treeVars.muonInnTkInnerHitPhi, "muonInnTkInnerHitPhi[nRecoMuons]/F");
    chain -> Branch("muonInnTkInnerHitEta", treeVars.muonInnTkInnerHitEta, "muonInnTkInnerHitEta[nRecoMuons]/F");
    chain -> Branch("muonInnTkInnerHitPx",  treeVars.muonInnTkInnerHitPx,   "muonInnTkInnerHitPx[nRecoMuons]/F");
    chain -> Branch("muonInnTkInnerHitPy",  treeVars.muonInnTkInnerHitPy,   "muonInnTkInnerHitPy[nRecoMuons]/F");
    chain -> Branch("muonInnTkInnerHitPz",  treeVars.muonInnTkInnerHitPz,   "muonInnTkInnerHitPz[nRecoMuons]/F");
    chain -> Branch("muonInnTkInnerHitP",   treeVars.muonInnTkInnerHitP,     "muonInnTkInnerHitP[nRecoMuons]/F");
    chain -> Branch("muonInnTkInnerHitPt",  treeVars.muonInnTkInnerHitPt,   "muonInnTkInnerHitPt[nRecoMuons]/F");
    
    chain -> Branch("muonInnTkOuterHitX",   treeVars.muonInnTkOuterHitX,     "muonInnTkOuterHitX[nRecoMuons]/F");
    chain -> Branch("muonInnTkOuterHitY",   treeVars.muonInnTkOuterHitY,     "muonInnTkOuterHitY[nRecoMuons]/F");
    chain -> Branch("muonInnTkOuterHitZ",   treeVars.muonInnTkOuterHitZ,     "muonInnTkOuterHitZ[nRecoMuons]/F");
    chain -> Branch("muonInnTkOuterHitPhi", treeVars.muonInnTkOuterHitPhi, "muonInnTkOuterHitPhi[nRecoMuons]/F");
    chain -> Branch("muonInnTkOuterHitEta", treeVars.muonInnTkOuterHitEta, "muonInnTkOuterHitEta[nRecoMuons]/F");
    chain -> Branch("muonInnTkOuterHitPx",  treeVars.muonInnTkOuterHitPx,   "muonInnTkOuterHitPx[nRecoMuons]/F");
    chain -> Branch("muonInnTkOuterHitPy",  treeVars.muonInnTkOuterHitPy,   "muonInnTkOuterHitPy[nRecoMuons]/F");
    chain -> Branch("muonInnTkOuterHitPz",  treeVars.muonInnTkOuterHitPz,   "muonInnTkOuterHitPz[nRecoMuons]/F");
    chain -> Branch("muonInnTkOuterHitP",   treeVars.muonInnTkOuterHitP,     "muonInnTkOuterHitP[nRecoMuons]/F");
    chain -> Branch("muonInnTkOuterHitPt",  treeVars.muonInnTkOuterHitPt,   "muonInnTkOuterHitPt[nRecoMuons]/F");
    
    chain -> Branch("muonOutTkInnerHitX",   treeVars.muonOutTkInnerHitX,     "muonOutTkInnerHitX[nRecoMuons]/F");
    chain -> Branch("muonOutTkInnerHitY",   treeVars.muonOutTkInnerHitY,     "muonOutTkInnerHitY[nRecoMuons]/F");
    chain -> Branch("muonOutTkInnerHitZ",   treeVars.muonOutTkInnerHitZ,     "muonOutTkInnerHitZ[nRecoMuons]/F");
    chain -> Branch("muonOutTkInnerHitPhi", treeVars.muonOutTkInnerHitPhi, "muonOutTkInnerHitPhi[nRecoMuons]/F");
    chain -> Branch("muonOutTkInnerHitEta", treeVars.muonOutTkInnerHitEta, "muonOutTkInnerHitEta[nRecoMuons]/F");
    chain -> Branch("muonOutTkInnerHitPx",  treeVars.muonOutTkInnerHitPx,   "muonOutTkInnerHitPx[nRecoMuons]/F");
    chain -> Branch("muonOutTkInnerHitPy",  treeVars.muonOutTkInnerHitPy,   "muonOutTkInnerHitPy[nRecoMuons]/F");
    chain -> Branch("muonOutTkInnerHitPz",  treeVars.muonOutTkInnerHitPz,   "muonOutTkInnerHitPz[nRecoMuons]/F");
    chain -> Branch("muonOutTkInnerHitP",   treeVars.muonOutTkInnerHitP,     "muonOutTkInnerHitP[nRecoMuons]/F");
    chain -> Branch("muonOutTkInnerHitPt",  treeVars.muonOutTkInnerHitPt,   "muonOutTkInnerHitPt[nRecoMuons]/F");
    
    chain -> Branch("muonOutTkOuterHitX",   treeVars.muonOutTkOuterHitX,     "muonOutTkOuterHitX[nRecoMuons]/F");
    chain -> Branch("muonOutTkOuterHitY",   treeVars.muonOutTkOuterHitY,     "muonOutTkOuterHitY[nRecoMuons]/F");
    chain -> Branch("muonOutTkOuterHitZ",   treeVars.muonOutTkOuterHitZ,     "muonOutTkOuterHitZ[nRecoMuons]/F");
    chain -> Branch("muonOutTkOuterHitPhi", treeVars.muonOutTkOuterHitPhi, "muonOutTkOuterHitPhi[nRecoMuons]/F");
    chain -> Branch("muonOutTkOuterHitEta", treeVars.muonOutTkOuterHitEta, "muonOutTkOuterHitEta[nRecoMuons]/F");
    chain -> Branch("muonOutTkOuterHitPx",  treeVars.muonOutTkOuterHitPx,   "muonOutTkOuterHitPx[nRecoMuons]/F");
    chain -> Branch("muonOutTkOuterHitPy",  treeVars.muonOutTkOuterHitPy,   "muonOutTkOuterHitPy[nRecoMuons]/F");
    chain -> Branch("muonOutTkOuterHitPz",  treeVars.muonOutTkOuterHitPz,   "muonOutTkOuterHitPz[nRecoMuons]/F");
    chain -> Branch("muonOutTkOuterHitP",   treeVars.muonOutTkOuterHitP,     "muonOutTkOuterHitP[nRecoMuons]/F");
    chain -> Branch("muonOutTkOuterHitPt",  treeVars.muonOutTkOuterHitPt,   "muonOutTkOuterHitPt[nRecoMuons]/F");
    
    chain -> Branch("muonLeg", treeVars.muonLeg, "muonLeg[nRecoMuons]/I"); 
    
    chain -> Branch("muonTkLengthInEcalApprox",            treeVars.muonTkLengthInEcalApprox,                       "muonTkLengthInEcalApprox[nRecoMuons]/F");
    chain -> Branch("muonTkLengthInEcalDetail",            treeVars.muonTkLengthInEcalDetail,                       "muonTkLengthInEcalDetail[nRecoMuons]/F");
    chain -> Branch("muonTkLengthInEcalDetailCurved",      treeVars.muonTkLengthInEcalDetailCurved,           "muonTkLengthInEcalDetailCurved[nRecoMuons]/F");
    chain -> Branch("muonTkLengthInEcalDetailCurved_high", treeVars.muonTkLengthInEcalDetailCurved_high, "muonTkLengthInEcalDetailCurved_high[nRecoMuons]/F");
    chain -> Branch("muonTkLengthInEcalDetailCurved_low",  treeVars.muonTkLengthInEcalDetailCurved_low,   "muonTkLengthInEcalDetailCurved_low[nRecoMuons]/F");
    
    chain -> Branch("muonTkInternalPointInEcalX", treeVars.muonTkInternalPointInEcalX, "muonTkInternalPointInEcalX[nRecoMuons]/F");
    chain -> Branch("muonTkInternalPointInEcalY", treeVars.muonTkInternalPointInEcalY, "muonTkInternalPointInEcalY[nRecoMuons]/F");
    chain -> Branch("muonTkInternalPointInEcalZ", treeVars.muonTkInternalPointInEcalZ, "muonTkInternalPointInEcalZ[nRecoMuons]/F");
    chain -> Branch("muonTkExternalPointInEcalX", treeVars.muonTkExternalPointInEcalX, "muonTkExternalPointInEcalX[nRecoMuons]/F");
    chain -> Branch("muonTkExternalPointInEcalY", treeVars.muonTkExternalPointInEcalY, "muonTkExternalPointInEcalY[nRecoMuons]/F");
    chain -> Branch("muonTkExternalPointInEcalZ", treeVars.muonTkExternalPointInEcalZ, "muonTkExternalPointInEcalZ[nRecoMuons]/F");
    chain -> Branch("muonTkInternalPointInEcalCurvedX", treeVars.muonTkInternalPointInEcalCurvedX, "muonTkInternalPointInEcalCurvedX[nRecoMuons]/F");
    chain -> Branch("muonTkInternalPointInEcalCurvedY", treeVars.muonTkInternalPointInEcalCurvedY, "muonTkInternalPointInEcalCurvedY[nRecoMuons]/F");
    chain -> Branch("muonTkInternalPointInEcalCurvedZ", treeVars.muonTkInternalPointInEcalCurvedZ, "muonTkInternalPointInEcalCurvedZ[nRecoMuons]/F");
    chain -> Branch("muonTkExternalPointInEcalCurvedX", treeVars.muonTkExternalPointInEcalCurvedX, "muonTkExternalPointInEcalCurvedX[nRecoMuons]/F");
    chain -> Branch("muonTkExternalPointInEcalCurvedY", treeVars.muonTkExternalPointInEcalCurvedY, "muonTkExternalPointInEcalCurvedY[nRecoMuons]/F");
    chain -> Branch("muonTkExternalPointInEcalCurvedZ", treeVars.muonTkExternalPointInEcalCurvedZ, "muonTkExternalPointInEcalCurvedZ[nRecoMuons]/F");
    chain -> Branch("muonTkInternalPointInEcalCurvedPx", treeVars.muonTkInternalPointInEcalCurvedPx, "muonTkInternalPointInEcalCurvedPx[nRecoMuons]/F");
    chain -> Branch("muonTkInternalPointInEcalCurvedPy", treeVars.muonTkInternalPointInEcalCurvedPy, "muonTkInternalPointInEcalCurvedPy[nRecoMuons]/F");
    chain -> Branch("muonTkInternalPointInEcalCurvedPz", treeVars.muonTkInternalPointInEcalCurvedPz, "muonTkInternalPointInEcalCurvedPz[nRecoMuons]/F");
    chain -> Branch("muonTkExternalPointInEcalCurvedPx", treeVars.muonTkExternalPointInEcalCurvedPx, "muonTkExternalPointInEcalCurvedPx[nRecoMuons]/F");
    chain -> Branch("muonTkExternalPointInEcalCurvedPy", treeVars.muonTkExternalPointInEcalCurvedPy, "muonTkExternalPointInEcalCurvedPy[nRecoMuons]/F");
    chain -> Branch("muonTkExternalPointInEcalCurvedPz", treeVars.muonTkExternalPointInEcalCurvedPz, "muonTkExternalPointInEcalCurvedPz[nRecoMuons]/F");    

    chain -> Branch("nMuonCrossedXtals", treeVars.nMuonCrossedXtals, "nMuonCrossedXtals[nRecoMuons]/I");
    chain -> Branch("nMuonCrossedXtalsCurved", treeVars.nMuonCrossedXtalsCurved, "nMuonCrossedXtalsCurved[nRecoMuons]/I");
    chain -> Branch("muonCrossedXtalHashedIndex", treeVars.muonCrossedXtalHashedIndex,          "muonCrossedXtalHashedIndex[nRecoMuons][250]/I");
    chain -> Branch("muonCrossedXtalHashedIndexCurved", treeVars.muonCrossedXtalHashedIndexCurved, "muonCrossedXtalHashedIndexCurved[nRecoMuons][250]/I");
    chain -> Branch("muonCrossedXtalTkLength", treeVars.muonCrossedXtalTkLength,                   "muonCrossedXtalTkLength[nRecoMuons][250]/F");
    chain -> Branch("muonCrossedXtalTkLengthCurved", treeVars.muonCrossedXtalTkLengthCurved, "muonCrossedXtalTkLengthCurved[nRecoMuons][250]/F");
  } // MUON VARIABLES
  //*/
  


  ///*
  // TPG VARIABLES
  if(EcalTimeTreeContent::tpgVariables)
  {
    chain->Branch("tpgNTowers",         &treeVars.tpgNTowers,                                   "tpgNTowers/I");
    chain->Branch("tpgIEta",             treeVars.tpgIEta,                             "tpgIEta[tpgNTowers]/I");
    chain->Branch("tpgIPhi",             treeVars.tpgIPhi,                             "tpgIPhi[tpgNTowers]/I");
    chain->Branch("tpgNbOfXtals",        treeVars.tpgNbOfXtals,                   "tpgNbOfXtals[tpgNTowers]/I");
    chain->Branch("tpgEnRec",            treeVars.tpgEnRec,                           "tpgEnRec[tpgNTowers]/F");
    chain->Branch("tpgADC",              treeVars.tpgADC,                               "tpgADC[tpgNTowers]/I");
    chain->Branch("tpgNActiveTriggers", &treeVars.tpgNActiveTriggers,                   "tpgNActiveTriggers/I");
    chain->Branch("tpgActiveTriggers",   treeVars.tpgActiveTriggers, "tpgActiveTriggers[tpgNActiveTriggers]/I");

    chain->Branch("tpEmulNTowers", &treeVars.tpEmulNTowers,          "tpEmulNTowers/I");
    chain->Branch("tpEmulIEta",     treeVars.tpEmulIEta, "tpEmulIEta[tpEmulNTowers]/I");
    chain->Branch("tpEmulIPhi",     treeVars.tpEmulIPhi, "tpEmulIPhi[tpEmulNTowers]/I");
    chain->Branch("tpEmulADC1",     treeVars.tpEmulADC1, "tpEmulADC1[tpEmulNTowers]/I");
    chain->Branch("tpEmulADC2",     treeVars.tpEmulADC2, "tpEmulADC2[tpEmulNTowers]/I");
    chain->Branch("tpEmulADC3",     treeVars.tpEmulADC3, "tpEmulADC3[tpEmulNTowers]/I");
    chain->Branch("tpEmulADC4",     treeVars.tpEmulADC4, "tpEmulADC4[tpEmulNTowers]/I");
    chain->Branch("tpEmulADC5",     treeVars.tpEmulADC5, "tpEmulADC5[tpEmulNTowers]/I");
  }    
  //*/



  ///*
  // L1 VARIABLES
  if(EcalTimeTreeContent::l1Variables)    
  {
      
    chain->Branch("l1NActiveTriggers", &treeVars.l1NActiveTriggers,                  "l1NActiveTriggers/I");
    chain->Branch("l1ActiveTriggers",   treeVars.l1ActiveTriggers, "l1ActiveTriggers[l1NActiveTriggers]/I");
    chain->Branch("l1NActiveTechTriggers", &treeVars.l1NActiveTechTriggers,                  "l1NActiveTechTriggers/I");
    chain->Branch("l1ActiveTechTriggers",   treeVars.l1ActiveTechTriggers, "l1ActiveTechTriggers[l1NActiveTechTriggers]/I");
    
    chain->Branch("l1GtNEm",    &treeVars.l1GtNEm,   "l1GtNEm/I"   );
    chain->Branch("l1GtEmBx",   treeVars.l1GtEmBx,   "l1GtEmBx[l1GtNEm]/I"  );
    chain->Branch("l1GtEmIEta", treeVars.l1GtEmIEta, "l1GtEmIEta[l1GtNEm]/I" );
    chain->Branch("l1GtEmIPhi", treeVars.l1GtEmIPhi, "l1GtEmIPhi[l1GtNEm]/I" );
    chain->Branch("l1GtEmEta", treeVars.l1GtEmEta, "l1GtEmEta[l1GtNEm]/F" );
    chain->Branch("l1GtEmPhi", treeVars.l1GtEmPhi, "l1GtEmPhi[l1GtNEm]/F" );
    chain->Branch("l1GtEmRank", treeVars.l1GtEmRank, "l1GtEmRank[l1GtNEm]/I" );
    chain->Branch("l1GtEmEt",   treeVars.l1GtEmEt,   "l1GtEmEt[l1GtNEm]/F" );

    chain->Branch("l1NEmPartIso",           &treeVars.l1NEmPartIso,                                     "l1NEmPartIso/I");
    chain->Branch("l1EmPartIsoX",            treeVars.l1EmPartIsoX,                       "l1EmPartIsoX[l1NEmPartIso]/F");
    chain->Branch("l1EmPartIsoY",            treeVars.l1EmPartIsoY,                       "l1EmPartIsoY[l1NEmPartIso]/F");
    chain->Branch("l1EmPartIsoZ",            treeVars.l1EmPartIsoZ,                       "l1EmPartIsoZ[l1NEmPartIso]/F");
    chain->Branch("l1EmPartIsoEta",          treeVars.l1EmPartIsoEta,                   "l1EmPartIsoEta[l1NEmPartIso]/F");
    chain->Branch("l1EmPartIsoPhi",          treeVars.l1EmPartIsoPhi,                   "l1EmPartIsoPhi[l1NEmPartIso]/F");
    chain->Branch("l1EmPartIsoPx",           treeVars.l1EmPartIsoPx,                     "l1EmPartIsoPx[l1NEmPartIso]/F");
    chain->Branch("l1EmPartIsoPy",           treeVars.l1EmPartIsoPy,                     "l1EmPartIsoPy[l1NEmPartIso]/F");
    chain->Branch("l1EmPartIsoPz",           treeVars.l1EmPartIsoPz,                     "l1EmPartIsoPz[l1NEmPartIso]/F");
    chain->Branch("l1EmPartIsoE",            treeVars.l1EmPartIsoE,                       "l1EmPartIsoE[l1NEmPartIso]/F");
    chain->Branch("l1EmPartIsoPt",           treeVars.l1EmPartIsoPt,                     "l1EmPartIsoPt[l1NEmPartIso]/F");	
    chain->Branch("l1EmPartIsoBx",           treeVars.l1EmPartIsoBx,           	         "l1EmPartIsoBx[l1NEmPartIso]/i");
    chain->Branch("l1EmPartIsoType",         treeVars.l1EmPartIsoType,                 "l1EmPartIsoType[l1NEmPartIso]/I");
    chain->Branch("l1EmPartIsoCandIEta",     treeVars.l1EmPartIsoCandIEta,         "l1EmPartIsoCandIEta[l1NEmPartIso]/I");
    chain->Branch("l1EmPartIsoCandIPhi",     treeVars.l1EmPartIsoCandIPhi,         "l1EmPartIsoCandIPhi[l1NEmPartIso]/I");
    chain->Branch("l1EmPartIsoGctCandIndex", treeVars.l1EmPartIsoGctCandIndex, "l1EmPartIsoGctCandIndex[l1NEmPartIso]/I");
    
    chain->Branch("l1NEmPartNonIso",           &treeVars.l1NEmPartNonIso,                                        "l1NEmPartNonIso/I");
    chain->Branch("l1EmPartNonIsoX",            treeVars.l1EmPartNonIsoX,                       "l1EmPartNonIsoX[l1NEmPartNonIso]/F");
    chain->Branch("l1EmPartNonIsoY",            treeVars.l1EmPartNonIsoY,                       "l1EmPartNonIsoY[l1NEmPartNonIso]/F");
    chain->Branch("l1EmPartNonIsoZ",            treeVars.l1EmPartNonIsoZ,                       "l1EmPartNonIsoZ[l1NEmPartNonIso]/F");
    chain->Branch("l1EmPartNonIsoEta",          treeVars.l1EmPartNonIsoEta,                   "l1EmPartNonIsoEta[l1NEmPartNonIso]/F");
    chain->Branch("l1EmPartNonIsoPhi",          treeVars.l1EmPartNonIsoPhi,                   "l1EmPartNonIsoPhi[l1NEmPartNonIso]/F");
    chain->Branch("l1EmPartNonIsoPx",           treeVars.l1EmPartNonIsoPx,                     "l1EmPartNonIsoPx[l1NEmPartNonIso]/F");
    chain->Branch("l1EmPartNonIsoPy",           treeVars.l1EmPartNonIsoPy,                     "l1EmPartNonIsoPy[l1NEmPartNonIso]/F");
    chain->Branch("l1EmPartNonIsoPz",           treeVars.l1EmPartNonIsoPz,                     "l1EmPartNonIsoPz[l1NEmPartNonIso]/F");
    chain->Branch("l1EmPartNonIsoE",            treeVars.l1EmPartNonIsoE,                       "l1EmPartNonIsoE[l1NEmPartNonIso]/F");
    chain->Branch("l1EmPartNonIsoPt",           treeVars.l1EmPartNonIsoPt,                     "l1EmPartNonIsoPt[l1NEmPartNonIso]/F");
    chain->Branch("l1EmPartNonIsoBx",           treeVars.l1EmPartNonIsoBx,                     "l1EmPartNonIsoBx[l1NEmPartNonIso]/i");
    chain->Branch("l1EmPartNonIsoType",         treeVars.l1EmPartNonIsoType,                 "l1EmPartNonIsoType[l1NEmPartNonIso]/I");
    chain->Branch("l1EmPartNonIsoCandIEta",     treeVars.l1EmPartNonIsoCandIEta,         "l1EmPartNonIsoCandIEta[l1NEmPartNonIso]/I");
    chain->Branch("l1EmPartNonIsoCandIPhi",     treeVars.l1EmPartNonIsoCandIPhi,         "l1EmPartNonIsoCandIPhi[l1NEmPartNonIso]/I");
    chain->Branch("l1EmPartNonIsoGctCandIndex",	treeVars.l1EmPartNonIsoGctCandIndex, "l1EmPartNonIsoGctCandIndex[l1NEmPartNonIso]/I");
  
    chain->Branch("l1NGctCandIso",           &treeVars.l1NGctCandIso,                                        "l1NGctCandIso/I");
    chain->Branch("l1GctCandIsoIEta",         treeVars.l1GctCandIsoIEta,                 "l1GctCandIsoIEta[l1NGctCandIso]/I");
    chain->Branch("l1GctCandIsoIPhi",         treeVars.l1GctCandIsoIPhi,                 "l1GctCandIsoIPhi[l1NGctCandIso]/I");
    chain->Branch("l1GctCandIsoRank",         treeVars.l1GctCandIsoRank,                 "l1GctCandIsoRank[l1NGctCandIso]/i");
    chain->Branch("l1GctCandIsoIsIsolated",   treeVars.l1GctCandIsoIsIsolated,     "l1GctCandIsoIsIsolated[l1NGctCandIso]/O");
    chain->Branch("l1GctCandIsoCapBlock",     treeVars.l1GctCandIsoCapBlock,         "l1GctCandIsoCapBlock[l1NGctCandIso]/i");
    chain->Branch("l1GctCandIsoCapIndex",     treeVars.l1GctCandIsoCapIndex,         "l1GctCandIsoCapIndex[l1NGctCandIso]/i");
    chain->Branch("l1GctCandIsoBx",           treeVars.l1GctCandIsoBx,                     "l1GctCandIsoBx[l1NGctCandIso]/i");	  
    chain->Branch("l1GctCandIsoCaloRegIndex", treeVars.l1GctCandIsoCaloRegIndex, "l1GctCandIsoCaloRegIndex[l1NGctCandIso]/I");
    
    chain->Branch("l1NGctCandNonIso",           &treeVars.l1NGctCandNonIso,                                           "l1NGctCandNonIso/I");
    chain->Branch("l1GctCandNonIsoIEta",         treeVars.l1GctCandNonIsoIEta,                 "l1GctCandNonIsoIEta[l1NGctCandNonIso]/I");
    chain->Branch("l1GctCandNonIsoIPhi",         treeVars.l1GctCandNonIsoIPhi,                 "l1GctCandNonIsoIPhi[l1NGctCandNonIso]/I");
    chain->Branch("l1GctCandNonIsoRank",         treeVars.l1GctCandNonIsoRank,                 "l1GctCandNonIsoRank[l1NGctCandNonIso]/i");
    chain->Branch("l1GctCandNonIsoIsIsolated",   treeVars.l1GctCandNonIsoIsIsolated,     "l1GctCandNonIsoIsIsolated[l1NGctCandNonIso]/O");
    chain->Branch("l1GctCandNonIsoCapBlock",     treeVars.l1GctCandNonIsoCapBlock,         "l1GctCandNonIsoCapBlock[l1NGctCandNonIso]/i");
    chain->Branch("l1GctCandNonIsoCapIndex",     treeVars.l1GctCandNonIsoCapIndex,         "l1GctCandNonIsoCapIndex[l1NGctCandNonIso]/i");
    chain->Branch("l1GctCandNonIsoBx",           treeVars.l1GctCandNonIsoBx,                     "l1GctCandNonIsoBx[l1NGctCandNonIso]/i");
    chain->Branch("l1GctCandNonIsoCaloRegIndex", treeVars.l1GctCandNonIsoCaloRegIndex, "l1GctCandNonIsoCaloRegIndex[l1NGctCandNonIso]/I");
    
    chain->Branch("l1NCaloCand",         &treeVars.l1NCaloCand,                                  "l1NCaloCand/I");
    chain->Branch("l1CaloCandIEta",       treeVars.l1CaloCandIEta,             "l1CaloCandIEta[l1NCaloCand]/I");
    chain->Branch("l1CaloCandIPhi",       treeVars.l1CaloCandIPhi,             "l1CaloCandIPhi[l1NCaloCand]/I");
    chain->Branch("l1CaloCandRctCard",    treeVars.l1CaloCandRctCard,       "l1CaloCandRctCard[l1NCaloCand]/I");
    chain->Branch("l1CaloCandRctReg",     treeVars.l1CaloCandRctReg,         "l1CaloCandRctReg[l1NCaloCand]/I");
    chain->Branch("l1CaloCandRctCrate",   treeVars.l1CaloCandRctCrate,     "l1CaloCandRctCrate[l1NCaloCand]/I");
    chain->Branch("l1CaloCandRank",       treeVars.l1CaloCandRank,             "l1CaloCandRank[l1NCaloCand]/i");
    chain->Branch("l1CaloCandIsIsolated", treeVars.l1CaloCandIsIsolated, "l1CaloCandIsIsolated[l1NCaloCand]/O");
    chain->Branch("l1CaloCandBx",         treeVars.l1CaloCandBx,                 "l1CaloCandBx[l1NCaloCand]/i");
    
    chain->Branch("l1NCaloReg",         &treeVars.l1NCaloReg,                                 "l1NCaloReg/I");
    chain->Branch("l1CaloRegIEta",       treeVars.l1CaloRegIEta,             "l1CaloRegIEta[l1NCaloReg]/I");
    chain->Branch("l1CaloRegIPhi",       treeVars.l1CaloRegIPhi,             "l1CaloRegIPhi[l1NCaloReg]/I");
    chain->Branch("l1CaloRegCapBlock",   treeVars.l1CaloRegCapBlock,     "l1CaloRegCapBlock[l1NCaloReg]/I");
    chain->Branch("l1CaloRegCapIndex",   treeVars.l1CaloRegCapIndex,     "l1CaloRegCapIndex[l1NCaloReg]/I");
    chain->Branch("l1CaloRegRctCard",    treeVars.l1CaloRegRctCard,       "l1CaloRegRctCard[l1NCaloReg]/I");
    chain->Branch("l1CaloRegRctReg",     treeVars.l1CaloRegRctReg,         "l1CaloRegRctReg[l1NCaloReg]/I");
    chain->Branch("l1CaloRegRctCrate",   treeVars.l1CaloRegRctCrate,     "l1CaloRegRctCrate[l1NCaloReg]/I");
    chain->Branch("l1CaloRegRank",       treeVars.l1CaloRegRank,             "l1CaloRegRank[l1NCaloReg]/i");
    chain->Branch("l1CaloRegIsIsolated", treeVars.l1CaloRegIsIsolated, "l1CaloRegIsIsolated[l1NCaloReg]/O");
    chain->Branch("l1CaloRegBx",         treeVars.l1CaloRegBx,                 "l1CaloRegBx[l1NCaloReg]/i");
    chain->Branch("l1CaloCandIndex",     treeVars.l1CaloCandIndex,         "l1CaloCandIndex[l1NCaloReg]/I");
    
    
    //l1 extra particles - muons
    chain->Branch("l1NMuons",        &treeVars.l1NMuons,                           "l1NMuons/I");
    chain->Branch("l1MuonX",          treeVars.l1MuonX,                   "l1MuonX[l1NMuons]/F");
    chain->Branch("l1MuonY",          treeVars.l1MuonY,                   "l1MuonY[l1NMuons]/F");
    chain->Branch("l1MuonZ",          treeVars.l1MuonZ,                   "l1MuonZ[l1NMuons]/F");
    chain->Branch("l1MuonEta",        treeVars.l1MuonEta,               "l1MuonEta[l1NMuons]/F");
    chain->Branch("l1MuonPhi",        treeVars.l1MuonPhi,               "l1MuonPhi[l1NMuons]/F");
    chain->Branch("l1MuonPx",         treeVars.l1MuonPx,                 "l1MuonPx[l1NMuons]/F");
    chain->Branch("l1MuonPy",         treeVars.l1MuonPy,                 "l1MuonPy[l1NMuons]/F");
    chain->Branch("l1MuonPz",         treeVars.l1MuonPz,                 "l1MuonPz[l1NMuons]/F");
    chain->Branch("l1MuonE",          treeVars.l1MuonE,                   "l1MuonE[l1NMuons]/F");
    chain->Branch("l1MuonPt",         treeVars.l1MuonPt,                 "l1MuonPt[l1NMuons]/F");
    chain->Branch("l1MuonBx",         treeVars.l1MuonBx,                 "l1MuonBx[l1NMuons]/i");
    chain->Branch("l1MuonIsIsolated", treeVars.l1MuonIsIsolated, "l1MuonIsIsolated[l1NMuons]/O");
    chain->Branch("l1MuonIsMip",      treeVars.l1MuonIsMip,           "l1MuonIsMip[l1NMuons]/O");
    chain->Branch("l1MuonIsForward",  treeVars.l1MuonIsForward,   "l1MuonIsForward[l1NMuons]/O");
    chain->Branch("l1MuonIsRPC",      treeVars.l1MuonIsRPC,           "l1MuonIsRPC[l1NMuons]/O");
    
    
    //gmt info
    chain->Branch("l1NGmtCand",                 &treeVars.l1NGmtCand,                                               "l1NGmtCand/I");
    chain->Branch("l1GmtCandIEta",               treeVars.l1GmtCandIEta,                             "l1GmtCandIEta[l1NGmtCand]/I");
    chain->Branch("l1GmtCandIPhi",               treeVars.l1GmtCandIPhi,                             "l1GmtCandIPhi[l1NGmtCand]/I");
    chain->Branch("l1GmtCandIPt",                treeVars.l1GmtCandIPt,                               "l1GmtCandIPt[l1NGmtCand]/I");
    chain->Branch("l1GmtCandEta",                treeVars.l1GmtCandEta,                               "l1GmtCandEta[l1NGmtCand]/F");
    chain->Branch("l1GmtCandPhi",                treeVars.l1GmtCandPhi,                               "l1GmtCandPhi[l1NGmtCand]/F");
    chain->Branch("l1GmtCandPt",                 treeVars.l1GmtCandPt,                                 "l1GmtCandPt[l1NGmtCand]/F");
    chain->Branch("l1GmtCandCharge",             treeVars.l1GmtCandCharge,                         "l1GmtCandCharge[l1NGmtCand]/I");
    chain->Branch("l1GmtCandUseInSingleMuonTrg", treeVars.l1GmtCandUseInSingleMuonTrg, "l1GmtCandUseInSingleMuonTrg[l1NGmtCand]/O");
    chain->Branch("l1GmtCandUseInDiMuonTrg",     treeVars.l1GmtCandUseInDiMuonTrg,         "l1GmtCandUseInDiMuonTrg[l1NGmtCand]/O");
    chain->Branch("l1GmtCandIsMatchedCand",      treeVars.l1GmtCandIsMatchedCand,           "l1GmtCandIsMatchedCand[l1NGmtCand]/O");
    chain->Branch("l1GmtCandIsHaloCand",         treeVars.l1GmtCandIsHaloCand,                 "l1GmtCandIsHaloCand[l1NGmtCand]/O");
    chain->Branch("l1GmtCandIsol",               treeVars.l1GmtCandIsol,                             "l1GmtCandIsol[l1NGmtCand]/O");
    chain->Branch("l1GmtCandMip",                treeVars.l1GmtCandMip,                               "l1GmtCandMip[l1NGmtCand]/O");
    chain->Branch("l1GmtCandQuality",            treeVars.l1GmtCandQuality,                       "l1GmtCandQuality[l1NGmtCand]/I");
    chain->Branch("l1GmtCandBx",                 treeVars.l1GmtCandBx,                                 "l1GmtCandBx[l1NGmtCand]/I");
    
    
  } // TPG VARIABLES
  //*/
}






void initializeBranches(TTree* chain, EcalTimeTreeContent& treeVars)
{
  treeVars.runId = 0;
  treeVars.lumiSection = 0;
  treeVars.unixTime = 0;
  treeVars.orbit = 0;
  treeVars.bx = 0;
  treeVars.eventId = 0; 
  treeVars.eventNaiveId = 0; 
  treeVars.timeStampLow = 0;
  treeVars.timeStampHigh = 0;
  
  

  ///*  
  // TRG VARIABLES  
  if(EcalTimeTreeContent::trgVariables)
  {    
    treeVars.isECALL1 = false; 
    treeVars.isRPCL1 = false; 
    treeVars.isDTL1 = false;
    treeVars.isCSCL1 = false;
    treeVars.isHCALL1 = false;

    for(int i = 0; i < 3 ; ++i)
    { 
      treeVars.isRPCL1Bx[i] = false; 
      treeVars.isDTL1Bx[i] = false; 
      treeVars.isCSCL1Bx[i] = false; 
      treeVars.isECALL1Bx[i] = false; 
      treeVars.isHCALL1Bx[i] = false; 
    }
  } // TRG VARIABLES  
  //*/
  
  
  
  ///*
  // ECAL VARIABLES  
  if(EcalTimeTreeContent::ecalVariables)
  {    
    //supercluster variables
    treeVars.nSuperClusters = 0;
    treeVars.nBarrelSuperClusters = 0;
    treeVars.nEndcapSuperClusters = 0;
    for(int i = 0; i < MAXSC; ++i)
    {
    treeVars.superClusterRawEnergy[i] = 0.;
    treeVars.superClusterPhiWidth[i] = 0.;
    treeVars.superClusterEtaWidth[i] = 0.;
    treeVars.superClusterEta[i] = 0.;
    treeVars.superClusterPhi[i] = 0.;
    treeVars.superClusterX[i] = 0.;
    treeVars.superClusterY[i] = 0.;
    treeVars.superClusterZ[i] = 0.;
    treeVars.superClusterVertexX[i] = 0.;
    treeVars.superClusterVertexY[i] = 0.;
    treeVars.superClusterVertexZ[i] = 0.;
    
    treeVars.nClustersInSuperCluster[i] = 0;
    }
    
    
    //basic cluster variables	
    treeVars.nClusters = 0;
    for(int i = 0; i < MAXC; ++i)
    {
    treeVars.clusterEnergy[i] = 0.;
    treeVars.clusterTransverseEnergy[i] = 0.;
    treeVars.clusterE1[i] = 0.;
    treeVars.clusterE2[i] = 0.;
    treeVars.clusterTime[i] = 0.;
    treeVars.clusterPhi[i] = 0.;
    treeVars.clusterEta[i] = 0.;
    treeVars.clusterXtals[i] = 0;
    treeVars.clusterXtalsAbove3Sigma[i] = 0;
    treeVars.clusterMaxId[i] = 0;
    treeVars.cluster2ndId[i] = 0;

    treeVars.nXtalsInCluster[i] = 0;
    }
    
    
    //clustershape variables    
    for(int i = 0; i < MAXC; ++i)
    {
    treeVars.clusterE2x2[i] = 0.;
    treeVars.clusterE3x2[i] = 0.;
    treeVars.clusterE3x3[i] = 0.;
    treeVars.clusterE4x4[i] = 0.;
    treeVars.clusterE5x5[i] = 0.;
    treeVars.clusterE2x5Right[i] = 0.;
    treeVars.clusterE2x5Left[i] = 0.;
    treeVars.clusterE2x5Top[i] = 0.;
    treeVars.clusterE2x5Bottom[i] = 0.;
    treeVars.clusterE3x2Ratio[i] = 0.;
    treeVars.clusterCovPhiPhi[i] = 0.;
    treeVars.clusterCovEtaEta[i] = 0.;
    treeVars.clusterCovEtaPhi[i] = 0.;
    treeVars.clusterLat[i] = 0.;
    treeVars.clusterPhiLat[i] = 0.;
    treeVars.clusterEtaLat[i] = 0.;
    treeVars.clusterZernike20[i] = 0.;
    treeVars.clusterZernike42[i] = 0.;
    }
    
  // it's convenient keeping vertex variables here within the ECAL group
  // since  each SC is assigned a vertex (which is the electron vertex, in case electrons be used)
  // vertex variables
    treeVars.nVertices=0;
    for(int i=0; i<MAXVTX; i++) {
      treeVars.vtxIsFake[i]=false;
      treeVars.vtxNTracks[i]=0;
      treeVars.vtxChi2[i]=0;
      treeVars.vtxNdof[i]=0;
      treeVars.vtxX[i]=0;
      treeVars.vtxDx[i]=0;
      treeVars.vtxY[i]=0;
      treeVars.vtxDy[i]=0;
      treeVars.vtxZ[i]=0;
      treeVars.vtxDz[i]=0;
  }
  
    // xtal variables inside a cluster
    for(int cl=0; cl<MAXC; cl++ ){
      for(int cryInClu=0; cryInClu<MAXXTALINC; cryInClu++){
	treeVars.xtalInBCHashedIndex[cl][cryInClu]=0;
	treeVars.xtalInBCIEta[cl][cryInClu]=0;
	treeVars.xtalInBCIPhi[cl][cryInClu]=0;
	treeVars.xtalInBCEta[cl][cryInClu]=0;
	treeVars.xtalInBCPhi[cl][cryInClu]=0;
	treeVars.xtalInBCIx[cl][cryInClu]=0;
	treeVars.xtalInBCIy[cl][cryInClu]=0;
	treeVars.xtalInBCFlag[cl][cryInClu]=0;
	treeVars.xtalInBCEnergy[cl][cryInClu]=0;
	treeVars.xtalInBCTime[cl][cryInClu]=0;
	treeVars.xtalInBCTimeErr[cl][cryInClu]=0;
	treeVars.xtalInBCAmplitudeADC[cl][cryInClu]=0;
	treeVars.xtalInBCChi2[cl][cryInClu]=0;
	treeVars.xtalInBCOutOfTimeChi2[cl][cryInClu]=0;
        treeVars.xtalInBCSwissCross[cl][cryInClu]=0;
      }}

  } // ECAL VARIABLES
  //*/  



  // HCAL VARIABLES
  if( EcalTimeTreeContent::hcalVariables ) 
  {  
    treeVars.hbNRecHits = 0;
    for(int i = 0; i < MAXHCALRECHITS; ++i)
    {
      treeVars.hbRecHitDetId[i] = 0;
      treeVars.hbRecHitEta[i] = 0.;
      treeVars.hbRecHitPhi[i] = 0.;
      treeVars.hbRecHitE[i] = 0.;
      treeVars.hbRecHitTime[i] = 0.;
    }

 
    treeVars.nCaloTowers = 0;
    for(int i = 0; i < MAXCALOTOWERS; ++i)
    {
      treeVars.caloTowerEmEnergy[i] = 0.;
      treeVars.caloTowerHadEnergy[i] = 0.;
      treeVars.caloTowerOuterEnergy[i] = 0.;
      treeVars.caloTowerEmEta[i] = 0.;
      treeVars.caloTowerEmPhi[i] = 0.;
      treeVars.caloTowerHadEta[i] = 0.;
      treeVars.caloTowerHadPhi[i] = 0.;
    }
  } //HCAL variables


  
  ///*
  // MUON VARIABLES  
  if(EcalTimeTreeContent::muonVariables)
  {    
    // muon variables
    treeVars.nRecoMuons = 0;
    for(int i = 0; i < 20; ++i)
    {
      treeVars.muonX[i] = 0.;
      treeVars.muonY[i] = 0.;
      treeVars.muonZ[i] = 0.;
      treeVars.muonPhi[i] = 0.;
      treeVars.muonEta[i] = 0.;
      treeVars.muond0[i] = 0.;
      treeVars.muondz[i] = 0.;
      treeVars.muonPx[i] = 0.;
      treeVars.muonPy[i] = 0.;
      treeVars.muonPz[i] = 0.;
      treeVars.muonP[i] = 0.;
      treeVars.muonPt[i] = 0.;
      treeVars.muonPtError[i] = 0.;
      treeVars.muonCharge[i] = 0.;
      treeVars.muonQOverP[i] = 0.;
      treeVars.muonQOverPError[i] = 0.;
      treeVars.muonNChi2[i] = 0.;
      treeVars.muonNDof[i] = 0.;
      treeVars.muonNHits[i] = 0.;
      
      treeVars.muonInnerHitX[i] = 0.;
      treeVars.muonInnerHitY[i] = 0.;
      treeVars.muonInnerHitZ[i] = 0.;
      treeVars.muonInnerHitEta[i] = 0.;
      treeVars.muonInnerHitPhi[i] = 0.;
      treeVars.muonInnerHitPx[i] = 0.;
      treeVars.muonInnerHitPy[i] = 0.;
      treeVars.muonInnerHitPz[i] = 0.;
      treeVars.muonInnerHitP[i] = 0.;
      treeVars.muonInnerHitPt[i] = 0.;
      
      treeVars.muonOuterHitX[i] = 0.;
      treeVars.muonOuterHitY[i] = 0.;
      treeVars.muonOuterHitZ[i] = 0.;
      treeVars.muonOuterHitEta[i] = 0.;
      treeVars.muonOuterHitPhi[i] = 0.;
      treeVars.muonOuterHitPx[i] = 0.;
      treeVars.muonOuterHitPy[i] = 0.;
      treeVars.muonOuterHitPz[i] = 0.;
      treeVars.muonOuterHitP[i] = 0.;
      treeVars.muonOuterHitPt[i] = 0.;
      
      treeVars.muonInnTkInnerHitX[i] = 0.;
      treeVars.muonInnTkInnerHitY[i] = 0.;
      treeVars.muonInnTkInnerHitZ[i] = 0.;
      treeVars.muonInnTkInnerHitEta[i] = 0.;
      treeVars.muonInnTkInnerHitPhi[i] = 0.;
      treeVars.muonInnTkInnerHitPx[i] = 0.;
      treeVars.muonInnTkInnerHitPy[i] = 0.;
      treeVars.muonInnTkInnerHitPz[i] = 0.;
      treeVars.muonInnTkInnerHitP[i] = 0.;
      treeVars.muonInnTkInnerHitPt[i] = 0.;
      
      treeVars.muonInnTkOuterHitX[i] = 0.;
      treeVars.muonInnTkOuterHitY[i] = 0.;
      treeVars.muonInnTkOuterHitZ[i] = 0.;
      treeVars.muonInnTkOuterHitEta[i] = 0.;
      treeVars.muonInnTkOuterHitPhi[i] = 0.;
      treeVars.muonInnTkOuterHitPx[i] = 0.;
      treeVars.muonInnTkOuterHitPy[i] = 0.;
      treeVars.muonInnTkOuterHitPz[i] = 0.;
      treeVars.muonInnTkOuterHitP[i] = 0.;
      treeVars.muonInnTkOuterHitPt[i] = 0.;
      
      treeVars.muonOutTkInnerHitX[i] = 0.;
      treeVars.muonOutTkInnerHitY[i] = 0.;
      treeVars.muonOutTkInnerHitZ[i] = 0.;
      treeVars.muonOutTkInnerHitEta[i] = 0.;
      treeVars.muonOutTkInnerHitPhi[i] = 0.;
      treeVars.muonOutTkInnerHitPx[i] = 0.;
      treeVars.muonOutTkInnerHitPy[i] = 0.;
      treeVars.muonOutTkInnerHitPz[i] = 0.;
      treeVars.muonOutTkInnerHitP[i] = 0.;
      treeVars.muonOutTkInnerHitPt[i] = 0.;
      
      treeVars.muonOutTkOuterHitX[i] = 0.;
      treeVars.muonOutTkOuterHitY[i] = 0.;
      treeVars.muonOutTkOuterHitZ[i] = 0.;
      treeVars.muonOutTkOuterHitEta[i] = 0.;
      treeVars.muonOutTkOuterHitPhi[i] = 0.;
      treeVars.muonOutTkOuterHitPx[i] = 0.;
      treeVars.muonOutTkOuterHitPy[i] = 0.;
      treeVars.muonOutTkOuterHitPz[i] = 0.;
      treeVars.muonOutTkOuterHitP[i] = 0.;
      treeVars.muonOutTkOuterHitPt[i] = 0.;
      
      treeVars.muonLeg[i] = 0;
      
      treeVars.muonTkLengthInEcalApprox[i] = 0.;
      treeVars.muonTkLengthInEcalDetail[i] = 0.;
      treeVars.muonTkLengthInEcalDetailCurved[i] = 0.;
      treeVars.muonTkLengthInEcalDetailCurved_high[i] = 0.;
      treeVars.muonTkLengthInEcalDetailCurved_low[i] = 0.;
      
      
      treeVars.muonTkInternalPointInEcalX[i] = 0.;
      treeVars.muonTkInternalPointInEcalY[i] = 0.;
      treeVars.muonTkInternalPointInEcalZ[i] = 0.;
      treeVars.muonTkExternalPointInEcalX[i] = 0.;
      treeVars.muonTkExternalPointInEcalY[i] = 0.;
      treeVars.muonTkExternalPointInEcalZ[i] = 0.;
      treeVars.muonTkInternalPointInEcalCurvedX[i] = 0.;
      treeVars.muonTkInternalPointInEcalCurvedY[i] = 0.;
      treeVars.muonTkInternalPointInEcalCurvedZ[i] = 0.;
      treeVars.muonTkExternalPointInEcalCurvedX[i] = 0.;
      treeVars.muonTkExternalPointInEcalCurvedY[i] = 0.;
      treeVars.muonTkExternalPointInEcalCurvedZ[i] = 0.;
      treeVars.muonTkInternalPointInEcalCurvedPx[i] = 0.;
      treeVars.muonTkInternalPointInEcalCurvedPy[i] = 0.;
      treeVars.muonTkInternalPointInEcalCurvedPz[i] = 0.;
      treeVars.muonTkExternalPointInEcalCurvedPx[i] = 0.;
      treeVars.muonTkExternalPointInEcalCurvedPy[i] = 0.;
      treeVars.muonTkExternalPointInEcalCurvedPz[i] = 0.;    
      
      treeVars.nMuonCrossedXtals[i] = 0;
      treeVars.nMuonCrossedXtalsCurved[i] = 0;
      for(int j = 0; j < 250; ++j)
      {
        treeVars.muonCrossedXtalHashedIndex[i][j] = 0;
        treeVars.muonCrossedXtalHashedIndexCurved[i][j] = 0;
        treeVars.muonCrossedXtalTkLength[i][j] = 0.;
        treeVars.muonCrossedXtalTkLengthCurved[i][j] = 0.;
      }
    }
   
  } // MUON VARIABLES 
  //*/

  // TPG VARIABLES  
  if(EcalTimeTreeContent::tpgVariables)
  {       
    treeVars.tpgNTowers = 0;
    treeVars.tpgNActiveTriggers = 0;
    treeVars.tpEmulNTowers = 0;
  } // TPG VARIABLES



  // L1 VARIABLES
  if(EcalTimeTreeContent::l1Variables)
  {
    treeVars.l1NActiveTriggers = 0;
    treeVars.l1GtNEm = 0;
    treeVars.l1NEmPartIso = 0;
    treeVars.l1NEmPartNonIso = 0;
    treeVars.l1NGctCandIso = 0;
    treeVars.l1NGctCandNonIso = 0;
    treeVars.l1NCaloCand = 0;
    treeVars.l1NCaloReg = 0;
    treeVars.l1NMuons = 0;
    treeVars.l1NGmtCand = 0;
  } // L1 VARIABLES


}
