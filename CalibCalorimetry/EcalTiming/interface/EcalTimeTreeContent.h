#ifndef EcalTimeTreeContent_h
#define EcalTimeTreeContent_h

#include "TChain.h" 

#define MAXSC 50
#define MAXC 200
#define MAXXTALINC 25 // CAUTION: 
                      // if you change this, you need to change by hand the hard-coded '25' which is in "chain -> Branch("xtalInBCHashedIndex",..." in EcalTimeTreeContent.cc
#define MAXVTX 40
#define MAXHCALRECHITS 100
#define MAXCALOTOWERS 100
#define MAXMU 20
#define MAXTOWERSINTPGSUMMARY 100
#define MAXL1OBJS 100


struct EcalTimeTreeContent
{
  // Flags
  static bool trgVariables;
  static bool ecalVariables;
  static bool ecalShapeVariables;
  static bool hcalVariables;
  static bool muonVariables;
  static bool tkAssVariables;
  static bool tpgVariables;
  static bool l1Variables;  
  
  
  unsigned int runId;
  unsigned int lumiSection;
  unsigned int unixTime; /// Time in seconds since January 1, 1970.
  unsigned int orbit;
  unsigned int bx;
  unsigned int eventId;
  unsigned int eventNaiveId;
  unsigned int timeStampLow;
  unsigned int timeStampHigh;
  
  
  
  //trigger variables
  bool isRPCL1;
  bool isDTL1;
  bool isCSCL1;
  bool isECALL1;
  bool isHCALL1;
  
                 
  bool isECALL1Bx[3]; // 
  bool isRPCL1Bx [3]; // 
  bool isDTL1Bx  [3]; //
  bool isCSCL1Bx [3]; //
  bool isHCALL1Bx[3]; //   
  
  
  //ecalVariables variables
  int nSuperClusters;
  int nBarrelSuperClusters;
  int nEndcapSuperClusters;
  float superClusterRawEnergy[MAXSC];
  float superClusterPhiWidth[MAXSC];
  float superClusterEtaWidth[MAXSC];
  float superClusterPhi[MAXSC];
  float superClusterEta[MAXSC];
  float superClusterX[MAXSC];
  float superClusterY[MAXSC];
  float superClusterZ[MAXSC];
  float superClusterVertexX[MAXSC];
  float superClusterVertexY[MAXSC];
  float superClusterVertexZ[MAXSC];

  int nClustersInSuperCluster[MAXSC];  
  int xtalIndexInSuperCluster[MAXSC];
    
  // basic cluster variables	
  int nClusters;
  float clusterEnergy[MAXC];
  float clusterTransverseEnergy[MAXC];
  float clusterE1[MAXC];
  float clusterE2[MAXC];
  float clusterTime[MAXC];
  float clusterPhi[MAXC];
  float clusterEta[MAXC];
  int clusterXtals[MAXC];
  int clusterXtalsAbove3Sigma[MAXC];
  unsigned int clusterMaxId[MAXC];
  unsigned int cluster2ndId[MAXC];
  
  int nXtalsInCluster[MAXC];    
  
  
  // clustershape variables for basic clusters
  float clusterE2x2[MAXC];
  float clusterE3x2[MAXC];
  float clusterE3x3[MAXC];
  float clusterE4x4[MAXC];
  float clusterE5x5[MAXC];
  float clusterE2x5Right[MAXC];
  float clusterE2x5Left[MAXC];
  float clusterE2x5Top[MAXC];
  float clusterE2x5Bottom[MAXC];
  float clusterE3x2Ratio[MAXC];
  float clusterCovPhiPhi[MAXC];
  float clusterCovEtaEta[MAXC];
  float clusterCovEtaPhi[MAXC];
  float clusterLat[MAXC];
  float clusterPhiLat[MAXC];
  float clusterEtaLat[MAXC];
  float clusterZernike20[MAXC];
  float clusterZernike42[MAXC];

  // xtal variables inside a basic cluster
  int   xtalInBCHashedIndex[MAXC][MAXXTALINC];
  int   xtalInBCIEta[MAXC][MAXXTALINC];
  int   xtalInBCIPhi[MAXC][MAXXTALINC];
  float xtalInBCEta[MAXC][MAXXTALINC];
  float xtalInBCPhi[MAXC][MAXXTALINC];
  int   xtalInBCIx[MAXC][MAXXTALINC];
  int   xtalInBCIy[MAXC][MAXXTALINC];
  int   xtalInBCFlag[MAXC][MAXXTALINC];
  float xtalInBCEnergy[MAXC][MAXXTALINC];
  float xtalInBCTime[MAXC][MAXXTALINC];
  float xtalInBCTimeErr[MAXC][MAXXTALINC];
  float xtalInBCAmplitudeADC[MAXC][MAXXTALINC];
  float xtalInBCChi2[MAXC][MAXXTALINC];
  float xtalInBCOutOfTimeChi2[MAXC][MAXXTALINC];
  float xtalInBCSwissCross[MAXC][MAXXTALINC];
    
  // vertex variables
  int   nVertices;
  bool  vtxIsFake[MAXVTX];
  int   vtxNTracks[MAXVTX];
  float vtxChi2[MAXVTX];
  float vtxNdof[MAXVTX];
  float vtxX[MAXVTX];
  float vtxDx[MAXVTX];
  float vtxY[MAXVTX];
  float vtxDy[MAXVTX];
  float vtxZ[MAXVTX];
  float vtxDz[MAXVTX];

  
  // hcal variables
  int   hbNRecHits;
  int   hbRecHitDetId[MAXHCALRECHITS];
  float hbRecHitEta[MAXHCALRECHITS];
  float hbRecHitPhi[MAXHCALRECHITS];
  float hbRecHitE[MAXHCALRECHITS];
  float hbRecHitTime[MAXHCALRECHITS];
  
  int nCaloTowers;
  float caloTowerEmEnergy[MAXCALOTOWERS];
  float caloTowerHadEnergy[MAXCALOTOWERS];
  float caloTowerOuterEnergy[MAXCALOTOWERS];
  float caloTowerEmEta[MAXCALOTOWERS];
  float caloTowerEmPhi[MAXCALOTOWERS];
  float caloTowerHadEta[MAXCALOTOWERS];
  float caloTowerHadPhi[MAXCALOTOWERS];
  
  
  
  // muon variables
  int nRecoMuons;
  float muonX[MAXMU];
  float muonY[MAXMU];
  float muonZ[MAXMU];
  float muonPhi[MAXMU];
  float muonEta[MAXMU];
  float muond0[MAXMU];
  float muondz[MAXMU];
  float muonPx[MAXMU];
  float muonPy[MAXMU];
  float muonPz[MAXMU];
  float muonP[MAXMU];
  float muonPt[MAXMU];
  float muonPtError[MAXMU];
  float muonCharge[MAXMU];
  float muonQOverP[MAXMU];
  float muonQOverPError[MAXMU];
  float muonNChi2[MAXMU];
  float muonNDof[MAXMU];
  float muonNHits[MAXMU];
  
  float muonInnerHitX[MAXMU];
  float muonInnerHitY[MAXMU];
  float muonInnerHitZ[MAXMU];
  float muonInnerHitPhi[MAXMU];
  float muonInnerHitEta[MAXMU];
  float muonInnerHitPx[MAXMU];
  float muonInnerHitPy[MAXMU];
  float muonInnerHitPz[MAXMU];
  float muonInnerHitP[MAXMU];
  float muonInnerHitPt[MAXMU];
  
  float muonOuterHitX[MAXMU];
  float muonOuterHitY[MAXMU];
  float muonOuterHitZ[MAXMU];
  float muonOuterHitPhi[MAXMU];
  float muonOuterHitEta[MAXMU];
  float muonOuterHitPx[MAXMU];
  float muonOuterHitPy[MAXMU];
  float muonOuterHitPz[MAXMU];
  float muonOuterHitP[MAXMU];
  float muonOuterHitPt[MAXMU];
  
  float muonInnTkInnerHitX[MAXMU];
  float muonInnTkInnerHitY[MAXMU];
  float muonInnTkInnerHitZ[MAXMU];
  float muonInnTkInnerHitPhi[MAXMU];
  float muonInnTkInnerHitEta[MAXMU];
  float muonInnTkInnerHitPx[MAXMU];
  float muonInnTkInnerHitPy[MAXMU];
  float muonInnTkInnerHitPz[MAXMU];
  float muonInnTkInnerHitP[MAXMU];
  float muonInnTkInnerHitPt[MAXMU];
  
  float muonInnTkOuterHitX[MAXMU];
  float muonInnTkOuterHitY[MAXMU];
  float muonInnTkOuterHitZ[MAXMU];
  float muonInnTkOuterHitPhi[MAXMU];
  float muonInnTkOuterHitEta[MAXMU];
  float muonInnTkOuterHitPx[MAXMU];
  float muonInnTkOuterHitPy[MAXMU];
  float muonInnTkOuterHitPz[MAXMU];
  float muonInnTkOuterHitP[MAXMU];
  float muonInnTkOuterHitPt[MAXMU];
  
  float muonOutTkInnerHitX[MAXMU];
  float muonOutTkInnerHitY[MAXMU];
  float muonOutTkInnerHitZ[MAXMU];
  float muonOutTkInnerHitPhi[MAXMU];
  float muonOutTkInnerHitEta[MAXMU];
  float muonOutTkInnerHitPx[MAXMU];
  float muonOutTkInnerHitPy[MAXMU];
  float muonOutTkInnerHitPz[MAXMU];
  float muonOutTkInnerHitP[MAXMU];
  float muonOutTkInnerHitPt[MAXMU];
  
  float muonOutTkOuterHitX[MAXMU];
  float muonOutTkOuterHitY[MAXMU];
  float muonOutTkOuterHitZ[MAXMU];
  float muonOutTkOuterHitPhi[MAXMU];
  float muonOutTkOuterHitEta[MAXMU];
  float muonOutTkOuterHitPx[MAXMU];
  float muonOutTkOuterHitPy[MAXMU];
  float muonOutTkOuterHitPz[MAXMU];
  float muonOutTkOuterHitP[MAXMU];
  float muonOutTkOuterHitPt[MAXMU];
  
  int muonLeg[MAXMU];
  
  float muonTkLengthInEcalApprox[MAXMU];
  float muonTkLengthInEcalDetail[MAXMU];
  float muonTkLengthInEcalDetailCurved[MAXMU];
  float muonTkLengthInEcalDetailCurved_high[MAXMU];
  float muonTkLengthInEcalDetailCurved_low[MAXMU];
  
  float muonTkInternalPointInEcalX[MAXMU];
  float muonTkInternalPointInEcalY[MAXMU];
  float muonTkInternalPointInEcalZ[MAXMU];
  float muonTkExternalPointInEcalX[MAXMU];
  float muonTkExternalPointInEcalY[MAXMU];
  float muonTkExternalPointInEcalZ[MAXMU];
  float muonTkInternalPointInEcalCurvedX[MAXMU];
  float muonTkInternalPointInEcalCurvedY[MAXMU];
  float muonTkInternalPointInEcalCurvedZ[MAXMU];
  float muonTkExternalPointInEcalCurvedX[MAXMU];
  float muonTkExternalPointInEcalCurvedY[MAXMU];
  float muonTkExternalPointInEcalCurvedZ[MAXMU];
  float muonTkInternalPointInEcalCurvedPx[MAXMU];
  float muonTkInternalPointInEcalCurvedPy[MAXMU];
  float muonTkInternalPointInEcalCurvedPz[MAXMU];
  float muonTkExternalPointInEcalCurvedPx[MAXMU];
  float muonTkExternalPointInEcalCurvedPy[MAXMU];
  float muonTkExternalPointInEcalCurvedPz[MAXMU];  

  int   nMuonCrossedXtals[MAXMU];
  int   nMuonCrossedXtalsCurved[MAXMU];
  int   muonCrossedXtalHashedIndex[MAXMU][250];
  int   muonCrossedXtalHashedIndexCurved[MAXMU][250];
  float muonCrossedXtalTkLength[MAXMU][250];
  float muonCrossedXtalTkLengthCurved[MAXMU][250];
  
  
  
  //trackAssociator variables
  float muonTkAtEcalPhi[MAXMU];
  float muonTkAtEcalEta[MAXMU];
  float muonTkAtEcalX[MAXMU];
  float muonTkAtEcalY[MAXMU];
  float muonTkAtEcalZ[MAXMU];
  float muonTkAtHcalPhi[MAXMU];
  float muonTkAtHcalEta[MAXMU];
  float muonTkAtHcalX[MAXMU];
  float muonTkAtHcalY[MAXMU];
  float muonTkAtHcalZ[MAXMU];
  
  float muonEcalEnergy3x3[MAXMU];
  float muonEcalEnergy5x5[MAXMU];
  float muonEcalEnergyCrossed[MAXMU];
  float muonHcalEnergy3x3[MAXMU];
  float muonHcalEnergyCrossed[MAXMU];
  int muonNCrossedEcalDetId[MAXMU];
  unsigned int muonMaxEneEcalDetIdCrossed[MAXMU];
  
  float muonTkLengthInEcalApprox_TkAss[MAXMU];
  float muonTkLengthInEcalDetail_TkAss[MAXMU];
  
  
  
  // TPG variables
  int   tpgNTowers;
  int   tpgIEta[MAXTOWERSINTPGSUMMARY];
  int   tpgIPhi[MAXTOWERSINTPGSUMMARY];
  int   tpgNbOfXtals[MAXTOWERSINTPGSUMMARY];
  float tpgEnRec[MAXTOWERSINTPGSUMMARY];
  int   tpgADC[MAXTOWERSINTPGSUMMARY];
  int   tpgNActiveTriggers;
  int   tpgActiveTriggers[128];

  int tpEmulNTowers;
  int tpEmulIEta[MAXTOWERSINTPGSUMMARY];
  int tpEmulIPhi[MAXTOWERSINTPGSUMMARY];
  int tpEmulADC1[MAXTOWERSINTPGSUMMARY];
  int tpEmulADC2[MAXTOWERSINTPGSUMMARY];
  int tpEmulADC3[MAXTOWERSINTPGSUMMARY];
  int tpEmulADC4[MAXTOWERSINTPGSUMMARY];
  int tpEmulADC5[MAXTOWERSINTPGSUMMARY];

  // l1Variables
  int l1NActiveTriggers;
  int l1ActiveTriggers[128];
  int l1NActiveTechTriggers;
  int l1ActiveTechTriggers[128];
  
  //GT +-1BX
  int l1GtNEm;
  int l1GtEmBx[MAXL1OBJS];
  int l1GtEmIEta[MAXL1OBJS];          
  int l1GtEmIPhi[MAXL1OBJS];
  float l1GtEmEta[MAXL1OBJS];
  float l1GtEmPhi[MAXL1OBJS];
  int l1GtEmRank[MAXL1OBJS]; 
  float l1GtEmEt[MAXL1OBJS]; 
  
  // L1 EM Objects
  int          l1NEmPartIso;
  float        l1EmPartIsoX[MAXL1OBJS];
  float        l1EmPartIsoY[MAXL1OBJS];
  float        l1EmPartIsoZ[MAXL1OBJS];
  float        l1EmPartIsoEta[MAXL1OBJS];
  float        l1EmPartIsoPhi[MAXL1OBJS];
  float        l1EmPartIsoPx[MAXL1OBJS];
  float        l1EmPartIsoPy[MAXL1OBJS];
  float        l1EmPartIsoPz[MAXL1OBJS];
  float        l1EmPartIsoE[MAXL1OBJS];
  float        l1EmPartIsoPt[MAXL1OBJS];
  unsigned int l1EmPartIsoBx[MAXL1OBJS];
  int          l1EmPartIsoType[MAXL1OBJS];
  int          l1EmPartIsoCandIEta[MAXL1OBJS];
  int          l1EmPartIsoCandIPhi[MAXL1OBJS];
  int          l1EmPartIsoGctCandIndex[MAXL1OBJS];
  
  int          l1NEmPartNonIso;
  float        l1EmPartNonIsoX[MAXL1OBJS];
  float        l1EmPartNonIsoY[MAXL1OBJS];
  float        l1EmPartNonIsoZ[MAXL1OBJS];
  float        l1EmPartNonIsoEta[MAXL1OBJS];
  float        l1EmPartNonIsoPhi[MAXL1OBJS];
  float        l1EmPartNonIsoPx[MAXL1OBJS];
  float        l1EmPartNonIsoPy[MAXL1OBJS];
  float        l1EmPartNonIsoPz[MAXL1OBJS];
  float        l1EmPartNonIsoE[MAXL1OBJS];
  float        l1EmPartNonIsoPt[MAXL1OBJS];
  unsigned int l1EmPartNonIsoBx[MAXL1OBJS];
  int          l1EmPartNonIsoType[MAXL1OBJS];
  int          l1EmPartNonIsoCandIEta[MAXL1OBJS];
  int          l1EmPartNonIsoCandIPhi[MAXL1OBJS];
  int          l1EmPartNonIsoGctCandIndex[MAXL1OBJS];
  
  int          l1NGctCandIso;
  int          l1GctCandIsoIEta[MAXL1OBJS];
  int          l1GctCandIsoIPhi[MAXL1OBJS];
  unsigned int l1GctCandIsoRank[MAXL1OBJS];
  bool         l1GctCandIsoIsIsolated[MAXL1OBJS];
  unsigned int l1GctCandIsoCapBlock[MAXL1OBJS];
  unsigned int l1GctCandIsoCapIndex[MAXL1OBJS];
  unsigned int l1GctCandIsoBx[MAXL1OBJS];
  int          l1GctCandIsoCaloRegIndex[MAXL1OBJS];
  
  int          l1NGctCandNonIso;
  int          l1GctCandNonIsoIEta[MAXL1OBJS];
  int          l1GctCandNonIsoIPhi[MAXL1OBJS];
  unsigned int l1GctCandNonIsoRank[MAXL1OBJS];
  bool         l1GctCandNonIsoIsIsolated[MAXL1OBJS];
  unsigned int l1GctCandNonIsoCapBlock[MAXL1OBJS];
  unsigned int l1GctCandNonIsoCapIndex[MAXL1OBJS];
  unsigned int l1GctCandNonIsoBx[MAXL1OBJS];
  int          l1GctCandNonIsoCaloRegIndex[MAXL1OBJS];
  
  int          l1NCaloCand;
  int          l1CaloCandIEta[MAXL1OBJS];
  int          l1CaloCandIPhi[MAXL1OBJS];
  int          l1CaloCandRctCard[MAXL1OBJS];
  int          l1CaloCandRctReg[MAXL1OBJS];
  int          l1CaloCandRctCrate[MAXL1OBJS];
  unsigned int l1CaloCandRank[MAXL1OBJS];
  bool         l1CaloCandIsIsolated[MAXL1OBJS];
  unsigned int l1CaloCandBx[MAXL1OBJS];
  
  int          l1NCaloReg;
  int          l1CaloRegIEta[MAXL1OBJS];
  int          l1CaloRegIPhi[MAXL1OBJS];
  int          l1CaloRegCapBlock[MAXL1OBJS];
  int          l1CaloRegCapIndex[MAXL1OBJS];
  int          l1CaloRegRctCard[MAXL1OBJS];
  int          l1CaloRegRctReg[MAXL1OBJS];
  int          l1CaloRegRctCrate[MAXL1OBJS];
  unsigned int l1CaloRegRank[MAXL1OBJS];
  bool         l1CaloRegIsIsolated[MAXL1OBJS];
  unsigned int l1CaloRegBx[MAXL1OBJS];
  int          l1CaloCandIndex[MAXL1OBJS];

  
  
  //L1 MUON OBJECTS
  //l1 muon extra particle
  int   l1NMuons;//
  float l1MuonX[MAXL1OBJS];
  float l1MuonY[MAXL1OBJS];
  float l1MuonZ[MAXL1OBJS];
  float l1MuonEta[MAXL1OBJS];
  float l1MuonPhi[MAXL1OBJS];
  float l1MuonPx[MAXL1OBJS];
  float l1MuonPy[MAXL1OBJS];
  float l1MuonPz[MAXL1OBJS];
  float l1MuonE[MAXL1OBJS];
  float l1MuonPt[MAXL1OBJS];
  unsigned int l1MuonBx[MAXL1OBJS];
  bool   l1MuonIsIsolated[MAXL1OBJS];
  bool   l1MuonIsMip[MAXL1OBJS];
  bool   l1MuonIsForward[MAXL1OBJS];
  bool   l1MuonIsRPC[MAXL1OBJS];
  
  //GMT data
  int l1NGmtCand;
  int l1GmtCandIEta[MAXL1OBJS],  l1GmtCandIPhi[MAXL1OBJS], l1GmtCandIPt[MAXL1OBJS];
  float l1GmtCandEta[MAXL1OBJS], l1GmtCandPhi[MAXL1OBJS],  l1GmtCandPt[MAXL1OBJS];
  int l1GmtCandCharge[MAXL1OBJS];
  bool l1GmtCandUseInSingleMuonTrg[MAXL1OBJS];
  bool l1GmtCandUseInDiMuonTrg[MAXL1OBJS];
  bool l1GmtCandIsMatchedCand[MAXL1OBJS];
  bool l1GmtCandIsHaloCand[MAXL1OBJS];
  bool l1GmtCandIsol[MAXL1OBJS];
  bool l1GmtCandMip[MAXL1OBJS];
  int l1GmtCandQuality[MAXL1OBJS];
  int l1GmtCandBx[MAXL1OBJS];

};







// ------------------------------------------------------------------------
//! branch addresses settings

void setBranchAddresses(TTree* chain, EcalTimeTreeContent& treeVars);






// ------------------------------------------------------------------------
//! create branches for a tree

void setBranches(TTree* chain, EcalTimeTreeContent& treeVars);






// ------------------------------------------------------------------------
//! initialize branches

void initializeBranches(TTree* chain, EcalTimeTreeContent& treeVars);



#endif
