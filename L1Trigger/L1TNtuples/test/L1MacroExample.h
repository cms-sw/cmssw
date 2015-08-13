//////////////////////////////////////
// Example root macro for l1 ntuples
//////////////////////////////////////

#ifndef l1macroexample_h
#define l1macroexample_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TFriendElement.h>
#include <iostream>
#include <TList.h>

   const Int_t kMaxverbose = 1;
   const Int_t kMaxmaxRPC = 1;
   const Int_t kMaxmaxDTBX = 1;
   const Int_t kMaxmaxCSC = 1;
   const Int_t kMaxmaxGMT = 1;
   const Int_t kMaxphysVal = 1;
   const Int_t kMaxmaxRCTREG = 1;
   const Int_t kMaxmaxDTPH = 1;
   const Int_t kMaxmaxDTTH = 1;
   const Int_t kMaxmaxDTTR = 1;
   const Int_t kMaxmaxCSCTFTR = 1;
   const Int_t kMaxmaxCSCTFLCTSTR = 1;
   const Int_t kMaxmaxCSCTFLCTS = 1;
   const Int_t kMaxmaxCSCTFSPS = 1;
   
   const Int_t kMaxmaxJet = 1;
   const Int_t kMaxMAXCl = 1;
   const Int_t kMaxMAXCl = 1;

class L1MacroExample {
public:
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  
  bool doreco;
  bool domuonreco;
  bool dol1extra;
  
  bool dofits;
  
  // Declaration of leaf types for L1 ntuple
  //PromptL1::L1Event *Event;
   Int_t           run;
   Int_t           event;
   Int_t           lumi;
   Int_t           bx;
   ULong64_t       orbit;
   ULong64_t       time;
 //PromptL1::L1GCT *GCT;
   Bool_t          verbose_;
   Int_t           gctIsoEmSize;
   vector<float>   gctIsoEmEta;
   vector<float>   gctIsoEmPhi;
   vector<float>   gctIsoEmRnk;
   Int_t           gctNonIsoEmSize;
   vector<float>   gctNonIsoEmEta;
   vector<float>   gctNonIsoEmPhi;
   vector<float>   gctNonIsoEmRnk;
   Int_t           gctCJetSize;
   vector<float>   gctCJetEta;
   vector<float>   gctCJetPhi;
   vector<float>   gctCJetRnk;
   Int_t           gctFJetSize;
   vector<float>   gctFJetEta;
   vector<float>   gctFJetPhi;
   vector<float>   gctFJetRnk;
   Int_t           gctTJetSize;
   vector<float>   gctTJetEta;
   vector<float>   gctTJetPhi;
   vector<float>   gctTJetRnk;
   Float_t         gctEtMiss;
   Float_t         gctEtMissPhi;
   Float_t         gctHtMiss;
   Float_t         gctHtMissPhi;
   Float_t         gctEtHad;
   Float_t         gctEtTot;
   Int_t           gctHFRingEtSumSize;
   vector<float>   gctHFRingEtSumEta;
   Float_t         gctHFBitCountsSize;
   vector<float>   gctHFBitCountsEta;
 //PromptL1::L1GMT *GMT;
   Int_t           maxRPC_;
   Int_t           maxDTBX_;
   Int_t           maxCSC_;
   Int_t           maxGMT_;
   Bool_t          physVal_;
   Int_t           gmtEvBx;
   Int_t           gmtNdt;
   vector<int>     gmtBxdt;
   vector<float>   gmtPtdt;
   vector<int>     gmtChadt;
   vector<float>   gmtEtadt;
   vector<int>     gmtFineEtadt;
   vector<float>   gmtPhidt;
   vector<int>     gmtQualdt;
   vector<int>     gmtDwdt;
   vector<int>     gmtChdt;
   Int_t           gmtNcsc;
   vector<int>     gmtBxcsc;
   vector<float>   gmtPtcsc;
   vector<int>     gmtChacsc;
   vector<float>   gmtEtacsc;
   vector<float>   gmtPhicsc;
   vector<int>     gmtQualcsc;
   vector<int>     gmtDwcsc;
   Int_t           gmtNrpcb;
   vector<int>     gmtBxrpcb;
   vector<float>   gmtPtrpcb;
   vector<int>     gmtCharpcb;
   vector<float>   gmtEtarpcb;
   vector<float>   gmtPhirpcb;
   vector<int>     gmtQualrpcb;
   vector<int>     gmtDwrpcb;
   Int_t           gmtNrpcf;
   vector<int>     gmtBxrpcf;
   vector<float>   gmtPtrpcf;
   vector<int>     gmtCharpcf;
   vector<float>   gmtEtarpcf;
   vector<float>   gmtPhirpcf;
   vector<int>     gmtQualrpcf;
   vector<int>     gmtDwrpcf;
   Int_t           gmtN;
   vector<int>     gmtCandBx;
   vector<float>   gmtPt;
   vector<int>     gmtCha;
   vector<float>   gmtEta;
   vector<float>   gmtPhi;
   vector<int>     gmtQual;
   vector<int>     gmtDet;
   vector<int>     gmtRank;
   vector<int>     gmtIsol;
   vector<int>     gmtMip;
   vector<int>     gmtDw;
   vector<int>     gmtIdxRPCb;
   vector<int>     gmtIdxRPCf;
   vector<int>     gmtIdxDTBX;
   vector<int>     gmtIdxCSC;
 //PromptL1::L1GT  *GT;
 //vector<unsigned long long> gttw1;
 //vector<unsigned long long> gttw2;
 //vector<unsigned long long> gttt;
   Int_t           gtNele;
   vector<int>     gtBxel;
   vector<float>   gtRankel;
   vector<float>   gtPhiel;
   vector<float>   gtEtael;
 //vector<bool>    gtIsoel;
   Int_t           gtNjet;
   vector<int>     gtBxjet;
   vector<float>   gtRankjet;
   vector<float>   gtPhijet;
   vector<float>   gtEtajet;
 //vector<bool>    gtTaujet;
 //vector<bool>    gtFwdjet;
 //PromptL1::L1RCT *RCT;
   Int_t           maxRCTREG_;
   Int_t           rctRegSize;
   vector<float>   rctRegEta;
   vector<float>   rctRegPhi;
   vector<float>   rctRegRnk;
   vector<int>     rctRegVeto;
   vector<int>     rctRegBx;
   vector<int>     rctRegOverFlow;
   vector<int>     rctRegMip;
   vector<int>     rctRegFGrain;
   Int_t           rctEmSize;
   vector<int>     rctIsIsoEm;
   vector<float>   rctEmEta;
   vector<float>   rctEmPhi;
   vector<float>   rctEmRnk;
   vector<int>     rctEmBx;
 //PromptL1::L1DTTF *DTTF;
   Int_t           maxDTPH_;
   Int_t           maxDTTH_;
   Int_t           maxDTTR_;
   Int_t           dttf_phSize;
   vector<int>     dttf_phBx;
   vector<int>     dttf_phWh;
   vector<int>     dttf_phSe;
   vector<int>     dttf_phSt;
   vector<float>   dttf_phAng;
   vector<float>   dttf_phBandAng;
   vector<int>     dttf_phCode;
   vector<float>   dttf_phX;
   vector<float>   dttf_phY;
   Int_t           dttf_thSize;
   vector<int>     dttf_thBx;
   vector<int>     dttf_thWh;
   vector<int>     dttf_thSe;
   vector<int>     dttf_thSt;
   vector<float>   dttf_thX;
   vector<float>   dttf_thY;
   TMatrixT<double> dttf_thTheta;
   TMatrixT<double> dttf_thCode;
   Int_t           dttf_trSize;
   vector<int>     dttf_trBx;
   vector<int>     dttf_trTag;
   vector<int>     dttf_trQual;
   vector<int>     dttf_trPtPck;
   vector<float>   dttf_trPtVal;
   vector<int>     dttf_trPhiPck;
   vector<float>   dttf_trPhiVal;
   vector<int>     dttf_trPhiGlob;
   vector<int>     dttf_trChPck;
   vector<int>     dttf_trWh;
   vector<int>     dttf_trSc;
 //PromptL1::L1CSCTF *CSCTF;
   UInt_t          maxCSCTFTR_;
   UInt_t          maxCSCTFLCTSTR_;
   UInt_t          maxCSCTFLCTS_;
   UInt_t          maxCSCTFSPS_;
   Int_t           csctf_trSize;
   vector<int>     csctf_trEndcap;
   vector<int>     csctf_trSector;
   vector<int>     csctf_trBx;
   vector<int>     csctf_trME1ID;
   vector<int>     csctf_trME2ID;
   vector<int>     csctf_trME3ID;
   vector<int>     csctf_trME4ID;
   vector<int>     csctf_trMB1ID;
   vector<int>     csctf_trOutputLink;
   vector<int>     csctf_trCharge;
   vector<int>     csctf_trChargeValid;
   vector<int>     csctf_trForR;
   vector<int>     csctf_trPhi23;
   vector<int>     csctf_trPhi12;
   vector<int>     csctf_trPhiSign;
   vector<int>     csctf_trEtaBit;
   vector<int>     csctf_trPhiBit;
   vector<int>     csctf_trPtBit;
   vector<float>   csctf_trEta;
   vector<float>   csctf_trPhi;
   vector<float>   csctf_trPhi_02PI;
   vector<float>   csctf_trPt;
   vector<int>     csctf_trMode;
   vector<int>     csctf_trQuality;
   vector<int>     csctf_trNumLCTs;
   TMatrixT<double> csctf_trLctEndcap;
   TMatrixT<double> csctf_trLctSector;
   TMatrixT<double> csctf_trLctSubSector;
   TMatrixT<double> csctf_trLctBx;
   TMatrixT<double> csctf_trLctBx0;
   TMatrixT<double> csctf_trLctStation;
   TMatrixT<double> csctf_trLctRing;
   TMatrixT<double> csctf_trLctChamber;
   TMatrixT<double> csctf_trLctTriggerCSCID;
   TMatrixT<double> csctf_trLctFpga;
   TMatrixT<double> csctf_trLctlocalPhi;
   TMatrixT<double> csctf_trLctglobalPhi;
   TMatrixT<double> csctf_trLctglobalEta;
   TMatrixT<double> csctf_trLctstripNum;
   TMatrixT<double> csctf_trLctwireGroup;
   Int_t           csctf_lctSize;
   vector<int>     csctf_lctEndcap;
   vector<int>     csctf_lctSector;
   vector<int>     csctf_lctSubSector;
   vector<int>     csctf_lctBx;
   vector<int>     csctf_lctBx0;
   vector<int>     csctf_lctStation;
   vector<int>     csctf_lctRing;
   vector<int>     csctf_lctChamber;
   vector<int>     csctf_lctTriggerCSCID;
   vector<int>     csctf_lctFpga;
   vector<int>     csctf_lctlocalPhi;
   vector<int>     csctf_lctglobalPhi;
   vector<int>     csctf_lctglobalEta;
   vector<int>     csctf_lctstripNum;
   vector<int>     csctf_lctwireGroup;
   Int_t           csctf_nsp;
   vector<int>     csctf_stSPslot;
   vector<int>     csctf_stL1A_BXN;
   vector<unsigned long> csctf_stTrkCounter;
   vector<unsigned long> csctf_stOrbCounter;

  // List of branches for L1 NTUPLE
   TBranch        *b_Event_run;   //!
   TBranch        *b_Event_event;   //!
   TBranch        *b_Event_lumi;   //!
   TBranch        *b_Event_bx;   //!
   TBranch        *b_Event_orbit;   //!
   TBranch        *b_Event_time;   //!
   TBranch        *b_GCT_verbose_;   //!
   TBranch        *b_GCT_gctIsoEmSize;   //!
   TBranch        *b_GCT_gctIsoEmEta;   //!
   TBranch        *b_GCT_gctIsoEmPhi;   //!
   TBranch        *b_GCT_gctIsoEmRnk;   //!
   TBranch        *b_GCT_gctNonIsoEmSize;   //!
   TBranch        *b_GCT_gctNonIsoEmEta;   //!
   TBranch        *b_GCT_gctNonIsoEmPhi;   //!
   TBranch        *b_GCT_gctNonIsoEmRnk;   //!
   TBranch        *b_GCT_gctCJetSize;   //!
   TBranch        *b_GCT_gctCJetEta;   //!
   TBranch        *b_GCT_gctCJetPhi;   //!
   TBranch        *b_GCT_gctCJetRnk;   //!
   TBranch        *b_GCT_gctFJetSize;   //!
   TBranch        *b_GCT_gctFJetEta;   //!
   TBranch        *b_GCT_gctFJetPhi;   //!
   TBranch        *b_GCT_gctFJetRnk;   //!
   TBranch        *b_GCT_gctTJetSize;   //!
   TBranch        *b_GCT_gctTJetEta;   //!
   TBranch        *b_GCT_gctTJetPhi;   //!
   TBranch        *b_GCT_gctTJetRnk;   //!
   TBranch        *b_GCT_gctEtMiss;   //!
   TBranch        *b_GCT_gctEtMissPhi;   //!
   TBranch        *b_GCT_gctHtMiss;   //!
   TBranch        *b_GCT_gctHtMissPhi;   //!
   TBranch        *b_GCT_gctEtHad;   //!
   TBranch        *b_GCT_gctEtTot;   //!
   TBranch        *b_GCT_gctHFRingEtSumSize;   //!
   TBranch        *b_GCT_gctHFRingEtSumEta;   //!
   TBranch        *b_GCT_gctHFBitCountsSize;   //!
   TBranch        *b_GCT_gctHFBitCountsEta;   //!
   TBranch        *b_GMT_maxRPC_;   //!
   TBranch        *b_GMT_maxDTBX_;   //!
   TBranch        *b_GMT_maxCSC_;   //!
   TBranch        *b_GMT_maxGMT_;   //!
   TBranch        *b_GMT_physVal_;   //!
   TBranch        *b_GMT_gmtEvBx;   //!
   TBranch        *b_GMT_gmtNdt;   //!
   TBranch        *b_GMT_gmtBxdt;   //!
   TBranch        *b_GMT_gmtPtdt;   //!
   TBranch        *b_GMT_gmtChadt;   //!
   TBranch        *b_GMT_gmtEtadt;   //!
   TBranch        *b_GMT_gmtFineEtadt;   //!
   TBranch        *b_GMT_gmtPhidt;   //!
   TBranch        *b_GMT_gmtQualdt;   //!
   TBranch        *b_GMT_gmtDwdt;   //!
   TBranch        *b_GMT_gmtChdt;   //!
   TBranch        *b_GMT_gmtNcsc;   //!
   TBranch        *b_GMT_gmtBxcsc;   //!
   TBranch        *b_GMT_gmtPtcsc;   //!
   TBranch        *b_GMT_gmtChacsc;   //!
   TBranch        *b_GMT_gmtEtacsc;   //!
   TBranch        *b_GMT_gmtPhicsc;   //!
   TBranch        *b_GMT_gmtQualcsc;   //!
   TBranch        *b_GMT_gmtDwcsc;   //!
   TBranch        *b_GMT_gmtNrpcb;   //!
   TBranch        *b_GMT_gmtBxrpcb;   //!
   TBranch        *b_GMT_gmtPtrpcb;   //!
   TBranch        *b_GMT_gmtCharpcb;   //!
   TBranch        *b_GMT_gmtEtarpcb;   //!
   TBranch        *b_GMT_gmtPhirpcb;   //!
   TBranch        *b_GMT_gmtQualrpcb;   //!
   TBranch        *b_GMT_gmtDwrpcb;   //!
   TBranch        *b_GMT_gmtNrpcf;   //!
   TBranch        *b_GMT_gmtBxrpcf;   //!
   TBranch        *b_GMT_gmtPtrpcf;   //!
   TBranch        *b_GMT_gmtCharpcf;   //!
   TBranch        *b_GMT_gmtEtarpcf;   //!
   TBranch        *b_GMT_gmtPhirpcf;   //!
   TBranch        *b_GMT_gmtQualrpcf;   //!
   TBranch        *b_GMT_gmtDwrpcf;   //!
   TBranch        *b_GMT_gmtN;   //!
   TBranch        *b_GMT_gmtCandBx;   //!
   TBranch        *b_GMT_gmtPt;   //!
   TBranch        *b_GMT_gmtCha;   //!
   TBranch        *b_GMT_gmtEta;   //!
   TBranch        *b_GMT_gmtPhi;   //!
   TBranch        *b_GMT_gmtQual;   //!
   TBranch        *b_GMT_gmtDet;   //!
   TBranch        *b_GMT_gmtRank;   //!
   TBranch        *b_GMT_gmtIsol;   //!
   TBranch        *b_GMT_gmtMip;   //!
   TBranch        *b_GMT_gmtDw;   //!
   TBranch        *b_GMT_gmtIdxRPCb;   //!
   TBranch        *b_GMT_gmtIdxRPCf;   //!
   TBranch        *b_GMT_gmtIdxDTBX;   //!
   TBranch        *b_GMT_gmtIdxCSC;   //!
   TBranch        *b_GT_gtNele;   //!
   TBranch        *b_GT_gtBxel;   //!
   TBranch        *b_GT_gtRankel;   //!
   TBranch        *b_GT_gtPhiel;   //!
   TBranch        *b_GT_gtEtael;   //!
   TBranch        *b_GT_gtNjet;   //!
   TBranch        *b_GT_gtBxjet;   //!
   TBranch        *b_GT_gtRankjet;   //!
   TBranch        *b_GT_gtPhijet;   //!
   TBranch        *b_GT_gtEtajet;   //!
   TBranch        *b_RCT_maxRCTREG_;   //!
   TBranch        *b_RCT_rctRegSize;   //!
   TBranch        *b_RCT_rctRegEta;   //!
   TBranch        *b_RCT_rctRegPhi;   //!
   TBranch        *b_RCT_rctRegRnk;   //!
   TBranch        *b_RCT_rctRegVeto;   //!
   TBranch        *b_RCT_rctRegBx;   //!
   TBranch        *b_RCT_rctRegOverFlow;   //!
   TBranch        *b_RCT_rctRegMip;   //!
   TBranch        *b_RCT_rctRegFGrain;   //!
   TBranch        *b_RCT_rctEmSize;   //!
   TBranch        *b_RCT_rctIsIsoEm;   //!
   TBranch        *b_RCT_rctEmEta;   //!
   TBranch        *b_RCT_rctEmPhi;   //!
   TBranch        *b_RCT_rctEmRnk;   //!
   TBranch        *b_RCT_rctEmBx;   //!
   TBranch        *b_DTTF_maxDTPH_;   //!
   TBranch        *b_DTTF_maxDTTH_;   //!
   TBranch        *b_DTTF_maxDTTR_;   //!
   TBranch        *b_DTTF_dttf_phSize;   //!
   TBranch        *b_DTTF_dttf_phBx;   //!
   TBranch        *b_DTTF_dttf_phWh;   //!
   TBranch        *b_DTTF_dttf_phSe;   //!
   TBranch        *b_DTTF_dttf_phSt;   //!
   TBranch        *b_DTTF_dttf_phAng;   //!
   TBranch        *b_DTTF_dttf_phBandAng;   //!
   TBranch        *b_DTTF_dttf_phCode;   //!
   TBranch        *b_DTTF_dttf_phX;   //!
   TBranch        *b_DTTF_dttf_phY;   //!
   TBranch        *b_DTTF_dttf_thSize;   //!
   TBranch        *b_DTTF_dttf_thBx;   //!
   TBranch        *b_DTTF_dttf_thWh;   //!
   TBranch        *b_DTTF_dttf_thSe;   //!
   TBranch        *b_DTTF_dttf_thSt;   //!
   TBranch        *b_DTTF_dttf_thX;   //!
   TBranch        *b_DTTF_dttf_thY;   //!
   TBranch        *b_DTTF_dttf_thTheta;   //!
   TBranch        *b_DTTF_dttf_thCode;   //!
   TBranch        *b_DTTF_dttf_trSize;   //!
   TBranch        *b_DTTF_dttf_trBx;   //!
   TBranch        *b_DTTF_dttf_trTag;   //!
   TBranch        *b_DTTF_dttf_trQual;   //!
   TBranch        *b_DTTF_dttf_trPtPck;   //!
   TBranch        *b_DTTF_dttf_trPtVal;   //!
   TBranch        *b_DTTF_dttf_trPhiPck;   //!
   TBranch        *b_DTTF_dttf_trPhiVal;   //!
   TBranch        *b_DTTF_dttf_trPhiGlob;   //!
   TBranch        *b_DTTF_dttf_trChPck;   //!
   TBranch        *b_DTTF_dttf_trWh;   //!
   TBranch        *b_DTTF_dttf_trSc;   //!
   TBranch        *b_CSCTF_maxCSCTFTR_;   //!
   TBranch        *b_CSCTF_maxCSCTFLCTSTR_;   //!
   TBranch        *b_CSCTF_maxCSCTFLCTS_;   //!
   TBranch        *b_CSCTF_maxCSCTFSPS_;   //!
   TBranch        *b_CSCTF_csctf_trSize;   //!
   TBranch        *b_CSCTF_csctf_trEndcap;   //!
   TBranch        *b_CSCTF_csctf_trSector;   //!
   TBranch        *b_CSCTF_csctf_trBx;   //!
   TBranch        *b_CSCTF_csctf_trME1ID;   //!
   TBranch        *b_CSCTF_csctf_trME2ID;   //!
   TBranch        *b_CSCTF_csctf_trME3ID;   //!
   TBranch        *b_CSCTF_csctf_trME4ID;   //!
   TBranch        *b_CSCTF_csctf_trMB1ID;   //!
   TBranch        *b_CSCTF_csctf_trOutputLink;   //!
   TBranch        *b_CSCTF_csctf_trCharge;   //!
   TBranch        *b_CSCTF_csctf_trChargeValid;   //!
   TBranch        *b_CSCTF_csctf_trForR;   //!
   TBranch        *b_CSCTF_csctf_trPhi23;   //!
   TBranch        *b_CSCTF_csctf_trPhi12;   //!
   TBranch        *b_CSCTF_csctf_trPhiSign;   //!
   TBranch        *b_CSCTF_csctf_trEtaBit;   //!
   TBranch        *b_CSCTF_csctf_trPhiBit;   //!
   TBranch        *b_CSCTF_csctf_trPtBit;   //!
   TBranch        *b_CSCTF_csctf_trEta;   //!
   TBranch        *b_CSCTF_csctf_trPhi;   //!
   TBranch        *b_CSCTF_csctf_trPhi_02PI;   //!
   TBranch        *b_CSCTF_csctf_trPt;   //!
   TBranch        *b_CSCTF_csctf_trMode;   //!
   TBranch        *b_CSCTF_csctf_trQuality;   //!
   TBranch        *b_CSCTF_csctf_trNumLCTs;   //!
   TBranch        *b_CSCTF_csctf_trLctEndcap;   //!
   TBranch        *b_CSCTF_csctf_trLctSector;   //!
   TBranch        *b_CSCTF_csctf_trLctSubSector;   //!
   TBranch        *b_CSCTF_csctf_trLctBx;   //!
   TBranch        *b_CSCTF_csctf_trLctBx0;   //!
   TBranch        *b_CSCTF_csctf_trLctStation;   //!
   TBranch        *b_CSCTF_csctf_trLctRing;   //!
   TBranch        *b_CSCTF_csctf_trLctChamber;   //!
   TBranch        *b_CSCTF_csctf_trLctTriggerCSCID;   //!
   TBranch        *b_CSCTF_csctf_trLctFpga;   //!
   TBranch        *b_CSCTF_csctf_trLctlocalPhi;   //!
   TBranch        *b_CSCTF_csctf_trLctglobalPhi;   //!
   TBranch        *b_CSCTF_csctf_trLctglobalEta;   //!
   TBranch        *b_CSCTF_csctf_trLctstripNum;   //!
   TBranch        *b_CSCTF_csctf_trLctwireGroup;   //!
   TBranch        *b_CSCTF_csctf_lctSize;   //!
   TBranch        *b_CSCTF_csctf_lctEndcap;   //!
   TBranch        *b_CSCTF_csctf_lctSector;   //!
   TBranch        *b_CSCTF_csctf_lctSubSector;   //!
   TBranch        *b_CSCTF_csctf_lctBx;   //!
   TBranch        *b_CSCTF_csctf_lctBx0;   //!
   TBranch        *b_CSCTF_csctf_lctStation;   //!
   TBranch        *b_CSCTF_csctf_lctRing;   //!
   TBranch        *b_CSCTF_csctf_lctChamber;   //!
   TBranch        *b_CSCTF_csctf_lctTriggerCSCID;   //!
   TBranch        *b_CSCTF_csctf_lctFpga;   //!
   TBranch        *b_CSCTF_csctf_lctlocalPhi;   //!
   TBranch        *b_CSCTF_csctf_lctglobalPhi;   //!
   TBranch        *b_CSCTF_csctf_lctglobalEta;   //!
   TBranch        *b_CSCTF_csctf_lctstripNum;   //!
   TBranch        *b_CSCTF_csctf_lctwireGroup;   //!
   TBranch        *b_CSCTF_csctf_nsp;   //!
   TBranch        *b_CSCTF_csctf_stSPslot;   //!
   TBranch        *b_CSCTF_csctf_stL1A_BXN;   //!
   TBranch        *b_CSCTF_csctf_stTrkCounter;   //!
   TBranch        *b_CSCTF_csctf_stOrbCounter;   //!

 // Declaration of leaf types for muon reco
  //PromptL1::Muon  *Muon;
   Int_t           nMuons;
   vector<int>     muon_type;
   vector<double>  muons_ch;
   vector<double>  muons_pt;
   vector<double>  muons_p;
   vector<double>  muons_eta;
   vector<double>  muons_phi;
   vector<double>  muons_validhits;
   vector<double>  muons_normchi2;
   vector<double>  muons_imp_point_x;
   vector<double>  muons_imp_point_y;
   vector<double>  muons_imp_point_z;
   vector<double>  muons_imp_point_p;
   vector<double>  muons_imp_point_pt;
   vector<double>  muons_phi_hb;
   vector<double>  muons_z_hb;
   vector<double>  muons_r_he_p;
   vector<double>  muons_r_he_n;
   vector<double>  muons_phi_he_p;
   vector<double>  muons_phi_he_n;
   vector<double>  muons_tr_ch;
   vector<double>  muons_tr_pt;
   vector<double>  muons_tr_p;
   vector<double>  muons_tr_eta;
   vector<double>  muons_tr_phi;
   vector<double>  muons_tr_validhits;
   vector<double>  muons_tr_normchi2;
   vector<double>  muons_tr_imp_point_x;
   vector<double>  muons_tr_imp_point_y;
   vector<double>  muons_tr_imp_point_z;
   vector<double>  muons_tr_imp_point_p;
   vector<double>  muons_tr_imp_point_pt;
   vector<double>  muons_sa_phi_mb2;
   vector<double>  muons_sa_z_mb2;
   vector<double>  muons_sa_pseta;
   vector<double>  muons_sa_normchi2;
   vector<double>  muons_sa_validhits;
   vector<double>  muons_sa_ch;
   vector<double>  muons_sa_pt;
   vector<double>  muons_sa_p;
   vector<double>  muons_sa_eta;
   vector<double>  muons_sa_phi;
   vector<double>  muons_sa_outer_pt;
   vector<double>  muons_sa_inner_pt;
   vector<double>  muons_sa_outer_eta;
   vector<double>  muons_sa_inner_eta;
   vector<double>  muons_sa_outer_phi;
   vector<double>  muons_sa_inner_phi;
   vector<double>  muons_sa_outer_x;
   vector<double>  muons_sa_outer_y;
   vector<double>  muons_sa_outer_z;
   vector<double>  muons_sa_inner_x;
   vector<double>  muons_sa_inner_y;
   vector<double>  muons_sa_inner_z;
   vector<double>  muons_sa_imp_point_x;
   vector<double>  muons_sa_imp_point_y;
   vector<double>  muons_sa_imp_point_z;
   vector<double>  muons_sa_imp_point_p;
   vector<double>  muons_sa_imp_point_pt;
   vector<double>  muons_sa_phi_hb;
   vector<double>  muons_sa_z_hb;
   vector<double>  muons_sa_r_he_p;
   vector<double>  muons_sa_r_he_n;
   vector<double>  muons_sa_phi_he_p;
   vector<double>  muons_sa_phi_he_n;

   // List of branches for muon reco
  TBranch        *b_Muon_nMuons;   //!
   TBranch        *b_Muon_muon_type;   //!
   TBranch        *b_Muon_muons_ch;   //!
   TBranch        *b_Muon_muons_pt;   //!
   TBranch        *b_Muon_muons_p;   //!
   TBranch        *b_Muon_muons_eta;   //!
   TBranch        *b_Muon_muons_phi;   //!
   TBranch        *b_Muon_muons_validhits;   //!
   TBranch        *b_Muon_muons_normchi2;   //!
   TBranch        *b_Muon_muons_imp_point_x;   //!
   TBranch        *b_Muon_muons_imp_point_y;   //!
   TBranch        *b_Muon_muons_imp_point_z;   //!
   TBranch        *b_Muon_muons_imp_point_p;   //!
   TBranch        *b_Muon_muons_imp_point_pt;   //!
   TBranch        *b_Muon_muons_phi_hb;   //!
   TBranch        *b_Muon_muons_z_hb;   //!
   TBranch        *b_Muon_muons_r_he_p;   //!
   TBranch        *b_Muon_muons_r_he_n;   //!
   TBranch        *b_Muon_muons_phi_he_p;   //!
   TBranch        *b_Muon_muons_phi_he_n;   //!
   TBranch        *b_Muon_muons_tr_ch;   //!
   TBranch        *b_Muon_muons_tr_pt;   //!
   TBranch        *b_Muon_muons_tr_p;   //!
   TBranch        *b_Muon_muons_tr_eta;   //!
   TBranch        *b_Muon_muons_tr_phi;   //!
   TBranch        *b_Muon_muons_tr_validhits;   //!
   TBranch        *b_Muon_muons_tr_normchi2;   //!
   TBranch        *b_Muon_muons_tr_imp_point_x;   //!
   TBranch        *b_Muon_muons_tr_imp_point_y;   //!
   TBranch        *b_Muon_muons_tr_imp_point_z;   //!
   TBranch        *b_Muon_muons_tr_imp_point_p;   //!
   TBranch        *b_Muon_muons_tr_imp_point_pt;   //!
   TBranch        *b_Muon_muons_sa_phi_mb2;   //!
   TBranch        *b_Muon_muons_sa_z_mb2;   //!
   TBranch        *b_Muon_muons_sa_pseta;   //!
   TBranch        *b_Muon_muons_sa_normchi2;   //!
   TBranch        *b_Muon_muons_sa_validhits;   //!
   TBranch        *b_Muon_muons_sa_ch;   //!
   TBranch        *b_Muon_muons_sa_pt;   //!
   TBranch        *b_Muon_muons_sa_p;   //!
   TBranch        *b_Muon_muons_sa_eta;   //!
   TBranch        *b_Muon_muons_sa_phi;   //!
   TBranch        *b_Muon_muons_sa_outer_pt;   //!
   TBranch        *b_Muon_muons_sa_inner_pt;   //!
   TBranch        *b_Muon_muons_sa_outer_eta;   //!
   TBranch        *b_Muon_muons_sa_inner_eta;   //!
   TBranch        *b_Muon_muons_sa_outer_phi;   //!
   TBranch        *b_Muon_muons_sa_inner_phi;   //!
   TBranch        *b_Muon_muons_sa_outer_x;   //!
   TBranch        *b_Muon_muons_sa_outer_y;   //!
   TBranch        *b_Muon_muons_sa_outer_z;   //!
   TBranch        *b_Muon_muons_sa_inner_x;   //!
   TBranch        *b_Muon_muons_sa_inner_y;   //!
   TBranch        *b_Muon_muons_sa_inner_z;   //!
   TBranch        *b_Muon_muons_sa_imp_point_x;   //!
   TBranch        *b_Muon_muons_sa_imp_point_y;   //!
   TBranch        *b_Muon_muons_sa_imp_point_z;   //!
   TBranch        *b_Muon_muons_sa_imp_point_p;   //!
   TBranch        *b_Muon_muons_sa_imp_point_pt;   //!
   TBranch        *b_Muon_muons_sa_phi_hb;   //!
   TBranch        *b_Muon_muons_sa_z_hb;   //!
   TBranch        *b_Muon_muons_sa_r_he_p;   //!
   TBranch        *b_Muon_muons_sa_r_he_n;   //!
   TBranch        *b_Muon_muons_sa_phi_he_p;   //!
   TBranch        *b_Muon_muons_sa_phi_he_n;   //!

  // Declaration of leaf types for reco
   //PromptL1::Jet   *Jet;
   UInt_t          maxJet_;
   UInt_t          nJets;
   vector<double>  e;
   vector<double>  et;
   vector<double>  eta;
   vector<double>  phi;
   vector<double>  eEMF;
   vector<double>  eHadHB;
   vector<double>  eHadHE;
   vector<double>  eHadHO;
   vector<double>  eHadHF;
   vector<double>  eEmEB;
   vector<double>  eEmEE;
   vector<double>  eEmHF;
   vector<double>  eMaxEcalTow;
   vector<double>  eMaxHcalTow;
   vector<double>  towerArea;
   vector<int>     towerSize;
   vector<int>     n60;
   vector<int>     n90;
 //PromptL1::Met   *Met;
   Double_t        met;
   Double_t        metPhi;
   Double_t        Ht;
   Double_t        mHt;
   Double_t        mHtPhi;
   Double_t        sumEt;
 //PromptL1::Clusters *SuperClusters;
   UInt_t          MAXCl_;
   UInt_t          nClusters;
   vector<double>  clusterEta;
   vector<double>  clusterPhi;
   vector<double>  clusterEt;
   vector<double>  clusterE;
 //PromptL1::Clusters *BasicClusters;
   UInt_t          MAXCl_;
   UInt_t          nClusters;
   vector<double>  clusterEta;
   vector<double>  clusterPhi;
   vector<double>  clusterEt;
   vector<double>  clusterE;

  
  // List of branches for jet reco
   TBranch        *b_Jet_maxJet_;   //!
   TBranch        *b_Jet_nJets;   //!
   TBranch        *b_Jet_e;   //!
   TBranch        *b_Jet_et;   //!
   TBranch        *b_Jet_eta;   //!
   TBranch        *b_Jet_phi;   //!
   TBranch        *b_Jet_eEMF;   //!
   TBranch        *b_Jet_eHadHB;   //!
   TBranch        *b_Jet_eHadHE;   //!
   TBranch        *b_Jet_eHadHO;   //!
   TBranch        *b_Jet_eHadHF;   //!
   TBranch        *b_Jet_eEmEB;   //!
   TBranch        *b_Jet_eEmEE;   //!
   TBranch        *b_Jet_eEmHF;   //!
   TBranch        *b_Jet_eMaxEcalTow;   //!
   TBranch        *b_Jet_eMaxHcalTow;   //!
   TBranch        *b_Jet_towerArea;   //!
   TBranch        *b_Jet_towerSize;   //!
   TBranch        *b_Jet_n60;   //!
   TBranch        *b_Jet_n90;   //!
   TBranch        *b_Met_met;   //!
   TBranch        *b_Met_metPhi;   //!
   TBranch        *b_Met_Ht;   //!
   TBranch        *b_Met_mHt;   //!
   TBranch        *b_Met_mHtPhi;   //!
   TBranch        *b_Met_sumEt;   //!
   TBranch        *b_SuperClusters_MAXCl_;   //!
   TBranch        *b_SuperClusters_nClusters;   //!
   TBranch        *b_SuperClusters_clusterEta;   //!
   TBranch        *b_SuperClusters_clusterPhi;   //!
   TBranch        *b_SuperClusters_clusterEt;   //!
   TBranch        *b_SuperClusters_clusterE;   //!
   TBranch        *b_BasicClusters_MAXCl_;   //!
   TBranch        *b_BasicClusters_nClusters;   //!
   TBranch        *b_BasicClusters_clusterEta;   //!
   TBranch        *b_BasicClusters_clusterPhi;   //!
   TBranch        *b_BasicClusters_clusterEt;   //!
   TBranch        *b_BasicClusters_clusterE;   //!
  
  L1MacroExample(char *fname="", bool dofits_=false);

  TFile* rf;

  virtual ~L1MacroExample();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual void     Loop();
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef l1macroexample_cxx

L1MacroExample::L1MacroExample(char* fname, bool dofits_)
{
  dofits = dofits_;
  doreco = true;
  domuonreco = true;
  dol1extra = true;

  if (fname == "") {
    rf = new TFile("L1Tree.root");
  } else {
    rf = new TFile(fname);
  }
  TTree* tree = new TTree;
  tree = (TTree*)rf->Get("l1NtupleProducer/L1Tree");
  TTree* ftreemuon = new TTree;
  TTree* ftreejets = new TTree;
  ftreemuon = (TTree*)rf->Get("l1MuonRecoTreeProducer/MuonRecoTree");
  ftreejets = (TTree*)rf->Get("l1RecoTreeProducer/RecoTree");
  if (!ftreejets) {
    std::cout<<"RecoTree not found, it will be skipped..."<<std::endl;
    doreco=false;
  } else {
    tree->AddFriend(ftreejets);
    std::cout<<"RecoTree added as friend tree..."<<std::endl;
  }
  if (!ftreemuon) {
    std::cout<<"MuonRecoTree not found, it will be skipped..."<<std::endl;
    domuonreco=false;
  } else {
    tree->AddFriend(ftreemuon);    
    std::cout<<"MuonRecoTree added as friend tree..."<<std::endl;
  }
  dol1extra=false;                        // add the l1 extra tree check here
  Init(tree);
}

L1MacroExample::~L1MacroExample()
{
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t L1MacroExample::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}

Long64_t L1MacroExample::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (!fChain->InheritsFrom(TChain::Class()))  return centry;
   TChain *chain = (TChain*)fChain;
   if (chain->GetTreeNumber() != fCurrent) {
      fCurrent = chain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void L1MacroExample::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   std::cout<<"Setting branch addresses for L1... ";
   fChain->SetBranchAddress("run", &run, &b_Event_run);
   fChain->SetBranchAddress("event", &event, &b_Event_event);
   fChain->SetBranchAddress("lumi", &lumi, &b_Event_lumi);
   fChain->SetBranchAddress("bx", &bx, &b_Event_bx);
   fChain->SetBranchAddress("orbit", &orbit, &b_Event_orbit);
   fChain->SetBranchAddress("time", &time, &b_Event_time);
   fChain->SetBranchAddress("verbose_", &verbose_, &b_GCT_verbose_);
   fChain->SetBranchAddress("gctIsoEmSize", &gctIsoEmSize, &b_GCT_gctIsoEmSize);
   fChain->SetBranchAddress("gctIsoEmEta", &gctIsoEmEta, &b_GCT_gctIsoEmEta);
   fChain->SetBranchAddress("gctIsoEmPhi", &gctIsoEmPhi, &b_GCT_gctIsoEmPhi);
   fChain->SetBranchAddress("gctIsoEmRnk", &gctIsoEmRnk, &b_GCT_gctIsoEmRnk);
   fChain->SetBranchAddress("gctNonIsoEmSize", &gctNonIsoEmSize, &b_GCT_gctNonIsoEmSize);
   fChain->SetBranchAddress("gctNonIsoEmEta", &gctNonIsoEmEta, &b_GCT_gctNonIsoEmEta);
   fChain->SetBranchAddress("gctNonIsoEmPhi", &gctNonIsoEmPhi, &b_GCT_gctNonIsoEmPhi);
   fChain->SetBranchAddress("gctNonIsoEmRnk", &gctNonIsoEmRnk, &b_GCT_gctNonIsoEmRnk);
   fChain->SetBranchAddress("gctCJetSize", &gctCJetSize, &b_GCT_gctCJetSize);
   fChain->SetBranchAddress("gctCJetEta", &gctCJetEta, &b_GCT_gctCJetEta);
   fChain->SetBranchAddress("gctCJetPhi", &gctCJetPhi, &b_GCT_gctCJetPhi);
   fChain->SetBranchAddress("gctCJetRnk", &gctCJetRnk, &b_GCT_gctCJetRnk);
   fChain->SetBranchAddress("gctFJetSize", &gctFJetSize, &b_GCT_gctFJetSize);
   fChain->SetBranchAddress("gctFJetEta", &gctFJetEta, &b_GCT_gctFJetEta);
   fChain->SetBranchAddress("gctFJetPhi", &gctFJetPhi, &b_GCT_gctFJetPhi);
   fChain->SetBranchAddress("gctFJetRnk", &gctFJetRnk, &b_GCT_gctFJetRnk);
   fChain->SetBranchAddress("gctTJetSize", &gctTJetSize, &b_GCT_gctTJetSize);
   fChain->SetBranchAddress("gctTJetEta", &gctTJetEta, &b_GCT_gctTJetEta);
   fChain->SetBranchAddress("gctTJetPhi", &gctTJetPhi, &b_GCT_gctTJetPhi);
   fChain->SetBranchAddress("gctTJetRnk", &gctTJetRnk, &b_GCT_gctTJetRnk);
   fChain->SetBranchAddress("gctEtMiss", &gctEtMiss, &b_GCT_gctEtMiss);
   fChain->SetBranchAddress("gctEtMissPhi", &gctEtMissPhi, &b_GCT_gctEtMissPhi);
   fChain->SetBranchAddress("gctHtMiss", &gctHtMiss, &b_GCT_gctHtMiss);
   fChain->SetBranchAddress("gctHtMissPhi", &gctHtMissPhi, &b_GCT_gctHtMissPhi);
   fChain->SetBranchAddress("gctEtHad", &gctEtHad, &b_GCT_gctEtHad);
   fChain->SetBranchAddress("gctEtTot", &gctEtTot, &b_GCT_gctEtTot);
   fChain->SetBranchAddress("gctHFRingEtSumSize", &gctHFRingEtSumSize, &b_GCT_gctHFRingEtSumSize);
   fChain->SetBranchAddress("gctHFRingEtSumEta", &gctHFRingEtSumEta, &b_GCT_gctHFRingEtSumEta);
   fChain->SetBranchAddress("gctHFBitCountsSize", &gctHFBitCountsSize, &b_GCT_gctHFBitCountsSize);
   fChain->SetBranchAddress("gctHFBitCountsEta", &gctHFBitCountsEta, &b_GCT_gctHFBitCountsEta);
   fChain->SetBranchAddress("maxRPC_", &maxRPC_, &b_GMT_maxRPC_);
   fChain->SetBranchAddress("maxDTBX_", &maxDTBX_, &b_GMT_maxDTBX_);
   fChain->SetBranchAddress("maxCSC_", &maxCSC_, &b_GMT_maxCSC_);
   fChain->SetBranchAddress("maxGMT_", &maxGMT_, &b_GMT_maxGMT_);
   fChain->SetBranchAddress("physVal_", &physVal_, &b_GMT_physVal_);
   fChain->SetBranchAddress("gmtEvBx", &gmtEvBx, &b_GMT_gmtEvBx);
   fChain->SetBranchAddress("gmtNdt", &gmtNdt, &b_GMT_gmtNdt);
   fChain->SetBranchAddress("gmtBxdt", &gmtBxdt, &b_GMT_gmtBxdt);
   fChain->SetBranchAddress("gmtPtdt", &gmtPtdt, &b_GMT_gmtPtdt);
   fChain->SetBranchAddress("gmtChadt", &gmtChadt, &b_GMT_gmtChadt);
   fChain->SetBranchAddress("gmtEtadt", &gmtEtadt, &b_GMT_gmtEtadt);
   fChain->SetBranchAddress("gmtFineEtadt", &gmtFineEtadt, &b_GMT_gmtFineEtadt);
   fChain->SetBranchAddress("gmtPhidt", &gmtPhidt, &b_GMT_gmtPhidt);
   fChain->SetBranchAddress("gmtQualdt", &gmtQualdt, &b_GMT_gmtQualdt);
   fChain->SetBranchAddress("gmtDwdt", &gmtDwdt, &b_GMT_gmtDwdt);
   fChain->SetBranchAddress("gmtChdt", &gmtChdt, &b_GMT_gmtChdt);
   fChain->SetBranchAddress("gmtNcsc", &gmtNcsc, &b_GMT_gmtNcsc);
   fChain->SetBranchAddress("gmtBxcsc", &gmtBxcsc, &b_GMT_gmtBxcsc);
   fChain->SetBranchAddress("gmtPtcsc", &gmtPtcsc, &b_GMT_gmtPtcsc);
   fChain->SetBranchAddress("gmtChacsc", &gmtChacsc, &b_GMT_gmtChacsc);
   fChain->SetBranchAddress("gmtEtacsc", &gmtEtacsc, &b_GMT_gmtEtacsc);
   fChain->SetBranchAddress("gmtPhicsc", &gmtPhicsc, &b_GMT_gmtPhicsc);
   fChain->SetBranchAddress("gmtQualcsc", &gmtQualcsc, &b_GMT_gmtQualcsc);
   fChain->SetBranchAddress("gmtDwcsc", &gmtDwcsc, &b_GMT_gmtDwcsc);
   fChain->SetBranchAddress("gmtNrpcb", &gmtNrpcb, &b_GMT_gmtNrpcb);
   fChain->SetBranchAddress("gmtBxrpcb", &gmtBxrpcb, &b_GMT_gmtBxrpcb);
   fChain->SetBranchAddress("gmtPtrpcb", &gmtPtrpcb, &b_GMT_gmtPtrpcb);
   fChain->SetBranchAddress("gmtCharpcb", &gmtCharpcb, &b_GMT_gmtCharpcb);
   fChain->SetBranchAddress("gmtEtarpcb", &gmtEtarpcb, &b_GMT_gmtEtarpcb);
   fChain->SetBranchAddress("gmtPhirpcb", &gmtPhirpcb, &b_GMT_gmtPhirpcb);
   fChain->SetBranchAddress("gmtQualrpcb", &gmtQualrpcb, &b_GMT_gmtQualrpcb);
   fChain->SetBranchAddress("gmtDwrpcb", &gmtDwrpcb, &b_GMT_gmtDwrpcb);
   fChain->SetBranchAddress("gmtNrpcf", &gmtNrpcf, &b_GMT_gmtNrpcf);
   fChain->SetBranchAddress("gmtBxrpcf", &gmtBxrpcf, &b_GMT_gmtBxrpcf);
   fChain->SetBranchAddress("gmtPtrpcf", &gmtPtrpcf, &b_GMT_gmtPtrpcf);
   fChain->SetBranchAddress("gmtCharpcf", &gmtCharpcf, &b_GMT_gmtCharpcf);
   fChain->SetBranchAddress("gmtEtarpcf", &gmtEtarpcf, &b_GMT_gmtEtarpcf);
   fChain->SetBranchAddress("gmtPhirpcf", &gmtPhirpcf, &b_GMT_gmtPhirpcf);
   fChain->SetBranchAddress("gmtQualrpcf", &gmtQualrpcf, &b_GMT_gmtQualrpcf);
   fChain->SetBranchAddress("gmtDwrpcf", &gmtDwrpcf, &b_GMT_gmtDwrpcf);
   fChain->SetBranchAddress("gmtN", &gmtN, &b_GMT_gmtN);
   fChain->SetBranchAddress("gmtCandBx", &gmtCandBx, &b_GMT_gmtCandBx);
   fChain->SetBranchAddress("gmtPt", &gmtPt, &b_GMT_gmtPt);
   fChain->SetBranchAddress("gmtCha", &gmtCha, &b_GMT_gmtCha);
   fChain->SetBranchAddress("gmtEta", &gmtEta, &b_GMT_gmtEta);
   fChain->SetBranchAddress("gmtPhi", &gmtPhi, &b_GMT_gmtPhi);
   fChain->SetBranchAddress("gmtQual", &gmtQual, &b_GMT_gmtQual);
   fChain->SetBranchAddress("gmtDet", &gmtDet, &b_GMT_gmtDet);
   fChain->SetBranchAddress("gmtRank", &gmtRank, &b_GMT_gmtRank);
   fChain->SetBranchAddress("gmtIsol", &gmtIsol, &b_GMT_gmtIsol);
   fChain->SetBranchAddress("gmtMip", &gmtMip, &b_GMT_gmtMip);
   fChain->SetBranchAddress("gmtDw", &gmtDw, &b_GMT_gmtDw);
   fChain->SetBranchAddress("gmtIdxRPCb", &gmtIdxRPCb, &b_GMT_gmtIdxRPCb);
   fChain->SetBranchAddress("gmtIdxRPCf", &gmtIdxRPCf, &b_GMT_gmtIdxRPCf);
   fChain->SetBranchAddress("gmtIdxDTBX", &gmtIdxDTBX, &b_GMT_gmtIdxDTBX);
   fChain->SetBranchAddress("gmtIdxCSC", &gmtIdxCSC, &b_GMT_gmtIdxCSC);
   fChain->SetBranchAddress("gtNele", &gtNele, &b_GT_gtNele);
   fChain->SetBranchAddress("gtBxel", &gtBxel, &b_GT_gtBxel);
   fChain->SetBranchAddress("gtRankel", &gtRankel, &b_GT_gtRankel);
   fChain->SetBranchAddress("gtPhiel", &gtPhiel, &b_GT_gtPhiel);
   fChain->SetBranchAddress("gtEtael", &gtEtael, &b_GT_gtEtael);
   fChain->SetBranchAddress("gtNjet", &gtNjet, &b_GT_gtNjet);
   fChain->SetBranchAddress("gtBxjet", &gtBxjet, &b_GT_gtBxjet);
   fChain->SetBranchAddress("gtRankjet", &gtRankjet, &b_GT_gtRankjet);
   fChain->SetBranchAddress("gtPhijet", &gtPhijet, &b_GT_gtPhijet);
   fChain->SetBranchAddress("gtEtajet", &gtEtajet, &b_GT_gtEtajet);
   fChain->SetBranchAddress("maxRCTREG_", &maxRCTREG_, &b_RCT_maxRCTREG_);
   fChain->SetBranchAddress("rctRegSize", &rctRegSize, &b_RCT_rctRegSize);
   fChain->SetBranchAddress("rctRegEta", &rctRegEta, &b_RCT_rctRegEta);
   fChain->SetBranchAddress("rctRegPhi", &rctRegPhi, &b_RCT_rctRegPhi);
   fChain->SetBranchAddress("rctRegRnk", &rctRegRnk, &b_RCT_rctRegRnk);
   fChain->SetBranchAddress("rctRegVeto", &rctRegVeto, &b_RCT_rctRegVeto);
   fChain->SetBranchAddress("rctRegBx", &rctRegBx, &b_RCT_rctRegBx);
   fChain->SetBranchAddress("rctRegOverFlow", &rctRegOverFlow, &b_RCT_rctRegOverFlow);
   fChain->SetBranchAddress("rctRegMip", &rctRegMip, &b_RCT_rctRegMip);
   fChain->SetBranchAddress("rctRegFGrain", &rctRegFGrain, &b_RCT_rctRegFGrain);
   fChain->SetBranchAddress("rctEmSize", &rctEmSize, &b_RCT_rctEmSize);
   fChain->SetBranchAddress("rctIsIsoEm", &rctIsIsoEm, &b_RCT_rctIsIsoEm);
   fChain->SetBranchAddress("rctEmEta", &rctEmEta, &b_RCT_rctEmEta);
   fChain->SetBranchAddress("rctEmPhi", &rctEmPhi, &b_RCT_rctEmPhi);
   fChain->SetBranchAddress("rctEmRnk", &rctEmRnk, &b_RCT_rctEmRnk);
   fChain->SetBranchAddress("rctEmBx", &rctEmBx, &b_RCT_rctEmBx);
   fChain->SetBranchAddress("maxDTPH_", &maxDTPH_, &b_DTTF_maxDTPH_);
   fChain->SetBranchAddress("maxDTTH_", &maxDTTH_, &b_DTTF_maxDTTH_);
   fChain->SetBranchAddress("maxDTTR_", &maxDTTR_, &b_DTTF_maxDTTR_);
   fChain->SetBranchAddress("dttf_phSize", &dttf_phSize, &b_DTTF_dttf_phSize);
   fChain->SetBranchAddress("dttf_phBx", &dttf_phBx, &b_DTTF_dttf_phBx);
   fChain->SetBranchAddress("dttf_phWh", &dttf_phWh, &b_DTTF_dttf_phWh);
   fChain->SetBranchAddress("dttf_phSe", &dttf_phSe, &b_DTTF_dttf_phSe);
   fChain->SetBranchAddress("dttf_phSt", &dttf_phSt, &b_DTTF_dttf_phSt);
   fChain->SetBranchAddress("dttf_phAng", &dttf_phAng, &b_DTTF_dttf_phAng);
   fChain->SetBranchAddress("dttf_phBandAng", &dttf_phBandAng, &b_DTTF_dttf_phBandAng);
   fChain->SetBranchAddress("dttf_phCode", &dttf_phCode, &b_DTTF_dttf_phCode);
   fChain->SetBranchAddress("dttf_phX", &dttf_phX, &b_DTTF_dttf_phX);
   fChain->SetBranchAddress("dttf_phY", &dttf_phY, &b_DTTF_dttf_phY);
   fChain->SetBranchAddress("dttf_thSize", &dttf_thSize, &b_DTTF_dttf_thSize);
   fChain->SetBranchAddress("dttf_thBx", &dttf_thBx, &b_DTTF_dttf_thBx);
   fChain->SetBranchAddress("dttf_thWh", &dttf_thWh, &b_DTTF_dttf_thWh);
   fChain->SetBranchAddress("dttf_thSe", &dttf_thSe, &b_DTTF_dttf_thSe);
   fChain->SetBranchAddress("dttf_thSt", &dttf_thSt, &b_DTTF_dttf_thSt);
   fChain->SetBranchAddress("dttf_thX", &dttf_thX, &b_DTTF_dttf_thX);
   fChain->SetBranchAddress("dttf_thY", &dttf_thY, &b_DTTF_dttf_thY);
   fChain->SetBranchAddress("dttf_thTheta", &dttf_thTheta, &b_DTTF_dttf_thTheta);
   fChain->SetBranchAddress("dttf_thCode", &dttf_thCode, &b_DTTF_dttf_thCode);
   fChain->SetBranchAddress("dttf_trSize", &dttf_trSize, &b_DTTF_dttf_trSize);
   fChain->SetBranchAddress("dttf_trBx", &dttf_trBx, &b_DTTF_dttf_trBx);
   fChain->SetBranchAddress("dttf_trTag", &dttf_trTag, &b_DTTF_dttf_trTag);
   fChain->SetBranchAddress("dttf_trQual", &dttf_trQual, &b_DTTF_dttf_trQual);
   fChain->SetBranchAddress("dttf_trPtPck", &dttf_trPtPck, &b_DTTF_dttf_trPtPck);
   fChain->SetBranchAddress("dttf_trPtVal", &dttf_trPtVal, &b_DTTF_dttf_trPtVal);
   fChain->SetBranchAddress("dttf_trPhiPck", &dttf_trPhiPck, &b_DTTF_dttf_trPhiPck);
   fChain->SetBranchAddress("dttf_trPhiVal", &dttf_trPhiVal, &b_DTTF_dttf_trPhiVal);
   fChain->SetBranchAddress("dttf_trPhiGlob", &dttf_trPhiGlob, &b_DTTF_dttf_trPhiGlob);
   fChain->SetBranchAddress("dttf_trChPck", &dttf_trChPck, &b_DTTF_dttf_trChPck);
   fChain->SetBranchAddress("dttf_trWh", &dttf_trWh, &b_DTTF_dttf_trWh);
   fChain->SetBranchAddress("dttf_trSc", &dttf_trSc, &b_DTTF_dttf_trSc);
   fChain->SetBranchAddress("maxCSCTFTR_", &maxCSCTFTR_, &b_CSCTF_maxCSCTFTR_);
   fChain->SetBranchAddress("maxCSCTFLCTSTR_", &maxCSCTFLCTSTR_, &b_CSCTF_maxCSCTFLCTSTR_);
   fChain->SetBranchAddress("maxCSCTFLCTS_", &maxCSCTFLCTS_, &b_CSCTF_maxCSCTFLCTS_);
   fChain->SetBranchAddress("maxCSCTFSPS_", &maxCSCTFSPS_, &b_CSCTF_maxCSCTFSPS_);
   fChain->SetBranchAddress("csctf_trSize", &csctf_trSize, &b_CSCTF_csctf_trSize);
   fChain->SetBranchAddress("csctf_trEndcap", &csctf_trEndcap, &b_CSCTF_csctf_trEndcap);
   fChain->SetBranchAddress("csctf_trSector", &csctf_trSector, &b_CSCTF_csctf_trSector);
   fChain->SetBranchAddress("csctf_trBx", &csctf_trBx, &b_CSCTF_csctf_trBx);
   fChain->SetBranchAddress("csctf_trME1ID", &csctf_trME1ID, &b_CSCTF_csctf_trME1ID);
   fChain->SetBranchAddress("csctf_trME2ID", &csctf_trME2ID, &b_CSCTF_csctf_trME2ID);
   fChain->SetBranchAddress("csctf_trME3ID", &csctf_trME3ID, &b_CSCTF_csctf_trME3ID);
   fChain->SetBranchAddress("csctf_trME4ID", &csctf_trME4ID, &b_CSCTF_csctf_trME4ID);
   fChain->SetBranchAddress("csctf_trMB1ID", &csctf_trMB1ID, &b_CSCTF_csctf_trMB1ID);
   fChain->SetBranchAddress("csctf_trOutputLink", &csctf_trOutputLink, &b_CSCTF_csctf_trOutputLink);
   fChain->SetBranchAddress("csctf_trCharge", &csctf_trCharge, &b_CSCTF_csctf_trCharge);
   fChain->SetBranchAddress("csctf_trChargeValid", &csctf_trChargeValid, &b_CSCTF_csctf_trChargeValid);
   fChain->SetBranchAddress("csctf_trForR", &csctf_trForR, &b_CSCTF_csctf_trForR);
   fChain->SetBranchAddress("csctf_trPhi23", &csctf_trPhi23, &b_CSCTF_csctf_trPhi23);
   fChain->SetBranchAddress("csctf_trPhi12", &csctf_trPhi12, &b_CSCTF_csctf_trPhi12);
   fChain->SetBranchAddress("csctf_trPhiSign", &csctf_trPhiSign, &b_CSCTF_csctf_trPhiSign);
   fChain->SetBranchAddress("csctf_trEtaBit", &csctf_trEtaBit, &b_CSCTF_csctf_trEtaBit);
   fChain->SetBranchAddress("csctf_trPhiBit", &csctf_trPhiBit, &b_CSCTF_csctf_trPhiBit);
   fChain->SetBranchAddress("csctf_trPtBit", &csctf_trPtBit, &b_CSCTF_csctf_trPtBit);
   fChain->SetBranchAddress("csctf_trEta", &csctf_trEta, &b_CSCTF_csctf_trEta);
   fChain->SetBranchAddress("csctf_trPhi", &csctf_trPhi, &b_CSCTF_csctf_trPhi);
   fChain->SetBranchAddress("csctf_trPhi_02PI", &csctf_trPhi_02PI, &b_CSCTF_csctf_trPhi_02PI);
   fChain->SetBranchAddress("csctf_trPt", &csctf_trPt, &b_CSCTF_csctf_trPt);
   fChain->SetBranchAddress("csctf_trMode", &csctf_trMode, &b_CSCTF_csctf_trMode);
   fChain->SetBranchAddress("csctf_trQuality", &csctf_trQuality, &b_CSCTF_csctf_trQuality);
   fChain->SetBranchAddress("csctf_trNumLCTs", &csctf_trNumLCTs, &b_CSCTF_csctf_trNumLCTs);
   fChain->SetBranchAddress("csctf_trLctEndcap", &csctf_trLctEndcap, &b_CSCTF_csctf_trLctEndcap);
   fChain->SetBranchAddress("csctf_trLctSector", &csctf_trLctSector, &b_CSCTF_csctf_trLctSector);
   fChain->SetBranchAddress("csctf_trLctSubSector", &csctf_trLctSubSector, &b_CSCTF_csctf_trLctSubSector);
   fChain->SetBranchAddress("csctf_trLctBx", &csctf_trLctBx, &b_CSCTF_csctf_trLctBx);
   fChain->SetBranchAddress("csctf_trLctBx0", &csctf_trLctBx0, &b_CSCTF_csctf_trLctBx0);
   fChain->SetBranchAddress("csctf_trLctStation", &csctf_trLctStation, &b_CSCTF_csctf_trLctStation);
   fChain->SetBranchAddress("csctf_trLctRing", &csctf_trLctRing, &b_CSCTF_csctf_trLctRing);
   fChain->SetBranchAddress("csctf_trLctChamber", &csctf_trLctChamber, &b_CSCTF_csctf_trLctChamber);
   fChain->SetBranchAddress("csctf_trLctTriggerCSCID", &csctf_trLctTriggerCSCID, &b_CSCTF_csctf_trLctTriggerCSCID);
   fChain->SetBranchAddress("csctf_trLctFpga", &csctf_trLctFpga, &b_CSCTF_csctf_trLctFpga);
   fChain->SetBranchAddress("csctf_trLctlocalPhi", &csctf_trLctlocalPhi, &b_CSCTF_csctf_trLctlocalPhi);
   fChain->SetBranchAddress("csctf_trLctglobalPhi", &csctf_trLctglobalPhi, &b_CSCTF_csctf_trLctglobalPhi);
   fChain->SetBranchAddress("csctf_trLctglobalEta", &csctf_trLctglobalEta, &b_CSCTF_csctf_trLctglobalEta);
   fChain->SetBranchAddress("csctf_trLctstripNum", &csctf_trLctstripNum, &b_CSCTF_csctf_trLctstripNum);
   fChain->SetBranchAddress("csctf_trLctwireGroup", &csctf_trLctwireGroup, &b_CSCTF_csctf_trLctwireGroup);
   fChain->SetBranchAddress("csctf_lctSize", &csctf_lctSize, &b_CSCTF_csctf_lctSize);
   fChain->SetBranchAddress("csctf_lctEndcap", &csctf_lctEndcap, &b_CSCTF_csctf_lctEndcap);
   fChain->SetBranchAddress("csctf_lctSector", &csctf_lctSector, &b_CSCTF_csctf_lctSector);
   fChain->SetBranchAddress("csctf_lctSubSector", &csctf_lctSubSector, &b_CSCTF_csctf_lctSubSector);
   fChain->SetBranchAddress("csctf_lctBx", &csctf_lctBx, &b_CSCTF_csctf_lctBx);
   fChain->SetBranchAddress("csctf_lctBx0", &csctf_lctBx0, &b_CSCTF_csctf_lctBx0);
   fChain->SetBranchAddress("csctf_lctStation", &csctf_lctStation, &b_CSCTF_csctf_lctStation);
   fChain->SetBranchAddress("csctf_lctRing", &csctf_lctRing, &b_CSCTF_csctf_lctRing);
   fChain->SetBranchAddress("csctf_lctChamber", &csctf_lctChamber, &b_CSCTF_csctf_lctChamber);
   fChain->SetBranchAddress("csctf_lctTriggerCSCID", &csctf_lctTriggerCSCID, &b_CSCTF_csctf_lctTriggerCSCID);
   fChain->SetBranchAddress("csctf_lctFpga", &csctf_lctFpga, &b_CSCTF_csctf_lctFpga);
   fChain->SetBranchAddress("csctf_lctlocalPhi", &csctf_lctlocalPhi, &b_CSCTF_csctf_lctlocalPhi);
   fChain->SetBranchAddress("csctf_lctglobalPhi", &csctf_lctglobalPhi, &b_CSCTF_csctf_lctglobalPhi);
   fChain->SetBranchAddress("csctf_lctglobalEta", &csctf_lctglobalEta, &b_CSCTF_csctf_lctglobalEta);
   fChain->SetBranchAddress("csctf_lctstripNum", &csctf_lctstripNum, &b_CSCTF_csctf_lctstripNum);
   fChain->SetBranchAddress("csctf_lctwireGroup", &csctf_lctwireGroup, &b_CSCTF_csctf_lctwireGroup);
   fChain->SetBranchAddress("csctf_nsp", &csctf_nsp, &b_CSCTF_csctf_nsp);
   fChain->SetBranchAddress("csctf_stSPslot", &csctf_stSPslot, &b_CSCTF_csctf_stSPslot);
   fChain->SetBranchAddress("csctf_stL1A_BXN", &csctf_stL1A_BXN, &b_CSCTF_csctf_stL1A_BXN);
   fChain->SetBranchAddress("csctf_stTrkCounter", &csctf_stTrkCounter, &b_CSCTF_csctf_stTrkCounter);
   fChain->SetBranchAddress("csctf_stOrbCounter", &csctf_stOrbCounter, &b_CSCTF_csctf_stOrbCounter);
   std::cout<<" Done."<<std::endl;

   if (doreco){
     std::cout<<"Setting branch addresses for reco... ";
     fChain->SetBranchAddress("maxJet_", &maxJet_, &b_Jet_maxJet_);
     fChain->SetBranchAddress("nJets", &nJets, &b_Jet_nJets);
     fChain->SetBranchAddress("e", &e, &b_Jet_e);
     fChain->SetBranchAddress("et", &et, &b_Jet_et);
     fChain->SetBranchAddress("eta", &eta, &b_Jet_eta);
     fChain->SetBranchAddress("phi", &phi, &b_Jet_phi);
     fChain->SetBranchAddress("eEMF", &eEMF, &b_Jet_eEMF);
     fChain->SetBranchAddress("eHadHB", &eHadHB, &b_Jet_eHadHB);
     fChain->SetBranchAddress("eHadHE", &eHadHE, &b_Jet_eHadHE);
     fChain->SetBranchAddress("eHadHO", &eHadHO, &b_Jet_eHadHO);
     fChain->SetBranchAddress("eHadHF", &eHadHF, &b_Jet_eHadHF);
     fChain->SetBranchAddress("eEmEB", &eEmEB, &b_Jet_eEmEB);
     fChain->SetBranchAddress("eEmEE", &eEmEE, &b_Jet_eEmEE);
     fChain->SetBranchAddress("eEmHF", &eEmHF, &b_Jet_eEmHF);
     fChain->SetBranchAddress("eMaxEcalTow", &eMaxEcalTow, &b_Jet_eMaxEcalTow);
     fChain->SetBranchAddress("eMaxHcalTow", &eMaxHcalTow, &b_Jet_eMaxHcalTow);
     fChain->SetBranchAddress("towerArea", &towerArea, &b_Jet_towerArea);
     fChain->SetBranchAddress("towerSize", &towerSize, &b_Jet_towerSize);
     fChain->SetBranchAddress("n60", &n60, &b_Jet_n60);
     fChain->SetBranchAddress("n90", &n90, &b_Jet_n90);
     fChain->SetBranchAddress("met", &met, &b_Met_met);
     fChain->SetBranchAddress("metPhi", &metPhi, &b_Met_metPhi);
     fChain->SetBranchAddress("Ht", &Ht, &b_Met_Ht);
     fChain->SetBranchAddress("mHt", &mHt, &b_Met_mHt);
     fChain->SetBranchAddress("mHtPhi", &mHtPhi, &b_Met_mHtPhi);
     fChain->SetBranchAddress("sumEt", &sumEt, &b_Met_sumEt);
     fChain->SetBranchAddress("MAXCl_", &MAXCl_, &b_SuperClusters_MAXCl_);
     fChain->SetBranchAddress("nClusters", &nClusters, &b_SuperClusters_nClusters);
     fChain->SetBranchAddress("clusterEta", &clusterEta, &b_SuperClusters_clusterEta);
     fChain->SetBranchAddress("clusterPhi", &clusterPhi, &b_SuperClusters_clusterPhi);
     fChain->SetBranchAddress("clusterEt", &clusterEt, &b_SuperClusters_clusterEt);
     fChain->SetBranchAddress("clusterE", &clusterE, &b_SuperClusters_clusterE);
     fChain->SetBranchAddress("MAXCl_", &MAXCl_, &b_BasicClusters_MAXCl_);
     fChain->SetBranchAddress("nClusters", &nClusters, &b_BasicClusters_nClusters);
     fChain->SetBranchAddress("clusterEta", &clusterEta, &b_BasicClusters_clusterEta);
     fChain->SetBranchAddress("clusterPhi", &clusterPhi, &b_BasicClusters_clusterPhi);
     fChain->SetBranchAddress("clusterEt", &clusterEt, &b_BasicClusters_clusterEt);
     fChain->SetBranchAddress("clusterE", &clusterE, &b_BasicClusters_clusterE);
     std::cout<<" Done."<<std::endl;
   }
   
   if (domuonreco){
     std::cout<<"Setting branch addresses for muons... ";
     fChain->SetBranchAddress("nMuons", &nMuons, &b_Muon_nMuons);
     fChain->SetBranchAddress("muon_type", &muon_type, &b_Muon_muon_type);
     fChain->SetBranchAddress("muons_ch", &muons_ch, &b_Muon_muons_ch);
     fChain->SetBranchAddress("muons_pt", &muons_pt, &b_Muon_muons_pt);
     fChain->SetBranchAddress("muons_p", &muons_p, &b_Muon_muons_p);
     fChain->SetBranchAddress("muons_eta", &muons_eta, &b_Muon_muons_eta);
     fChain->SetBranchAddress("muons_phi", &muons_phi, &b_Muon_muons_phi);
     fChain->SetBranchAddress("muons_validhits", &muons_validhits, &b_Muon_muons_validhits);
     fChain->SetBranchAddress("muons_normchi2", &muons_normchi2, &b_Muon_muons_normchi2);
     fChain->SetBranchAddress("muons_imp_point_x", &muons_imp_point_x, &b_Muon_muons_imp_point_x);
     fChain->SetBranchAddress("muons_imp_point_y", &muons_imp_point_y, &b_Muon_muons_imp_point_y);
     fChain->SetBranchAddress("muons_imp_point_z", &muons_imp_point_z, &b_Muon_muons_imp_point_z);
     fChain->SetBranchAddress("muons_imp_point_p", &muons_imp_point_p, &b_Muon_muons_imp_point_p);
     fChain->SetBranchAddress("muons_imp_point_pt", &muons_imp_point_pt, &b_Muon_muons_imp_point_pt);
     fChain->SetBranchAddress("muons_phi_hb", &muons_phi_hb, &b_Muon_muons_phi_hb);
     fChain->SetBranchAddress("muons_z_hb", &muons_z_hb, &b_Muon_muons_z_hb);
     fChain->SetBranchAddress("muons_r_he_p", &muons_r_he_p, &b_Muon_muons_r_he_p);
     fChain->SetBranchAddress("muons_r_he_n", &muons_r_he_n, &b_Muon_muons_r_he_n);
     fChain->SetBranchAddress("muons_phi_he_p", &muons_phi_he_p, &b_Muon_muons_phi_he_p);
     fChain->SetBranchAddress("muons_phi_he_n", &muons_phi_he_n, &b_Muon_muons_phi_he_n);
     fChain->SetBranchAddress("muons_tr_ch", &muons_tr_ch, &b_Muon_muons_tr_ch);
     fChain->SetBranchAddress("muons_tr_pt", &muons_tr_pt, &b_Muon_muons_tr_pt);
     fChain->SetBranchAddress("muons_tr_p", &muons_tr_p, &b_Muon_muons_tr_p);
     fChain->SetBranchAddress("muons_tr_eta", &muons_tr_eta, &b_Muon_muons_tr_eta);
     fChain->SetBranchAddress("muons_tr_phi", &muons_tr_phi, &b_Muon_muons_tr_phi);
     fChain->SetBranchAddress("muons_tr_validhits", &muons_tr_validhits, &b_Muon_muons_tr_validhits);
     fChain->SetBranchAddress("muons_tr_normchi2", &muons_tr_normchi2, &b_Muon_muons_tr_normchi2);
     fChain->SetBranchAddress("muons_tr_imp_point_x", &muons_tr_imp_point_x, &b_Muon_muons_tr_imp_point_x);
     fChain->SetBranchAddress("muons_tr_imp_point_y", &muons_tr_imp_point_y, &b_Muon_muons_tr_imp_point_y);
     fChain->SetBranchAddress("muons_tr_imp_point_z", &muons_tr_imp_point_z, &b_Muon_muons_tr_imp_point_z);
     fChain->SetBranchAddress("muons_tr_imp_point_p", &muons_tr_imp_point_p, &b_Muon_muons_tr_imp_point_p);
     fChain->SetBranchAddress("muons_tr_imp_point_pt", &muons_tr_imp_point_pt, &b_Muon_muons_tr_imp_point_pt);
     fChain->SetBranchAddress("muons_sa_phi_mb2", &muons_sa_phi_mb2, &b_Muon_muons_sa_phi_mb2);
     fChain->SetBranchAddress("muons_sa_z_mb2", &muons_sa_z_mb2, &b_Muon_muons_sa_z_mb2);
     fChain->SetBranchAddress("muons_sa_pseta", &muons_sa_pseta, &b_Muon_muons_sa_pseta);
     fChain->SetBranchAddress("muons_sa_normchi2", &muons_sa_normchi2, &b_Muon_muons_sa_normchi2);
     fChain->SetBranchAddress("muons_sa_validhits", &muons_sa_validhits, &b_Muon_muons_sa_validhits);
     fChain->SetBranchAddress("muons_sa_ch", &muons_sa_ch, &b_Muon_muons_sa_ch);
     fChain->SetBranchAddress("muons_sa_pt", &muons_sa_pt, &b_Muon_muons_sa_pt);
     fChain->SetBranchAddress("muons_sa_p", &muons_sa_p, &b_Muon_muons_sa_p);
     fChain->SetBranchAddress("muons_sa_eta", &muons_sa_eta, &b_Muon_muons_sa_eta);
     fChain->SetBranchAddress("muons_sa_phi", &muons_sa_phi, &b_Muon_muons_sa_phi);
     fChain->SetBranchAddress("muons_sa_outer_pt", &muons_sa_outer_pt, &b_Muon_muons_sa_outer_pt);
     fChain->SetBranchAddress("muons_sa_inner_pt", &muons_sa_inner_pt, &b_Muon_muons_sa_inner_pt);
     fChain->SetBranchAddress("muons_sa_outer_eta", &muons_sa_outer_eta, &b_Muon_muons_sa_outer_eta);
     fChain->SetBranchAddress("muons_sa_inner_eta", &muons_sa_inner_eta, &b_Muon_muons_sa_inner_eta);
     fChain->SetBranchAddress("muons_sa_outer_phi", &muons_sa_outer_phi, &b_Muon_muons_sa_outer_phi);
     fChain->SetBranchAddress("muons_sa_inner_phi", &muons_sa_inner_phi, &b_Muon_muons_sa_inner_phi);
     fChain->SetBranchAddress("muons_sa_outer_x", &muons_sa_outer_x, &b_Muon_muons_sa_outer_x);
     fChain->SetBranchAddress("muons_sa_outer_y", &muons_sa_outer_y, &b_Muon_muons_sa_outer_y);
     fChain->SetBranchAddress("muons_sa_outer_z", &muons_sa_outer_z, &b_Muon_muons_sa_outer_z);
     fChain->SetBranchAddress("muons_sa_inner_x", &muons_sa_inner_x, &b_Muon_muons_sa_inner_x);
     fChain->SetBranchAddress("muons_sa_inner_y", &muons_sa_inner_y, &b_Muon_muons_sa_inner_y);
     fChain->SetBranchAddress("muons_sa_inner_z", &muons_sa_inner_z, &b_Muon_muons_sa_inner_z);
     fChain->SetBranchAddress("muons_sa_imp_point_x", &muons_sa_imp_point_x, &b_Muon_muons_sa_imp_point_x);
     fChain->SetBranchAddress("muons_sa_imp_point_y", &muons_sa_imp_point_y, &b_Muon_muons_sa_imp_point_y);
     fChain->SetBranchAddress("muons_sa_imp_point_z", &muons_sa_imp_point_z, &b_Muon_muons_sa_imp_point_z);
     fChain->SetBranchAddress("muons_sa_imp_point_p", &muons_sa_imp_point_p, &b_Muon_muons_sa_imp_point_p);
     fChain->SetBranchAddress("muons_sa_imp_point_pt", &muons_sa_imp_point_pt, &b_Muon_muons_sa_imp_point_pt);
     fChain->SetBranchAddress("muons_sa_phi_hb", &muons_sa_phi_hb, &b_Muon_muons_sa_phi_hb);
     fChain->SetBranchAddress("muons_sa_z_hb", &muons_sa_z_hb, &b_Muon_muons_sa_z_hb);
     fChain->SetBranchAddress("muons_sa_r_he_p", &muons_sa_r_he_p, &b_Muon_muons_sa_r_he_p);
     fChain->SetBranchAddress("muons_sa_r_he_n", &muons_sa_r_he_n, &b_Muon_muons_sa_r_he_n);
     fChain->SetBranchAddress("muons_sa_phi_he_p", &muons_sa_phi_he_p, &b_Muon_muons_sa_phi_he_p);
     fChain->SetBranchAddress("muons_sa_phi_he_n", &muons_sa_phi_he_n, &b_Muon_muons_sa_phi_he_n);
  
     std::cout<<" Done."<<std::endl;
   }
   Notify();
}

Bool_t L1MacroExample::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void L1MacroExample::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t L1MacroExample::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef l1macroexample_cxx
