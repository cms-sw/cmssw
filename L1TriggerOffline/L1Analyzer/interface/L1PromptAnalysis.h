
//-------------------------------------------------
//
//   \class L1PromptAnalysis
/**
 *   Description:  This code is designed for l1 prompt analysis
//                 starting point is a GMTTreeMaker By Ivan Mikulec. 
*/
//
//   I. Mikulec            HEPHY Vienna
//
//--------------------------------------------------
#ifndef L1_PROMPT_ANALYSIS_H
#define L1_PROMPT_ANALYSIS_H

//---------------
// C++ Headers --
//---------------

#include <memory>
#include <string>

//----------------------
// Base Class Headers --
//----------------------
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "FWCore/Utilities/interface/InputTag.h"

class TFile;
class TTree;

//              ---------------------
//              -- Class Interface --
//              ---------------------

const int MAXGEN = 20;
const int MAXRPC = 12;
const int MAXDTBX = 12;
const int MAXCSC = 12;
const int MAXGMT = 12;
const int MAXGT = 12;
const int MAXRCTREG = 400;
const int MAXDTPH = 50;
const int MAXDTTH = 50;
const int MAXDTTR = 50;

class L1PromptAnalysis : public edm::EDAnalyzer {
public:
  // constructor
  explicit L1PromptAnalysis(const edm::ParameterSet&);
  virtual ~L1PromptAnalysis();

  // fill tree
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void book();

  virtual void beginJob();
  virtual void endJob();

private:
  //GENERAL block
  int runn;
  int eventn;
  int lumi;
  int bx;
  uint64_t orbitn;
  uint64_t timest;

  // Generator info
  float weight;
  float pthat;

  // simulation block
  int ngen;
  float pxgen[MAXGEN];
  float pygen[MAXGEN];
  float pzgen[MAXGEN];
  float ptgen[MAXGEN];
  float etagen[MAXGEN];
  float phigen[MAXGEN];
  int chagen[MAXGEN];
  float vxgen[MAXGEN];
  float vygen[MAXGEN];
  float vzgen[MAXGEN];
  int pargen[MAXGEN];

  // GMT data
  int bxgmt;

  //DTBX Trigger block
  int ndt;
  int bxd[MAXDTBX];
  float ptd[MAXDTBX];
  int chad[MAXDTBX];
  float etad[MAXDTBX];
  int etafined[MAXDTBX];
  float phid[MAXDTBX];
  int quald[MAXDTBX];
  int dwd[MAXDTBX];
  int chd[MAXDTBX];

  //CSC Trigger block
  int ncsc;
  int bxc[MAXCSC];
  float ptc[MAXCSC];
  int chac[MAXCSC];
  float etac[MAXCSC];
  float phic[MAXCSC];
  int qualc[MAXCSC];
  int dwc[MAXCSC];

  //RPCb Trigger
  int nrpcb;
  int bxrb[MAXRPC];
  float ptrb[MAXRPC];
  int charb[MAXRPC];
  float etarb[MAXRPC];
  float phirb[MAXRPC];
  int qualrb[MAXRPC];
  int dwrb[MAXRPC];

  //RPCf Trigger
  int nrpcf;
  int bxrf[MAXRPC];
  float ptrf[MAXRPC];
  int charf[MAXRPC];
  float etarf[MAXRPC];
  float phirf[MAXRPC];
  int qualrf[MAXRPC];
  int dwrf[MAXRPC];

  //Global Muon Trigger
  int ngmt;
  int bxg[MAXGMT];
  float ptg[MAXGMT];
  int chag[MAXGMT];
  float etag[MAXGMT];
  float phig[MAXGMT];
  int qualg[MAXGMT];
  int detg[MAXGMT];
  int rankg[MAXGMT];
  int isolg[MAXGMT];
  int mipg[MAXGMT];
  int dwg[MAXGMT];
  int idxRPCb[MAXGMT];
  int idxRPCf[MAXGMT];
  int idxDTBX[MAXGMT];
  int idxCSC[MAXGMT];

  // GT info
  uint64_t gttw1[3];
  uint64_t gttw2[3];
  uint64_t gttt[3];

  //PSB info
  int nele;
  int bxel[MAXGT];
  float rankel[MAXGT];
  float phiel[MAXGT];
  float etael[MAXGT];

  int njet;
  int bxjet[MAXGT];
  float rankjet[MAXGT];
  float phijet[MAXGT];
  float etajet[MAXGT];

  //GCT
  edm::InputTag gctCenJetsSource_;
  edm::InputTag gctForJetsSource_;
  edm::InputTag gctTauJetsSource_;
  edm::InputTag gctEnergySumsSource_;
  edm::InputTag gctIsoEmSource_;
  edm::InputTag gctNonIsoEmSource_;

  int gctIsoEmSize;
  float gctIsoEmEta[4];
  float gctIsoEmPhi[4];
  float gctIsoEmRnk[4];
  int gctNonIsoEmSize;
  float gctNonIsoEmEta[4];
  float gctNonIsoEmPhi[4];
  float gctNonIsoEmRnk[4];
  int gctCJetSize;
  float gctCJetEta[4];
  float gctCJetPhi[4];
  float gctCJetRnk[4];
  int gctFJetSize;
  float gctFJetEta[4];
  float gctFJetPhi[4];
  float gctFJetRnk[4];
  int gctTJetSize;
  float gctTJetEta[4];
  float gctTJetPhi[4];
  float gctTJetRnk[4];
  float gctEtMiss;
  float gctEtMissPhi;
  float gctEtHad;
  float gctEtTot;
  int gctHFRingEtSumSize;
  float gctHFRingEtSumEta[4];
  float gctHFBitCountsSize;
  float gctHFBitCountsEta[4];
  //  RCT

  edm::InputTag rctSource_;
  int rctRegSize;
  float rctRegEta[MAXRCTREG];
  float rctRegPhi[MAXRCTREG];
  float rctRegRnk[MAXRCTREG];
  int rctRegVeto[MAXRCTREG];
  int rctRegBx[MAXRCTREG];
  int rctRegOverFlow[MAXRCTREG];
  int rctRegMip[MAXRCTREG];
  int rctRegFGrain[MAXRCTREG];
  int rctEmSize;
  int rctIsIsoEm[MAXRCTREG];
  float rctEmEta[MAXRCTREG];
  float rctEmPhi[MAXRCTREG];
  float rctEmRnk[MAXRCTREG];
  int rctEmBx[MAXRCTREG];

  // DTTF
  edm::InputTag dttfSource_;

  int dttf_phSize;
  int dttf_phBx[MAXDTPH];
  int dttf_phWh[MAXDTPH];
  int dttf_phSe[MAXDTPH];
  int dttf_phSt[MAXDTPH];
  float dttf_phAng[MAXDTPH];
  float dttf_phBandAng[MAXDTPH];
  int dttf_phCode[MAXDTPH];
  float dttf_phX[MAXDTPH];
  float dttf_phY[MAXDTPH];

  int dttf_thSize;
  int dttf_thBx[MAXDTTH];
  int dttf_thWh[MAXDTTH];
  int dttf_thSe[MAXDTTH];
  int dttf_thSt[MAXDTTH];
  float dttf_thX[MAXDTTH];
  float dttf_thY[MAXDTTH];
  float dttf_thTheta[MAXDTTH][7];
  int dttf_thCode[MAXDTTH][7];

  int dttf_trSize;
  int dttf_trBx[MAXDTTR];
  int dttf_trTag[MAXDTTR];
  int dttf_trQual[MAXDTTR];
  int dttf_trPtPck[MAXDTTR];
  float dttf_trPtVal[MAXDTTR];
  int dttf_trPhiPck[MAXDTTR];
  float dttf_trPhiVal[MAXDTTR];
  int dttf_trPhiGlob[MAXDTTR];
  int dttf_trChPck[MAXDTTR];
  int dttf_trWh[MAXDTTR];
  int dttf_trSc[MAXDTTR];
  ///

  TFile* m_file;
  TTree* m_tree;

  edm::InputTag m_GMTInputTag;
  edm::InputTag m_GTEvmInputTag;
  edm::InputTag m_GTInputTag;
  edm::InputTag m_GeneratorInputTag;
  edm::InputTag m_SimulationInputTag;

  bool m_PhysVal;
  bool verbose_;
  std::string m_outfilename;
};

#endif
