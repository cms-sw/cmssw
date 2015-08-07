#ifndef ANALYSIS_SELECTIONS_H
#define ANALYSIS_SELECTIONS_H

#include "DQM/PhysicsHWW/interface/EGammaMvaEleEstimator.h"
#include "DQM/PhysicsHWW/interface/monitor.h"
#include "DQM/PhysicsHWW/interface/ElectronIDMVA.h"
#include "DQM/PhysicsHWW/interface/MuonIDMVA.h"
#include "DQM/PhysicsHWW/interface/MuonMVAEstimator.h"
#include "DQM/PhysicsHWW/interface/wwtypes.h"
#include "DQM/PhysicsHWW/interface/analysisEnums.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

namespace HWWFunctions {

  void doCutFlow(HWW&, int, EventMonitor&, EGammaMvaEleEstimator*, MuonMVAEstimator*);
  bool passFirstCuts(HWW&, int);
  bool passCharge(HWW&, int);
  bool passBaseline(HWW&, int, EGammaMvaEleEstimator*, MuonMVAEstimator*);
  bool passFullLep(HWW&, int, EGammaMvaEleEstimator*, MuonMVAEstimator*);
  bool passExtraLeptonVeto(HWW&, int, EGammaMvaEleEstimator*, MuonMVAEstimator*);
  bool passZVeto(HWW&, int);
  bool passMinMet(HWW&, int);
  bool passMinMet40(HWW&, int);
  bool passDPhiDiLepJet(HWW&, int);
  bool passSoftMuonVeto(HWW&, int);
  bool passTopVeto(HWW&, int);

  int  bestHypothesis(HWW& hww, const std::vector<int>&);
  bool isGoodVertex(HWW& hww, int);
  unsigned int nGoodVertex(HWW& hww);


  std::vector<JetPair> getJets(HWW& hww, int, double, double, bool, bool);
  std::vector<JetPair> getDefaultJets(HWW& hww, unsigned int, bool);
  unsigned int numberOfJets(HWW& hww, unsigned int);
  bool defaultBTag(HWW& hww, unsigned int, float);
  Bool_t comparePt(JetPair lv1, JetPair lv2);
  bool passMVAJetId(double, double, double, unsigned int);

  //
  // Electron Id
  //

  bool   ww_elBase(HWW& hww, unsigned int i);
  bool   ww_elId(HWW& hww, unsigned int i, bool useLHeleId, int useMVAeleId, EGammaMvaEleEstimator* egammaMvaEleEstimator);
  bool   ww_eld0(HWW& hww, unsigned int i);
  bool   ww_eld0PV(HWW& hww, unsigned int i);
  bool   ww_eldZPV(HWW& hww, unsigned int i);
  bool   ww_elIso(HWW& hww, unsigned int i);
  double ww_elIsoVal(HWW& hww, unsigned int i);

  // combined analysis selectors
  bool goodElectronTMVA(HWW& hww, EGammaMvaEleEstimator* egammaMvaEleEstimator, int useMVAeleId, unsigned int i); 
  bool goodElectronWithoutIsolation(HWW& hww, unsigned int i, bool useLHeleId, int useMVAeleId, EGammaMvaEleEstimator* egammaMvaEleEstimator);
  bool goodElectronIsolated(HWW& hww, unsigned int i, bool useLHeleId, int useMVAeleId, EGammaMvaEleEstimator* egammaMvaEleEstimator, bool lockToCoreSelectors);
  bool fakableElectron(HWW& hww, unsigned int i,EleFOTypes);
  bool ElectronFOV4(HWW& hww, unsigned int i);

  //
  // Muon Id
  //

  bool   ww_muBase(HWW& hww, unsigned int i);
  bool   ww_muId(HWW& hww, unsigned int i, bool useMVAmuId, MuonIDMVA *mva);
  bool   ww_muIso(HWW& hww, unsigned int i);
  bool   ww_muIso(HWW& hww, unsigned int i, MuonMVAEstimator* muonMVAEstimator, std::vector<Int_t> IdentifiedMu, std::vector<Int_t> IdentifiedEle);
  double ww_muIsoVal(HWW& hww, unsigned int i);
  bool   ww_mud0(HWW& hww, unsigned int i);
  bool   ww_mud0PV(HWW& hww, unsigned int i);
  bool   ww_mudZPV(HWW& hww, unsigned int i, float cut=0.1);

  // combined analysis selectors
  bool goodMuonTMVA(HWW& hww, MuonIDMVA *mva, unsigned int i);
  bool goodMuonWithoutIsolation(HWW& hww, unsigned int i, bool useMVAmuId, MuonIDMVA *mva,
                  MuonMVAEstimator* muonMVAEstimator, std::vector<Int_t> IdentifiedMu, std::vector<Int_t> IdentifiedEle);
  bool goodMuonIsolated(HWW& hww,  unsigned int i, bool lockToCoreSelectors, bool useMVAmuId, MuonIDMVA *mva, 
               MuonMVAEstimator* muonMVAEstimator, std::vector<Int_t> IdentifiedMu, std::vector<Int_t> IdentifiedEle );
  bool fakableMuon(HWW& hww, unsigned int i, MuFOTypes, MuonMVAEstimator* muonMVAEstimator,
                  std::vector<Int_t> IdentifiedMu, std::vector<Int_t> IdentifiedEle);
  bool passMuonRingsMVA(HWW& hww, unsigned int mu, MuonMVAEstimator* muonMVAEstimator, std::vector<Int_t> IdentifiedMu, std::vector<Int_t> IdentifiedEle);
  bool passMuonRingsMVAFO(HWW& hww, unsigned int mu, MuonMVAEstimator* muonMVAEstimator, std::vector<Int_t> IdentifiedMu, std::vector<Int_t> IdentifiedEle);


  //
  // leptons
  //
  std::vector<LeptonPair> getExtraLeptons(HWW& hww, int i_hyp, double minPt,  bool useLHeleId, int useMVAeleId, 
                      EGammaMvaEleEstimator* egammaMvaEleEstimator, bool useMVAmuId, MuonIDMVA *mumva,
                      MuonMVAEstimator* muonMVAEstimator, std::vector<Int_t> IdentifiedMu, std::vector<Int_t> IdentifiedEle);
  unsigned int numberOfExtraLeptons(HWW& hww, int i_hyp, double minPt, bool useLHeleId, int useMVAeleId, 
                    EGammaMvaEleEstimator* egammaMvaEleEstimator, bool useMVAmuId, MuonIDMVA *mumva,
                    MuonMVAEstimator* muonMVAEstimator, std::vector<Int_t> IdentifiedMu, std::vector<Int_t> IdentifiedEle);
  unsigned int numberOfSoftMuons(HWW& hww, int i_hyp, bool nonisolated);

  double nearestDeltaPhi(HWW& hww, double Phi, int i_hyp);
  double projectedMet(HWW& hww, unsigned int i_hyp, double met, double phi);


  bool toptag(HWW& hww, int i_hyp, double minPt, std::vector<JetPair> ignoreJets=std::vector<JetPair>());

}
#endif
