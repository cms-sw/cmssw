#ifndef RecoExamples_myFastSimVal_h
#define RecoExamples_myFastSimVal_h
#include <TH1.h>
#include <TProfile.h>
/* \class myFastSimVal
 *
 * \author Frank Chlebana
 *
 * \version 1
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"

class TFile;

class myFastSimVal : public edm::EDAnalyzer {
public:
  myFastSimVal( const edm::ParameterSet & );

private:
  void beginJob( ) override;
  void analyze( const edm::Event& , const edm::EventSetup& ) override;
  void endJob() override;
  std::string CaloJetAlgorithm1, CaloJetAlgorithm2, CaloJetAlgorithm3, 
    CaloJetAlgorithm4;
  std::string GenJetAlgorithm1,  GenJetAlgorithm2,  GenJetAlgorithm3, 
    GenJetAlgorithm4;
  std::string JetCorrectionService;

  TProfile hf_TowerDelR1, hf_TowerDelR2, hf_TowerDelR3;
  TProfile hf_TowerDelR12, hf_TowerDelR22, hf_TowerDelR32;
  TProfile hf_nJet1, hf_nJet2, hf_nJet3, hf_nJet4;
  TProfile hf_nJet1s, hf_nJet2s, hf_nJet3s, hf_nJet4s;
  TProfile hf_nJet11, hf_nJet21, hf_nJet31, hf_nJet41;
  TProfile hf_PtResponse1, hf_PtResponse2,  hf_PtResponse3, hf_PtResponse4;
  
  TH1F hf_sumTowerAllEx, hf_sumTowerAllEy;
  TH1F SumEt1, MET1;
  TH1F SumEt12, MET12;
  TH1F SumEt13, MET13;
  TH1F hf_TowerJetEt1;

  TH1F nTowers1, nTowers2, nTowers3, nTowers4;
  TH1F nTowersLeadJetPt1, nTowersLeadJetPt2, nTowersLeadJetPt3, nTowersLeadJetPt4;
  TH1F totEneLeadJetEta1_1, totEneLeadJetEta2_1, totEneLeadJetEta3_1;
  TH1F hadEneLeadJetEta1_1, hadEneLeadJetEta2_1, hadEneLeadJetEta3_1;
  TH1F emEneLeadJetEta1_1,  emEneLeadJetEta2_1,  emEneLeadJetEta3_1;
  TH1F totEneLeadJetEta1_2, totEneLeadJetEta2_2, totEneLeadJetEta3_2;
  TH1F hadEneLeadJetEta1_2, hadEneLeadJetEta2_2, hadEneLeadJetEta3_2;
  TH1F emEneLeadJetEta1_2,  emEneLeadJetEta2_2,  emEneLeadJetEta3_2;

  TH1F matchedAllPt11, matchedAllPt12, matchedAllPt13;
  TH1F matchedAllPt21, matchedAllPt22, matchedAllPt23;
  TH1F matchedAllPt31, matchedAllPt32, matchedAllPt33;
  TH1F matchedAllPt41, matchedAllPt42, matchedAllPt43;
  TH1F matchedPt11, matchedPt12, matchedPt13;
  TH1F matchedPt21, matchedPt22, matchedPt23;
  TH1F matchedPt31, matchedPt32, matchedPt33;
  TH1F matchedPt41, matchedPt42, matchedPt43;

  TH1F hadEneLeadJet1, hadEneLeadJet2, hadEneLeadJet3;
  TH1F hadEneLeadJet12, hadEneLeadJet22, hadEneLeadJet32;
  TH1F hadEneLeadJet13, hadEneLeadJet23, hadEneLeadJet33;
  TH1F emEneLeadJet1,  emEneLeadJet2,  emEneLeadJet3;
  TH1F emEneLeadJet12,  emEneLeadJet22,  emEneLeadJet32;
  TH1F emEneLeadJet13,  emEneLeadJet23,  emEneLeadJet33;
  TH1F hadFracEta11, hadFracEta21, hadFracEta31;
  TH1F hadFracEta12, hadFracEta22, hadFracEta32;
  TH1F hadFracEta13, hadFracEta23, hadFracEta33;
  TH1F hadFracLeadJet1, nTowersLeadJet1, nTowersSecondJet1;
  TH1F hadFracLeadJet2, nTowersLeadJet2, nTowersSecondJet2;
  TH1F hadFracLeadJet3, nTowersLeadJet3, nTowersSecondJet3;
  TH1F TowerEtLeadJet1, TowerEtLeadJet2, TowerEtLeadJet3;
  TH1F TowerEtLeadJet12, TowerEtLeadJet22, TowerEtLeadJet32;
  TH1F TowerEtLeadJet13, TowerEtLeadJet23, TowerEtLeadJet33;
  TH1F ZpMass, ZpMassGen, ZpMassGen10, ZpMassGen13, ZpMassGen40;
  TH1F ZpMass_700_10, ZpMass_700_13, ZpMass_700_40;
  TH1F ZpMassGen_700_10, ZpMassGen_700_13, ZpMassGen_700_40;
  TH1F ZpMass_2000_10, ZpMass_2000_13, ZpMass_2000_40;
  TH1F ZpMassGen_2000_10, ZpMassGen_2000_13, ZpMassGen_2000_40;
  TH1F ZpMass_5000_10, ZpMass_5000_13, ZpMass_5000_40;
  TH1F ZpMassGen_5000_10, ZpMassGen_5000_13, ZpMassGen_5000_40;
  TH1F topMassParton, topMass1, topMass2, topMass3;

  TH1F tMass, tbarMass;
  TH1F tMassGen, tbarMassGen;
  TH1F ZpMassResL101,    ZpMassResL102,    ZpMassResL103;
  TH1F ZpMassResL131,    ZpMassResL132,    ZpMassResL133;
  TH1F ZpMassResL401,    ZpMassResL402,    ZpMassResL403;
  TH1F ZpMassResRL101,   ZpMassResRL102,   ZpMassResRL103;
  TH1F ZpMassResRL131,   ZpMassResRL132,   ZpMassResRL133;
  TH1F ZpMassResRL401,   ZpMassResRL402,   ZpMassResRL403;
  TH1F ZpMassResRLoP101,   ZpMassResRLoP102,   ZpMassResRLoP103;
  TH1F ZpMassResRLoP131,   ZpMassResRLoP132,   ZpMassResRLoP133;
  TH1F ZpMassResRLoP401,   ZpMassResRLoP402,   ZpMassResRLoP403;
  TH1F ZpMassResPRL101,  ZpMassResPRL102,  ZpMassResPRL103;
  TH1F ZpMassResPRL131,  ZpMassResPRL132,  ZpMassResPRL133;
  TH1F ZpMassResPRL401,  ZpMassResPRL402,  ZpMassResPRL403;
  TH1F ZpMassRes101,     ZpMassRes102,     ZpMassRes103;
  TH1F ZpMassRes131,     ZpMassRes132,     ZpMassRes133;
  TH1F ZpMassRes401,     ZpMassRes402,     ZpMassRes403;

  TH1F dijetMassCor_700_1,dijetMassCor_700_101,dijetMassCor_700_131,dijetMassCor_700_401;
  TH1F dijetMassCor_2000_1,dijetMassCor_2000_101,dijetMassCor_2000_131,dijetMassCor_2000_401;
  TH1F dijetMassCor_5000_1,dijetMassCor_5000_101,dijetMassCor_5000_131,dijetMassCor_5000_401;

  TH1F ZpMassMatched1, ZpMassMatched2, ZpMassMatched3;
  TH1F dijetMass1,     dijetMass2,     dijetMass3, dijetMass4;
  TH1F dijetMass12,     dijetMass22,     dijetMass32, dijetMass42;
  TH1F dijetMass13,     dijetMass23,     dijetMass33, dijetMass43;
  TH1F dijetMass101, dijetMass131, dijetMass401;
  TH1F dijetMass102, dijetMass132, dijetMass402;
  TH1F dijetMass103, dijetMass133, dijetMass403;
  TH1F dijetMass_700_101, dijetMass_700_131, dijetMass_700_401;
  TH1F dijetMass_2000_101, dijetMass_2000_131, dijetMass_2000_401;
  TH1F dijetMass_5000_101, dijetMass_5000_131, dijetMass_5000_401;
  TH1F dijetMassP1,    dijetMassP2,    dijetMassP3;
  TH1F dijetMassP101, dijetMassP131, dijetMassP401;

  TH1F dijetMassP_700_101, dijetMassP_700_131, dijetMassP_700_401;
  TH1F dijetMassP_2000_101, dijetMassP_2000_131, dijetMassP_2000_401;
  TH1F dijetMassP_5000_101, dijetMassP_5000_131, dijetMassP_5000_401;

  TH1F dijetMassCor1, dijetMassCor101, dijetMassCor131, dijetMassCor401;

  TH1F dRPar1, dPhiPar1, dEtaPar1, dPtPar1;
  TH1F dRPar2, dPhiPar2, dEtaPar2, dPtPar2;
  TH1F dRPar3, dPhiPar3, dEtaPar3, dPtPar3;
  TH1F dRPar4, dPhiPar4, dEtaPar4, dPtPar4;

  TH1F dRParton, dRPartonMin;
  TH1F dR1, dPhi1, dEta1, dPt1, dPtFrac1, dPt20Frac1, dPt40Frac1, dPt80Frac1, dPt100Frac1;
  TH1F dR2, dPhi2, dEta2, dPt2, dPtFrac2, dPt20Frac2, dPt40Frac2, dPt80Frac2, dPt100Frac2;
  TH1F dR3, dPhi3, dEta3, dPt3, dPtFrac3, dPt20Frac3, dPt40Frac3, dPt80Frac3, dPt100Frac3;
  TH1F dR4, dPhi4, dEta4, dPt4, dPtFrac4, dPt20Frac4, dPt40Frac4, dPt80Frac4, dPt100Frac4;
  TH1F dR12, dPhi12, dEta12, dPt12;
  TH1F h_nCalJets1, h_nCalJets2, h_nCalJets3, h_nCalJets4;
  TH1F h_nGenJets1, h_nGenJets2, h_nGenJets3, h_nGenJets4;

  TH1F h_ptCal1, h_etaCal1, h_phiCal1;
  TH1F h_ptCal2, h_etaCal2, h_phiCal2;
  TH1F h_ptCal3, h_etaCal3, h_phiCal3;
  TH1F h_ptCal4, h_etaCal4, h_phiCal4;
  TH1F h_ptCal12, h_ptCal22, h_ptCal32, h_ptCal42;
  TH1F h_ptCal13, h_ptCal23, h_ptCal33, h_ptCal43;

  TH1F h_ptCalL1, h_etaCalL1, h_phiCalL1;
  TH1F h_ptCalL2, h_etaCalL2, h_phiCalL2; 
  TH1F h_ptCalL3, h_etaCalL3, h_phiCalL3; 
  TH1F h_ptCalL4, h_etaCalL4, h_phiCalL4; 
  TH1F h_ptCalL12, h_ptCalL22, h_ptCalL32, h_ptCalL42;
  TH1F h_ptCalL13, h_ptCalL23, h_ptCalL33, h_ptCalL43;

  TH1F h_ptGen1, h_etaGen1, h_phiGen1;
  TH1F h_ptGen2, h_etaGen2, h_phiGen2;
  TH1F h_ptGen3, h_etaGen3, h_phiGen3;
  TH1F h_ptGen4, h_etaGen4, h_phiGen4;

  TH1F h_ptGen12, h_ptGen22, h_ptGen32, h_ptGen42; 
  TH1F h_ptGen13, h_ptGen23, h_ptGen33, h_ptGen43; 

  TH1F h_ptGenL1, h_etaGenL1, h_phiGenL1;
  TH1F h_ptGenL2, h_etaGenL2, h_phiGenL2;
  TH1F h_ptGenL3, h_etaGenL3, h_phiGenL3;
  TH1F h_ptGenL12, h_ptGenL22, h_ptGenL32;
  TH1F h_ptGenL13, h_ptGenL23, h_ptGenL33;

  TH1F h_jetEt1, h_jetEt2,  h_jetEt3;
  TH1F h_missEt1s,h_missEt2s, h_missEt3s;
  TH1F h_missEt1,h_missEt2, h_missEt3;
  TH1F h_totMissEt1,h_totMissEt2, h_totMissEt3;

  TH1F h_lowPtCal11, h_lowPtCal21, h_lowPtCal31, h_lowPtCal41;
  TH1F h_lowPtCal12, h_lowPtCal22, h_lowPtCal32, h_lowPtCal42;
  TH1F h_lowPtCal13, h_lowPtCal23, h_lowPtCal33, h_lowPtCal43;

  TH1F h_lowPtCal1c11, h_lowPtCal2c11, h_lowPtCal3c11, h_lowPtCal4c11;
  TH1F h_lowPtCal1c12, h_lowPtCal2c12, h_lowPtCal3c12, h_lowPtCal4c12;
  TH1F h_lowPtCal1c13, h_lowPtCal2c13, h_lowPtCal3c13, h_lowPtCal4c13;

  TH1F h_jet1Pt1, h_jet2Pt1, h_jet3Pt1, h_jet4Pt1, h_jet5Pt1,
    h_jet6Pt1, h_jet7Pt1;
  TH1F h_jet1Pt2, h_jet2Pt2, h_jet3Pt2, h_jet4Pt2, h_jet5Pt2,
    h_jet6Pt2, h_jet7Pt2;
  TH1F h_jet1Pt3, h_jet2Pt3, h_jet3Pt3, h_jet4Pt3, h_jet5Pt3,
    h_jet6Pt3, h_jet7Pt3;
  TH1F h_jet1Pt4, h_jet2Pt4, h_jet3Pt4, h_jet4Pt4, h_jet5Pt4,
    h_jet6Pt4, h_jet7Pt4;

  TH1F ParMatch1, ParMatch2, ParMatch3;

  
  TFile* m_file;
};

#endif
