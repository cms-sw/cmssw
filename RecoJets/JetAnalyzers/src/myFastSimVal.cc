// myFastSimVal.cc
// Description:  Comparison between Jet Algorithms
// Author: Frank Chlebana
// Date:  08 - August - 2007
//
#include "RecoJets/JetAnalyzers/interface/myFastSimVal.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
// #include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
// #include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TCanvas.h>
#include <cmath>
using namespace edm;
using namespace reco;
using namespace std;

#define DEBUG 1
// #define MAXJETS 50
#define MAXJETS 100

// Get the algorithm of the jet collections we will read from the .cfg file
// which defines the value of the strings CaloJetAlgorithm and GenJetAlgorithm.
myFastSimVal::myFastSimVal(const ParameterSet& cfg)
    : CaloJetAlgorithm1(cfg.getParameter<string>("CaloJetAlgorithm1")),
      CaloJetAlgorithm2(cfg.getParameter<string>("CaloJetAlgorithm2")),
      CaloJetAlgorithm3(cfg.getParameter<string>("CaloJetAlgorithm3")),
      CaloJetAlgorithm4(cfg.getParameter<string>("CaloJetAlgorithm4")),
      GenJetAlgorithm1(cfg.getParameter<string>("GenJetAlgorithm1")),
      GenJetAlgorithm2(cfg.getParameter<string>("GenJetAlgorithm2")),
      GenJetAlgorithm3(cfg.getParameter<string>("GenJetAlgorithm3")),
      GenJetAlgorithm4(cfg.getParameter<string>("GenJetAlgorithm4")),
      JetCorrectionService(cfg.getParameter<string>("JetCorrectionService")) {}

int nEvent = 0;

void myFastSimVal::beginJob() {
  // Open the histogram file and book some associated histograms
  m_file = new TFile("histo.root", "RECREATE");

  tMassGen = TH1F("tMassGen", "T Mass Gen", 100, 0, 200);
  tbarMassGen = TH1F("tbarMassGen", "Tbar Mass Gen", 100, 0, 200);

  tMass = TH1F("tMass", "T Mass", 100, 0, 200);
  tbarMass = TH1F("tbarMass", "Tbar Mass", 100, 0, 200);

  topMassParton = TH1F("topMassParton", "Top Mass Parton", 100, 0, 200);
  topMass1 = TH1F("topMass1", "Top Mass 1", 100, 0, 200);
  topMass2 = TH1F("topMass2", "Top Mass 2", 100, 0, 200);
  topMass3 = TH1F("topMass3", "Top Mass 3", 100, 0, 200);

  ZpMass = TH1F("ZpMass", "Generated Zp Mass", 160, 0, 8000);
  ZpMassGen = TH1F("ZpMassGen", "Gen Zp Mass", 160, 0, 8000);
  ZpMassMatched1 = TH1F("ZpMassMatched1", "Calor Zp Mass 1", 160, 0, 8000);
  ZpMassMatched2 = TH1F("ZpMassMatched2", "Calor Zp Mass 2", 160, 0, 8000);
  ZpMassMatched3 = TH1F("ZpMassMatched3", "Calor Zp Mass 3", 160, 0, 8000);

  ZpMassGen10 = TH1F("ZpMassGen10", "Gen Zp Mass", 160, 0, 8000);
  ZpMassGen13 = TH1F("ZpMassGen13", "Gen Zp Mass", 160, 0, 8000);
  ZpMassGen40 = TH1F("ZpMassGen40", "Gen Zp Mass", 160, 0, 8000);

  ZpMass_700_10 = TH1F("ZpMass_700_10", "Parton Zp Mass", 100, 0, 1000);
  ZpMass_700_13 = TH1F("ZpMass_700_13", "Parton Zp Mass", 100, 0, 1000);
  ZpMass_700_40 = TH1F("ZpMass_700_40", "Parton Zp Mass", 100, 0, 1000);

  ZpMassGen_700_10 = TH1F("ZpMassGen_700_10", "Gen Zp Mass", 100, 0, 1000);
  ZpMassGen_700_13 = TH1F("ZpMassGen_700_13", "Gen Zp Mass", 100, 0, 1000);
  ZpMassGen_700_40 = TH1F("ZpMassGen_700_40", "Gen Zp Mass", 100, 0, 1000);

  ZpMassGen_2000_10 = TH1F("ZpMassGen_2000_10", "Gen Zp Mass", 100, 1500, 2500);
  ZpMassGen_2000_13 = TH1F("ZpMassGen_2000_13", "Gen Zp Mass", 100, 1500, 2500);
  ZpMassGen_2000_40 = TH1F("ZpMassGen_2000_40", "Gen Zp Mass", 100, 1500, 2500);

  ZpMass_2000_10 = TH1F("ZpMass_2000_10", "Parton Zp Mass", 100, 1500, 2500);
  ZpMass_2000_13 = TH1F("ZpMass_2000_13", "Parton Zp Mass", 100, 1500, 2500);
  ZpMass_2000_40 = TH1F("ZpMass_2000_40", "Parton Zp Mass", 100, 1500, 2500);

  ZpMassGen_5000_10 = TH1F("ZpMassGen_5000_10", "Gen Zp Mass", 150, 4000, 5500);
  ZpMassGen_5000_13 = TH1F("ZpMassGen_5000_13", "Gen Zp Mass", 150, 4000, 5500);
  ZpMassGen_5000_40 = TH1F("ZpMassGen_5000_40", "Gen Zp Mass", 150, 4000, 5500);

  ZpMass_5000_10 = TH1F("ZpMass_5000_10", "Parton Zp Mass", 150, 4000, 5500);
  ZpMass_5000_13 = TH1F("ZpMass_5000_13", "Parton Zp Mass", 150, 4000, 5500);
  ZpMass_5000_40 = TH1F("ZpMass_5000_40", "Parton Zp Mass", 150, 4000, 5500);

  ZpMassRes101 = TH1F("ZpMassRes101", "Zp Mass Resolution 1", 100, -2, 2);
  ZpMassRes102 = TH1F("ZpMassRes102", "Zp Mass Resolution 2", 100, -2, 2);
  ZpMassRes103 = TH1F("ZpMassRes103", "Zp Mass Resolution 3", 100, -2, 2);

  ZpMassRes131 = TH1F("ZpMassRes131", "Zp Mass Resolution 1", 100, -2, 2);
  ZpMassRes132 = TH1F("ZpMassRes132", "Zp Mass Resolution 2", 100, -2, 2);
  ZpMassRes133 = TH1F("ZpMassRes133", "Zp Mass Resolution 3", 100, -2, 2);

  ZpMassRes401 = TH1F("ZpMassRes401", "Zp Mass Resolution 1", 100, -2, 2);
  ZpMassRes402 = TH1F("ZpMassRes402", "Zp Mass Resolution 2", 100, -2, 2);
  ZpMassRes403 = TH1F("ZpMassRes403", "Zp Mass Resolution 3", 100, -2, 2);

  ZpMassResL101 = TH1F("ZpMassResL101", "Zp Mass Resolution Leading Jets 1", 100, 0, 2);
  ZpMassResL102 = TH1F("ZpMassResL102", "Zp Mass Resolution Leading Jets 2", 100, 0, 2);
  ZpMassResL103 = TH1F("ZpMassResL103", "Zp Mass Resolution Leading Jets 3", 100, 0, 2);

  ZpMassResRL101 = TH1F("ZpMassResRL101", "Zp Mass Res. Ratio Leading Jets 1", 100, 0, 2);
  ZpMassResRL102 = TH1F("ZpMassResRL102", "Zp Mass Res. Ratio Leading Jets 2", 100, 0, 2);
  ZpMassResRL103 = TH1F("ZpMassResRL103", "Zp Mass Res. Ratio Leading Jets 3", 100, 0, 2);

  ZpMassResRLoP101 = TH1F("ZpMassResRLoP101", "Zp Mass RLoP Ratio Leading Jets 1", 100, 0, 2);
  ZpMassResRLoP102 = TH1F("ZpMassResRLoP102", "Zp Mass RLoP Ratio Leading Jets 2", 100, 0, 2);
  ZpMassResRLoP103 = TH1F("ZpMassResRLoP103", "Zp Mass RLoP Ratio Leading Jets 3", 100, 0, 2);

  ZpMassResPRL101 = TH1F("ZpMassResPRL101", "Zp Mass Res. P Ratio Leading Jets 1", 100, 0, 2);
  ZpMassResPRL102 = TH1F("ZpMassResPRL102", "Zp Mass Res. P Ratio Leading Jets 2", 100, 0, 2);
  ZpMassResPRL103 = TH1F("ZpMassResPRL103", "Zp Mass Res. P Ratio Leading Jets 3", 100, 0, 2);

  ZpMassResL131 = TH1F("ZpMassResL131", "Zp Mass Resolution Leading Jets 1", 100, 0, 2);
  ZpMassResL132 = TH1F("ZpMassResL132", "Zp Mass Resolution Leading Jets 2", 100, 0, 2);
  ZpMassResL133 = TH1F("ZpMassResL133", "Zp Mass Resolution Leading Jets 3", 100, 0, 2);

  ZpMassResRL131 = TH1F("ZpMassResRL131", "Zp Mass Res. Ratio Leading Jets 1", 100, 0, 2);
  ZpMassResRL132 = TH1F("ZpMassResRL132", "Zp Mass Res. Ratio Leading Jets 2", 100, 0, 2);
  ZpMassResRL133 = TH1F("ZpMassResRL133", "Zp Mass Res. Ratio Leading Jets 3", 100, 0, 2);

  ZpMassResRLoP131 = TH1F("ZpMassResRLoP131", "Zp Mass RLoP Ratio Leading Jets 1", 100, 0, 2);
  ZpMassResRLoP132 = TH1F("ZpMassResRLoP132", "Zp Mass RLoP Ratio Leading Jets 2", 100, 0, 2);
  ZpMassResRLoP133 = TH1F("ZpMassResRLoP133", "Zp Mass RLoP Ratio Leading Jets 3", 100, 0, 2);

  ZpMassResPRL131 = TH1F("ZpMassResPRL131", "Zp Mass Res. P Ratio Leading Jets 1", 100, 0, 2);
  ZpMassResPRL132 = TH1F("ZpMassResPRL132", "Zp Mass Res. P Ratio Leading Jets 2", 100, 0, 2);
  ZpMassResPRL133 = TH1F("ZpMassResPRL133", "Zp Mass Res. P Ratio Leading Jets 3", 100, 0, 2);

  ZpMassResL401 = TH1F("ZpMassResL401", "Zp Mass Resolution Leading Jets 1", 100, 0, 2);
  ZpMassResL402 = TH1F("ZpMassResL402", "Zp Mass Resolution Leading Jets 2", 100, 0, 2);
  ZpMassResL403 = TH1F("ZpMassResL403", "Zp Mass Resolution Leading Jets 3", 100, 0, 2);

  ZpMassResRL401 = TH1F("ZpMassResRL401", "Zp Mass Res. Ratio Leading Jets 1", 100, 0, 2);
  ZpMassResRL402 = TH1F("ZpMassResRL402", "Zp Mass Res. Ratio Leading Jets 2", 100, 0, 2);
  ZpMassResRL403 = TH1F("ZpMassResRL403", "Zp Mass Res. Ratio Leading Jets 3", 100, 0, 2);

  ZpMassResRLoP401 = TH1F("ZpMassResRLoP401", "Zp Mass RLoP Ratio Leading Jets 1", 100, 0, 2);
  ZpMassResRLoP402 = TH1F("ZpMassResRLoP402", "Zp Mass RLoP Ratio Leading Jets 2", 100, 0, 2);
  ZpMassResRLoP403 = TH1F("ZpMassResRLoP403", "Zp Mass RLoP Ratio Leading Jets 3", 100, 0, 2);

  ZpMassResPRL401 = TH1F("ZpMassResPRL401", "Zp Mass Res. P Ratio Leading Jets 1", 100, 0, 2);
  ZpMassResPRL402 = TH1F("ZpMassResPRL402", "Zp Mass Res. P Ratio Leading Jets 2", 100, 0, 2);
  ZpMassResPRL403 = TH1F("ZpMassResPRL403", "Zp Mass Res. P Ratio Leading Jets 3", 100, 0, 2);

  dijetMass1 = TH1F("dijetMass1", "DiJet Mass 1", 100, 0, 4000);
  dijetMass12 = TH1F("dijetMass12", "DiJet Mass 1 2", 100, 0, 6000);
  dijetMass13 = TH1F("dijetMass13", "DiJet Mass 1 3", 100, 0, 12000);
  dijetMass2 = TH1F("dijetMass2", "DiJet Mass 2", 100, 0, 4000);
  dijetMass22 = TH1F("dijetMass22", "DiJet Mass 2 2", 100, 0, 6000);
  dijetMass23 = TH1F("dijetMass23", "DiJet Mass 2 3", 100, 0, 12000);
  dijetMass3 = TH1F("dijetMass3", "DiJet Mass 3", 100, 0, 4000);
  dijetMass32 = TH1F("dijetMass32", "DiJet Mass 3 2", 100, 0, 6000);
  dijetMass33 = TH1F("dijetMass33", "DiJet Mass 3 3", 100, 0, 12000);
  dijetMass4 = TH1F("dijetMass4", "DiJet Mass 4", 100, 0, 4000);
  dijetMass42 = TH1F("dijetMass42", "DiJet Mass 4 2", 100, 0, 6000);
  dijetMass43 = TH1F("dijetMass43", "DiJet Mass 4 3", 100, 0, 12000);

  dijetMass101 = TH1F("dijetMass101", "DiJet Mass 1", 100, 0, 6000);
  dijetMass131 = TH1F("dijetMass131", "DiJet Mass 1", 100, 0, 6000);
  dijetMass401 = TH1F("dijetMass401", "DiJet Mass 1", 100, 0, 6000);

  dijetMass102 = TH1F("dijetMass102", "DiJet Mass 2", 100, 0, 6000);
  dijetMass132 = TH1F("dijetMass132", "DiJet Mass 2", 100, 0, 6000);
  dijetMass402 = TH1F("dijetMass402", "DiJet Mass 2", 100, 0, 6000);

  dijetMass103 = TH1F("dijetMass103", "DiJet Mass 3", 100, 0, 10000);
  dijetMass133 = TH1F("dijetMass133", "DiJet Mass 3", 100, 0, 10000);
  dijetMass403 = TH1F("dijetMass403", "DiJet Mass 3", 100, 0, 10000);

  dijetMass_700_101 = TH1F("dijetMass_700_101", "DiJet Mass 1", 100, 0, 1000);
  dijetMass_700_131 = TH1F("dijetMass_700_131", "DiJet Mass 1", 100, 0, 1000);
  dijetMass_700_401 = TH1F("dijetMass_700_401", "DiJet Mass 1", 100, 0, 1000);

  dijetMass_2000_101 = TH1F("dijetMass_2000_101", "DiJet Mass 1", 100, 1500, 2500);
  dijetMass_2000_131 = TH1F("dijetMass_2000_131", "DiJet Mass 1", 100, 1500, 2500);
  dijetMass_2000_401 = TH1F("dijetMass_2000_401", "DiJet Mass 1", 100, 1500, 2500);

  dijetMass_5000_101 = TH1F("dijetMass_5000_101", "DiJet Mass 1", 150, 4000, 5500);
  dijetMass_5000_131 = TH1F("dijetMass_5000_131", "DiJet Mass 1", 150, 4000, 5500);
  dijetMass_5000_401 = TH1F("dijetMass_5000_401", "DiJet Mass 1", 150, 4000, 5500);

  dijetMassCor1 = TH1F("dijetMassCor1", "DiJet Mass 1", 160, 0, 8000);
  dijetMassCor101 = TH1F("dijetMassCor101", "DiJet Mass Cor 101", 160, 0, 8000);
  dijetMassCor131 = TH1F("dijetMassCor131", "DiJet Mass Cor 131", 160, 0, 8000);
  dijetMassCor401 = TH1F("dijetMassCor401", "DiJet Mass Cor 401", 160, 0, 8000);

  dijetMassCor_700_1 = TH1F("dijetMassCor_700_1", "DiJet Mass 1", 100, 0, 1000);
  dijetMassCor_700_101 = TH1F("dijetMassCor_700_101", "DiJet Mass Cor 101", 100, 0, 1000);
  dijetMassCor_700_131 = TH1F("dijetMassCor_700_131", "DiJet Mass Cor 131", 100, 0, 1000);
  dijetMassCor_700_401 = TH1F("dijetMassCor_700_401", "DiJet Mass Cor 401", 100, 0, 1000);

  dijetMassCor_2000_1 = TH1F("dijetMassCor_2000_1", "DiJet Mass 1", 100, 1500, 2500);
  dijetMassCor_2000_101 = TH1F("dijetMassCor_2000_101", "DiJet Mass Cor 101", 100, 1500, 2500);
  dijetMassCor_2000_131 = TH1F("dijetMassCor_2000_131", "DiJet Mass Cor 131", 100, 1500, 2500);
  dijetMassCor_2000_401 = TH1F("dijetMassCor_2000_401", "DiJet Mass Cor 401", 100, 1500, 2500);

  dijetMassCor_5000_1 = TH1F("dijetMassCor_5000_1", "DiJet Mass 1", 150, 4000, 5500);
  dijetMassCor_5000_101 = TH1F("dijetMassCor_5000_101", "DiJet Mass Cor 101", 150, 4000, 5500);
  dijetMassCor_5000_131 = TH1F("dijetMassCor_5000_131", "DiJet Mass Cor 131", 150, 4000, 5500);
  dijetMassCor_5000_401 = TH1F("dijetMassCor_5000_401", "DiJet Mass Cor 401", 150, 4000, 5500);

  dijetMassP1 = TH1F("dijetMassP1", "DiJet Mass P 1", 160, 0, 8000);
  dijetMassP2 = TH1F("dijetMassP2", "DiJet Mass P 2", 160, 0, 8000);
  dijetMassP3 = TH1F("dijetMassP3", "DiJet Mass P 3", 160, 0, 8000);

  dijetMassP101 = TH1F("dijetMassP101", "DiJet Mass P 1", 160, 0, 8000);
  dijetMassP131 = TH1F("dijetMassP131", "DiJet Mass P 1", 160, 0, 8000);
  dijetMassP401 = TH1F("dijetMassP401", "DiJet Mass P 1", 160, 0, 8000);

  dijetMassP_700_101 = TH1F("dijetMassP_700_101", "DiJet Mass P 1", 100, 0, 1000);
  dijetMassP_700_131 = TH1F("dijetMassP_700_131", "DiJet Mass P 1", 100, 0, 1000);
  dijetMassP_700_401 = TH1F("dijetMassP_700_401", "DiJet Mass P 1", 100, 0, 1000);

  dijetMassP_2000_101 = TH1F("dijetMassP_2000_101", "DiJet Mass P 1", 100, 1500, 2500);
  dijetMassP_2000_131 = TH1F("dijetMassP_2000_131", "DiJet Mass P 1", 100, 1500, 2500);
  dijetMassP_2000_401 = TH1F("dijetMassP_2000_401", "DiJet Mass P 1", 100, 1500, 2500);

  dijetMassP_5000_101 = TH1F("dijetMassP_5000_101", "DiJet Mass P 1", 150, 4000, 5500);
  dijetMassP_5000_131 = TH1F("dijetMassP_5000_131", "DiJet Mass P 1", 150, 4000, 5500);
  dijetMassP_5000_401 = TH1F("dijetMassP_5000_401", "DiJet Mass P 1", 150, 4000, 5500);

  totEneLeadJetEta1_1 = TH1F("totEneLeadJetEta1_1", "Total Energy Lead Jet Eta1 1", 100, 0, 1500);
  totEneLeadJetEta2_1 = TH1F("totEneLeadJetEta2_1", "Total Energy Lead Jet Eta2 1", 100, 0, 1500);
  totEneLeadJetEta3_1 = TH1F("totEneLeadJetEta3_1", "Total Energy Lead Jet Eta3 1", 100, 0, 1500);
  hadEneLeadJetEta1_1 = TH1F("hadEneLeadJetEta1_1", "Hadronic Energy Lead Jet Eta1 1", 100, 0, 1500);
  hadEneLeadJetEta2_1 = TH1F("hadEneLeadJetEta2_1", "Hadronic Energy Lead Jet Eta2 1", 100, 0, 1500);
  hadEneLeadJetEta3_1 = TH1F("hadEneLeadJetEta3_1", "Hadronic Energy Lead Jet Eta3 1", 100, 0, 1500);
  emEneLeadJetEta1_1 = TH1F("emEneLeadJetEta1_1", "EM Energy Lead Jet Eta1 1", 100, 0, 1500);
  emEneLeadJetEta2_1 = TH1F("emEneLeadJetEta2_1", "EM Energy Lead Jet Eta2 1", 100, 0, 1500);
  emEneLeadJetEta3_1 = TH1F("emEneLeadJetEta3_1", "EM Energy Lead Jet Eta3 1", 100, 0, 1500);

  totEneLeadJetEta1_2 = TH1F("totEneLeadJetEta1_2", "Total Energy Lead Jet Eta1 2", 100, 0, 6000);
  totEneLeadJetEta2_2 = TH1F("totEneLeadJetEta2_2", "Total Energy Lead Jet Eta2 2", 100, 0, 6000);
  totEneLeadJetEta3_2 = TH1F("totEneLeadJetEta3_2", "Total Energy Lead Jet Eta3 2", 100, 0, 6000);
  hadEneLeadJetEta1_2 = TH1F("hadEneLeadJetEta1_2", "Hadronic Energy Lead Jet Eta1 2", 100, 0, 6000);
  hadEneLeadJetEta2_2 = TH1F("hadEneLeadJetEta2_2", "Hadronic Energy Lead Jet Eta2 2", 100, 0, 6000);
  hadEneLeadJetEta3_2 = TH1F("hadEneLeadJetEta3_2", "Hadronic Energy Lead Jet Eta3 2", 100, 0, 6000);
  emEneLeadJetEta1_2 = TH1F("emEneLeadJetEta1_2", "EM Energy Lead Jet Eta1 2", 100, 0, 5000);
  emEneLeadJetEta2_2 = TH1F("emEneLeadJetEta2_2", "EM Energy Lead Jet Eta2 2", 100, 0, 5000);
  emEneLeadJetEta3_2 = TH1F("emEneLeadJetEta3_2", "EM Energy Lead Jet Eta3 2", 100, 0, 5000);

  hadEneLeadJet1 = TH1F("hadEneLeadJet1", "Hadronic Energy Lead Jet 1", 100, 0, 3000);
  hadEneLeadJet12 = TH1F("hadEneLeadJet12", "Hadronic Energy Lead Jet 1 2", 100, 0, 4000);
  hadEneLeadJet13 = TH1F("hadEneLeadJet13", "Hadronic Energy Lead Jet 1 3", 100, 0, 6000);
  hadEneLeadJet2 = TH1F("hadEneLeadJet2", "Hadronic Energy Lead Jet 2", 100, 0, 3000);
  hadEneLeadJet22 = TH1F("hadEneLeadJet22", "Hadronic Energy Lead Jet 2 2", 100, 0, 4000);
  hadEneLeadJet23 = TH1F("hadEneLeadJet23", "Hadronic Energy Lead Jet 2 3", 100, 0, 6000);
  hadEneLeadJet3 = TH1F("hadEneLeadJet3", "Hadronic Energy Lead Jet 3", 100, 0, 3000);
  hadEneLeadJet32 = TH1F("hadEneLeadJet32", "Hadronic Energy Lead Jet 3 2", 100, 0, 4000);
  hadEneLeadJet33 = TH1F("hadEneLeadJet33", "Hadronic Energy Lead Jet 3 3", 100, 0, 6000);

  emEneLeadJet1 = TH1F("emEneLeadJet1", "EM Energy Lead Jet 1", 100, 0, 1500);
  emEneLeadJet12 = TH1F("emEneLeadJet12", "EM Energy Lead Jet 1 2", 100, 0, 3000);
  emEneLeadJet13 = TH1F("emEneLeadJet13", "EM Energy Lead Jet 1 3", 100, 0, 5000);
  emEneLeadJet2 = TH1F("emEneLeadJet2", "EM Energy Lead Jet 2", 100, 0, 1500);
  emEneLeadJet22 = TH1F("emEneLeadJet22", "EM Energy Lead Jet 2 2", 100, 0, 3000);
  emEneLeadJet23 = TH1F("emEneLeadJet23", "EM Energy Lead Jet 2 3", 100, 0, 5000);
  emEneLeadJet3 = TH1F("emEneLeadJet3", "EM Energy Lead Jet 3", 100, 0, 1500);
  emEneLeadJet32 = TH1F("emEneLeadJet32", "EM Energy Lead Jet 3 2", 100, 0, 3000);
  emEneLeadJet33 = TH1F("emEneLeadJet33", "EM Energy Lead Jet 3 3", 100, 0, 5000);

  hadFracEta11 = TH1F("hadFracEta11", "Hadronic Fraction Eta1 Jet 1", 100, 0, 1);
  hadFracEta21 = TH1F("hadFracEta21", "Hadronic Fraction Eta2 Jet 1", 100, 0, 1);
  hadFracEta31 = TH1F("hadFracEta31", "Hadronic Fraction Eta3 Jet 1", 100, 0, 1);

  hadFracEta12 = TH1F("hadFracEta12", "Hadronic Fraction Eta1 Jet 2", 100, 0, 1);
  hadFracEta22 = TH1F("hadFracEta22", "Hadronic Fraction Eta2 Jet 2", 100, 0, 1);
  hadFracEta32 = TH1F("hadFracEta32", "Hadronic Fraction Eta3 Jet 2", 100, 0, 1);

  hadFracEta13 = TH1F("hadFracEta13", "Hadronic Fraction Eta1 Jet 3", 100, 0, 1);
  hadFracEta23 = TH1F("hadFracEta23", "Hadronic Fraction Eta2 Jet 3", 100, 0, 1);
  hadFracEta33 = TH1F("hadFracEta33", "Hadronic Fraction Eta3 Jet 3", 100, 0, 1);

  hadFracLeadJet1 = TH1F("hadFracLeadJet1", "Hadronic Fraction Lead Jet 1", 100, 0, 1);
  hadFracLeadJet2 = TH1F("hadFracLeadJet2", "Hadronic Fraction Lead Jet 2", 100, 0, 1);
  hadFracLeadJet3 = TH1F("hadFracLeadJet3", "Hadronic Fraction Lead Jet 3", 100, 0, 1);

  SumEt1 = TH1F("SumEt1", "SumEt 1", 100, 0, 1000);
  SumEt12 = TH1F("SumEt12", "SumEt 1 2", 100, 0, 4000);
  SumEt13 = TH1F("SumEt13", "SumEt 1 3", 100, 0, 15000);

  MET1 = TH1F("MET1", "MET 1", 100, 0, 200);
  MET12 = TH1F("MET12", "MET 1 2", 100, 0, 1000);
  MET13 = TH1F("MET13", "MET 1 3", 100, 0, 3000);

  nTowersLeadJet1 = TH1F("nTowersLeadJet1", "Number of Towers Lead Jet 1", 100, 0, 100);
  nTowersLeadJet2 = TH1F("nTowersLeadJet2", "Number of Towers Lead Jet 2", 100, 0, 100);
  nTowersLeadJet3 = TH1F("nTowersLeadJet3", "Number of Towers Lead Jet 3", 100, 0, 100);

  nTowersSecondJet1 = TH1F("nTowersSecondJet1", "Number of Towers Second Jet 1", 100, 0, 100);
  nTowersSecondJet2 = TH1F("nTowersSecondJet2", "Number of Towers Second Jet 2", 100, 0, 100);
  nTowersSecondJet3 = TH1F("nTowersSecondJet3", "Number of Towers Second Jet 3", 100, 0, 100);

  hf_PtResponse1 = TProfile("PtResponse1", "Pt Response 1", 100, -5, 5, 0, 10);
  hf_PtResponse2 = TProfile("PtResponse2", "Pt Response 2", 100, -5, 5, 0, 10);
  hf_PtResponse3 = TProfile("PtResponse3", "Pt Response 3", 100, -5, 5, 0, 10);
  hf_PtResponse4 = TProfile("PtResponse4", "Pt Response 4", 100, -5, 5, 0, 10);

  hf_TowerDelR1 = TProfile("hf_TowerDelR1", "Tower Del R 1", 100, 0, 2, 0, 10);
  hf_TowerDelR12 = TProfile("hf_TowerDelR12", "Tower Del R 1", 80, 0, 0.8, 0, 10);
  hf_TowerDelR2 = TProfile("hf_TowerDelR2", "Tower Del R 2", 100, 0, 2, 0, 10);
  hf_TowerDelR22 = TProfile("hf_TowerDelR22", "Tower Del R 2", 80, 0, 0.8, 0, 10);
  hf_TowerDelR3 = TProfile("hf_TowerDelR3", "Tower Del R 3", 100, 0, 2, 0, 10);
  hf_TowerDelR32 = TProfile("hf_TowerDelR32", "Tower Del R 3", 80, 0, 0.8, 0, 10);

  hf_sumTowerAllEx = TH1F("sumTowerAllEx", "Tower Ex", 100, -1000, 1000);
  hf_sumTowerAllEy = TH1F("sumTowerAllEy", "Tower Ey", 100, -1000, 1000);

  hf_TowerJetEt1 = TH1F("TowerJetEt1", "Tower/Jet Et 1", 50, 0, 1);

  nTowers1 = TH1F("nTowers1", "Number of Towers pt 0.5", 100, 0, 500);
  nTowers2 = TH1F("nTowers2", "Number of Towers pt 1.0", 100, 0, 500);
  nTowers3 = TH1F("nTowers3", "Number of Towers pt 1.5", 100, 0, 500);
  nTowers4 = TH1F("nTowers4", "Number of Towers pt 2.0", 100, 0, 500);

  nTowersLeadJetPt1 = TH1F("nTowersLeadJetPt1", "Number of Towers in Lead Jet pt 0.5", 100, 0, 200);
  nTowersLeadJetPt2 = TH1F("nTowersLeadJetPt2", "Number of Towers in Lead Jet pt 1.0", 100, 0, 200);
  nTowersLeadJetPt3 = TH1F("nTowersLeadJetPt3", "Number of Towers in Lead Jet pt 1.5", 100, 0, 200);
  nTowersLeadJetPt4 = TH1F("nTowersLeadJetPt4", "Number of Towers in Lead Jet pt 2.0", 100, 0, 200);

  TowerEtLeadJet1 = TH1F("TowerEtLeadJet1", "Towers Et Lead Jet 1", 100, 0, 2000);
  TowerEtLeadJet12 = TH1F("TowerEtLeadJet12", "Towers Et Lead Jet 1 2", 100, 0, 6000);
  TowerEtLeadJet13 = TH1F("TowerEtLeadJet13", "Towers Et Lead Jet 1 3", 100, 0, 300);
  TowerEtLeadJet2 = TH1F("TowerEtLeadJet2", "Towers Et Lead Jet 2", 100, 0, 2000);
  TowerEtLeadJet22 = TH1F("TowerEtLeadJet22", "Towers Et Lead Jet 2 2", 100, 0, 6000);
  TowerEtLeadJet23 = TH1F("TowerEtLeadJet23", "Towers Et Lead Jet 2 3", 100, 0, 300);
  TowerEtLeadJet3 = TH1F("TowerEtLeadJet3", "Towers Et Lead Jet 3", 100, 0, 2000);
  TowerEtLeadJet32 = TH1F("TowerEtLeadJet32", "Towers Et Lead Jet 3 2", 100, 0, 6000);
  TowerEtLeadJet33 = TH1F("TowerEtLeadJet33", "Towers Et Lead Jet 3 3", 100, 0, 300);

  hf_nJet1 = TProfile("hf_nJet1", "Num Jets 1", 100, 0, 5000, 0, 50);
  hf_nJet2 = TProfile("hf_nJet2", "Num Jets 2", 100, 0, 5000, 0, 50);
  hf_nJet3 = TProfile("hf_nJet3", "Num Jets 3", 100, 0, 5000, 0, 50);
  hf_nJet4 = TProfile("hf_nJet4", "Num Jets 4", 100, 0, 5000, 0, 50);

  hf_nJet1s = TProfile("hf_nJet1s", "Num Jets 1", 100, 0, 200, 0, 50);
  hf_nJet2s = TProfile("hf_nJet2s", "Num Jets 2", 100, 0, 200, 0, 50);
  hf_nJet3s = TProfile("hf_nJet3s", "Num Jets 3", 100, 0, 200, 0, 50);
  hf_nJet4s = TProfile("hf_nJet4s", "Num Jets 4", 100, 0, 200, 0, 50);

  hf_nJet11 = TProfile("hf_nJet11", "Num Jets 1 1", 100, 0, 3000, 0, 50);
  hf_nJet21 = TProfile("hf_nJet21", "Num Jets 2 1", 100, 0, 3000, 0, 50);
  hf_nJet31 = TProfile("hf_nJet31", "Num Jets 3 1", 100, 0, 3000, 0, 50);
  hf_nJet41 = TProfile("hf_nJet41", "Num Jets 4 1", 100, 0, 3000, 0, 50);

  dRPar1 = TH1F("dRPar1", "Parton dR with matched CaloJet1", 100, 0, 0.5);
  dPhiPar1 = TH1F("dPhiPar1", "Parton dPhi with matched CaloJet1", 200, -0.5, 0.5);
  dEtaPar1 = TH1F("dEtaPar1", "Parton dEta with matched CaloJet1", 200, -0.5, 0.5);
  dPtPar1 = TH1F("dPtPar1", "Parton dPt with matched CaloJet1", 200, -50, 50);

  dRPar2 = TH1F("dRPar2", "Parton dR with matched CaloJet2", 100, 0, 0.5);
  dPhiPar2 = TH1F("dPhiPar2", "Parton dPhi with matched CaloJet2", 200, -0.5, 0.5);
  dEtaPar2 = TH1F("dEtaPar2", "Parton dEta with matched CaloJet2", 200, -0.5, 0.5);
  dPtPar2 = TH1F("dPtPar2", "Parton dPt with matched CaloJet2", 200, -50, 50);

  dRPar3 = TH1F("dRPar3", "Parton dR with matched CaloJet3", 100, 0, 0.5);
  dPhiPar3 = TH1F("dPhiPar3", "Parton dPhi with matched CaloJet3", 200, -0.5, 0.5);
  dEtaPar3 = TH1F("dEtaPar3", "Parton dEta with matched CaloJet3", 200, -0.5, 0.5);
  dPtPar3 = TH1F("dPtPar3", "Parton dPt with matched CaloJet3", 200, -50, 50);

  dRPar4 = TH1F("dRPar4", "Parton dR with matched CaloJet4", 100, 0, 0.5);
  dPhiPar4 = TH1F("dPhiPar4", "Parton dPhi with matched CaloJet4", 200, -0.5, 0.5);
  dEtaPar4 = TH1F("dEtaPar4", "Parton dEta with matched CaloJet4", 200, -0.5, 0.5);
  dPtPar4 = TH1F("dPtPar4", "Parton dPt with matched CaloJet4", 200, -50, 50);

  dRParton = TH1F("dRParton", "dR Parton", 100, 0, 10.0);
  dRPartonMin = TH1F("dRPartonMin", "Min dR Parton", 100, 0, 2.0);

  dR1 = TH1F("dR1", "GenJets dR with matched CaloJet", 100, 0, 0.5);
  dPhi1 = TH1F("dPhi1", "GenJets dPhi with matched CaloJet", 200, -0.5, 0.5);
  dEta1 = TH1F("dEta1", "GenJets dEta with matched CaloJet", 200, -0.5, 0.5);
  dPt1 = TH1F("dPt1", "GenJets dPt with matched CaloJet", 200, -100, 100);
  dPtFrac1 = TH1F("dPtFrac1", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt20Frac1 = TH1F("dPt20Frac1", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt40Frac1 = TH1F("dPt40Frac1", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt80Frac1 = TH1F("dPt80Frac1", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt100Frac1 = TH1F("dPt100Frac1", "GenJets dPt frac with matched CaloJet", 100, -1, 1);

  dR2 = TH1F("dR2", "GenJets dR with matched CaloJet", 100, 0, 0.5);
  dPhi2 = TH1F("dPhi2", "GenJets dPhi with matched CaloJet", 200, -0.5, 0.5);
  dEta2 = TH1F("dEta2", "GenJets dEta with matched CaloJet", 200, -0.5, 0.5);
  dPt2 = TH1F("dPt2", "GenJets dPt with matched CaloJet", 200, -100, 100);
  dPtFrac2 = TH1F("dPtFrac2", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt20Frac2 = TH1F("dPt20Frac2", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt40Frac2 = TH1F("dPt40Frac2", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt80Frac2 = TH1F("dPt80Frac2", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt100Frac2 = TH1F("dPt100Frac2", "GenJets dPt frac with matched CaloJet", 100, -1, 1);

  dR3 = TH1F("dR3", "GenJets dR with matched CaloJet", 100, 0, 0.5);
  dPhi3 = TH1F("dPhi3", "GenJets dPhi with matched CaloJet", 200, -0.5, 0.5);
  dEta3 = TH1F("dEta3", "GenJets dEta with matched CaloJet", 200, -0.5, 0.5);
  dPt3 = TH1F("dPt3", "GenJets dPt with matched CaloJet", 200, -100, 100);
  dPtFrac3 = TH1F("dPtFrac3", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt20Frac3 = TH1F("dPt20Frac3", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt40Frac3 = TH1F("dPt40Frac3", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt80Frac3 = TH1F("dPt80Frac3", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt100Frac3 = TH1F("dPt100Frac3", "GenJets dPt frac with matched CaloJet", 100, -1, 1);

  dR4 = TH1F("dR4", "GenJets dR with matched CaloJet", 100, 0, 0.5);
  dPhi4 = TH1F("dPhi4", "GenJets dPhi with matched CaloJet", 200, -0.5, 0.5);
  dEta4 = TH1F("dEta4", "GenJets dEta with matched CaloJet", 200, -0.5, 0.5);
  dPt4 = TH1F("dPt4", "GenJets dPt with matched CaloJet", 200, -100, 100);
  dPtFrac4 = TH1F("dPtFrac4", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt20Frac4 = TH1F("dPt20Frac4", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt40Frac4 = TH1F("dPt40Frac4", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt80Frac4 = TH1F("dPt80Frac4", "GenJets dPt frac with matched CaloJet", 100, -1, 1);
  dPt100Frac4 = TH1F("dPt100Frac4", "GenJets dPt frac with matched CaloJet", 100, -1, 1);

  dR12 = TH1F("dR12", "dR MidPoint - SISCone", 100, 0, 0.5);
  dPhi12 = TH1F("dPhi12", "dPhi MidPoint - SISCone", 200, -0.5, 0.5);
  dEta12 = TH1F("dEta12", "dEta MidPoint - SISCone", 200, -0.5, 0.5);
  dPt12 = TH1F("dPt12", "dPt MidPoint - SISCone", 200, -100, 100);

  h_nCalJets1 = TH1F("nCalJets1", "Number of CalJets1", 20, 0, 20);
  h_nCalJets2 = TH1F("nCalJets2", "Number of CalJets2", 20, 0, 20);
  h_nCalJets3 = TH1F("nCalJets3", "Number of CalJets3", 20, 0, 20);
  h_nCalJets4 = TH1F("nCalJets4", "Number of CalJets4", 20, 0, 20);

  h_lowPtCal11 = TH1F("lowPtCal11", "Low p_{T} of CalJet1 Eta1", 100, 0, 100);
  h_lowPtCal21 = TH1F("lowPtCal21", "Low p_{T} of CalJet2 Eta1", 100, 0, 100);
  h_lowPtCal31 = TH1F("lowPtCal31", "Low p_{T} of CalJet3 Eta1", 100, 0, 100);
  h_lowPtCal41 = TH1F("lowPtCal41", "Low p_{T} of CalJet4 Eta1", 100, 0, 100);

  h_lowPtCal12 = TH1F("lowPtCal12", "Low p_{T} of CalJet1 Eta2", 100, 0, 100);
  h_lowPtCal22 = TH1F("lowPtCal22", "Low p_{T} of CalJet2 Eta2", 100, 0, 100);
  h_lowPtCal32 = TH1F("lowPtCal32", "Low p_{T} of CalJet3 Eta2", 100, 0, 100);
  h_lowPtCal42 = TH1F("lowPtCal42", "Low p_{T} of CalJet4 Eta2", 100, 0, 100);

  h_lowPtCal13 = TH1F("lowPtCal13", "Low p_{T} of CalJet1 Eta3", 100, 0, 100);
  h_lowPtCal23 = TH1F("lowPtCal23", "Low p_{T} of CalJet2 Eta3", 100, 0, 100);
  h_lowPtCal33 = TH1F("lowPtCal33", "Low p_{T} of CalJet3 Eta3", 100, 0, 100);
  h_lowPtCal43 = TH1F("lowPtCal43", "Low p_{T} of CalJet4 Eta3", 100, 0, 100);

  h_lowPtCal1c11 = TH1F("lowPtCal1c11", "Low p_{T} of CalJet1 c1 Eta1", 100, 0, 100);
  h_lowPtCal2c11 = TH1F("lowPtCal2c11", "Low p_{T} of CalJet2 c1 Eta1", 100, 0, 100);
  h_lowPtCal3c11 = TH1F("lowPtCal3c11", "Low p_{T} of CalJet3 c1 Eta1", 100, 0, 100);
  h_lowPtCal4c11 = TH1F("lowPtCal4c11", "Low p_{T} of CalJet4 c1 Eta1", 100, 0, 100);

  h_lowPtCal1c12 = TH1F("lowPtCal1c12", "Low p_{T} of CalJet1 c1 Eta2", 100, 0, 100);
  h_lowPtCal2c12 = TH1F("lowPtCal2c12", "Low p_{T} of CalJet2 c1 Eta2", 100, 0, 100);
  h_lowPtCal3c12 = TH1F("lowPtCal3c12", "Low p_{T} of CalJet3 c1 Eta2", 100, 0, 100);
  h_lowPtCal4c12 = TH1F("lowPtCal4c12", "Low p_{T} of CalJet4 c1 Eta2", 100, 0, 100);

  h_lowPtCal1c13 = TH1F("lowPtCal1c13", "Low p_{T} of CalJet1 c1 Eta3", 100, 0, 100);
  h_lowPtCal2c13 = TH1F("lowPtCal2c13", "Low p_{T} of CalJet2 c1 Eta3", 100, 0, 100);
  h_lowPtCal3c13 = TH1F("lowPtCal3c13", "Low p_{T} of CalJet3 c1 Eta3", 100, 0, 100);
  h_lowPtCal4c13 = TH1F("lowPtCal4c13", "Low p_{T} of CalJet4 c1 Eta3", 100, 0, 100);

  matchedAllPt11 = TH1F("matchedAllPt11", "p_{T} of CalJet1 Eta1", 50, 0, 250);
  matchedAllPt12 = TH1F("matchedAllPt12", "p_{T} of CalJet1 Eta2", 50, 0, 250);
  matchedAllPt13 = TH1F("matchedAllPt13", "p_{T} of CalJet1 Eta3", 50, 0, 250);
  matchedPt11 = TH1F("matchedPt11", "p_{T} of CalJet1 Eta1", 50, 0, 250);
  matchedPt12 = TH1F("matchedPt12", "p_{T} of CalJet1 Eta2", 50, 0, 250);
  matchedPt13 = TH1F("matchedPt13", "p_{T} of CalJet1 Eta3", 50, 0, 250);

  matchedAllPt21 = TH1F("matchedAllPt21", "p_{T} of CalJet2 Eta1", 50, 0, 250);
  matchedAllPt22 = TH1F("matchedAllPt22", "p_{T} of CalJet2 Eta2", 50, 0, 250);
  matchedAllPt23 = TH1F("matchedAllPt23", "p_{T} of CalJet2 Eta3", 50, 0, 250);
  matchedPt21 = TH1F("matchedPt21", "p_{T} of CalJet2 Eta1", 50, 0, 250);
  matchedPt22 = TH1F("matchedPt22", "p_{T} of CalJet2 Eta2", 50, 0, 250);
  matchedPt23 = TH1F("matchedPt23", "p_{T} of CalJet2 Eta3", 50, 0, 250);

  matchedAllPt31 = TH1F("matchedAllPt31", "p_{T} of CalJet3 Eta1", 50, 0, 250);
  matchedAllPt32 = TH1F("matchedAllPt32", "p_{T} of CalJet3 Eta2", 50, 0, 250);
  matchedAllPt33 = TH1F("matchedAllPt33", "p_{T} of CalJet3 Eta3", 50, 0, 250);
  matchedPt31 = TH1F("matchedPt31", "p_{T} of CalJet3 Eta1", 50, 0, 250);
  matchedPt32 = TH1F("matchedPt32", "p_{T} of CalJet3 Eta2", 50, 0, 250);
  matchedPt33 = TH1F("matchedPt33", "p_{T} of CalJet3 Eta3", 50, 0, 250);

  matchedAllPt41 = TH1F("matchedAllPt41", "p_{T} of CalJet4 Eta1", 50, 0, 250);
  matchedAllPt42 = TH1F("matchedAllPt42", "p_{T} of CalJet4 Eta2", 50, 0, 250);
  matchedAllPt43 = TH1F("matchedAllPt43", "p_{T} of CalJet4 Eta3", 50, 0, 250);
  matchedPt41 = TH1F("matchedPt41", "p_{T} of CalJet4 Eta1", 50, 0, 250);
  matchedPt42 = TH1F("matchedPt42", "p_{T} of CalJet4 Eta2", 50, 0, 250);
  matchedPt43 = TH1F("matchedPt43", "p_{T} of CalJet4 Eta3", 50, 0, 250);

  h_ptCal1 = TH1F("ptCal1", "p_{T} of CalJet1", 50, 0, 1000);
  h_ptCal12 = TH1F("ptCal12", "p_{T} of CalJet1 2", 50, 0, 6000);
  h_ptCal13 = TH1F("ptCal13", "p_{T} of CalJet1 2", 50, 0, 300);

  h_ptCal2 = TH1F("ptCal2", "p_{T} of CalJet2", 50, 0, 1000);
  h_ptCal22 = TH1F("ptCal22", "p_{T} of CalJet2 2", 50, 0, 6000);
  h_ptCal23 = TH1F("ptCal23", "p_{T} of CalJet2 2", 50, 0, 300);

  h_ptCal3 = TH1F("ptCal3", "p_{T} of CalJet3", 50, 0, 1000);
  h_ptCal32 = TH1F("ptCal32", "p_{T} of CalJet3 2", 50, 0, 6000);
  h_ptCal33 = TH1F("ptCal33", "p_{T} of CalJet3 2", 50, 0, 300);

  h_ptCal4 = TH1F("ptCal4", "p_{T} of CalJet4", 50, 0, 1000);
  h_ptCal42 = TH1F("ptCal42", "p_{T} of CalJet4 2", 50, 0, 6000);
  h_ptCal43 = TH1F("ptCal43", "p_{T} of CalJet4 2", 50, 0, 300);

  h_etaCal1 = TH1F("etaCal1", "#eta of  CalJet1", 100, -4, 4);
  h_etaCal2 = TH1F("etaCal2", "#eta of  CalJet2", 100, -4, 4);
  h_etaCal3 = TH1F("etaCal3", "#eta of  CalJet3", 100, -4, 4);
  h_etaCal4 = TH1F("etaCal4", "#eta of  CalJet4", 100, -4, 4);

  h_phiCal1 = TH1F("phiCal1", "#phi of  CalJet1", 50, -M_PI, M_PI);
  h_phiCal2 = TH1F("phiCal2", "#phi of  CalJet2", 50, -M_PI, M_PI);
  h_phiCal3 = TH1F("phiCal3", "#phi of  CalJet3", 50, -M_PI, M_PI);
  h_phiCal4 = TH1F("phiCal4", "#phi of  CalJet4", 50, -M_PI, M_PI);

  h_ptCalL1 = TH1F("ptCalL1", "p_{T} of CalJetL1", 50, 0, 300);
  h_ptCalL12 = TH1F("ptCalL12", "p_{T} of CalJetL1 2", 50, 0, 1200);
  h_ptCalL13 = TH1F("ptCalL13", "p_{T} of CalJetL1 3", 50, 0, 6000);
  h_ptCalL2 = TH1F("ptCalL2", "p_{T} of CalJetL2", 50, 0, 300);
  h_ptCalL22 = TH1F("ptCalL22", "p_{T} of CalJetL2 2", 50, 0, 1200);
  h_ptCalL23 = TH1F("ptCalL23", "p_{T} of CalJetL2 3", 50, 0, 6000);
  h_ptCalL3 = TH1F("ptCalL3", "p_{T} of CalJetL3", 50, 0, 300);
  h_ptCalL32 = TH1F("ptCalL32", "p_{T} of CalJetL3 2", 50, 0, 1200);
  h_ptCalL33 = TH1F("ptCalL33", "p_{T} of CalJetL3 3", 50, 0, 6000);
  h_ptCalL4 = TH1F("ptCalL4", "p_{T} of CalJetL4", 50, 0, 300);
  h_ptCalL42 = TH1F("ptCalL42", "p_{T} of CalJetL4 2", 50, 0, 1200);
  h_ptCalL43 = TH1F("ptCalL43", "p_{T} of CalJetL4 3", 50, 0, 6000);

  h_etaCalL1 = TH1F("etaCalL1", "#eta of  CalJetL1", 100, -4, 4);
  h_etaCalL2 = TH1F("etaCalL2", "#eta of  CalJetL2", 100, -4, 4);
  h_etaCalL3 = TH1F("etaCalL3", "#eta of  CalJetL3", 100, -4, 4);
  h_etaCalL4 = TH1F("etaCalL4", "#eta of  CalJetL4", 100, -4, 4);
  h_phiCalL1 = TH1F("phiCalL1", "#phi of  CalJetL1", 50, -M_PI, M_PI);
  h_phiCalL2 = TH1F("phiCalL2", "#phi of  CalJetL2", 50, -M_PI, M_PI);
  h_phiCalL3 = TH1F("phiCalL3", "#phi of  CalJetL3", 50, -M_PI, M_PI);
  h_phiCalL4 = TH1F("phiCalL4", "#phi of  CalJetL4", 50, -M_PI, M_PI);

  h_nGenJets1 = TH1F("nGenJets1", "Number of GenJets1", 20, 0, 20);
  h_nGenJets2 = TH1F("nGenJets2", "Number of GenJets2", 20, 0, 20);
  h_nGenJets3 = TH1F("nGenJets3", "Number of GenJets3", 20, 0, 20);
  h_nGenJets4 = TH1F("nGenJets4", "Number of GenJets4", 20, 0, 20);

  h_ptGen1 = TH1F("ptGen1", "p_{T} of GenJet1", 50, 0, 1000);
  h_ptGen12 = TH1F("ptGen12", "p_{T} of GenJet1 2", 50, 0, 6000);
  h_ptGen13 = TH1F("ptGen13", "p_{T} of GenJet1 3", 50, 0, 300);

  h_ptGen2 = TH1F("ptGen2", "p_{T} of GenJet2", 50, 0, 1000);
  h_ptGen22 = TH1F("ptGen22", "p_{T} of GenJet2 2", 50, 0, 6000);
  h_ptGen23 = TH1F("ptGen23", "p_{T} of GenJet2 3", 50, 0, 300);

  h_ptGen3 = TH1F("ptGen3", "p_{T} of GenJet3", 50, 0, 1000);
  h_ptGen32 = TH1F("ptGen32", "p_{T} of GenJet3 2", 50, 0, 6000);
  h_ptGen33 = TH1F("ptGen33", "p_{T} of GenJet3 3", 50, 0, 300);

  h_ptGen4 = TH1F("ptGen4", "p_{T} of GenJet4", 50, 0, 1000);
  h_ptGen42 = TH1F("ptGen42", "p_{T} of GenJet4 2", 50, 0, 6000);
  h_ptGen43 = TH1F("ptGen43", "p_{T} of GenJet4 3", 50, 0, 300);

  h_etaGen1 = TH1F("etaGen1", "#eta of GenJet1", 100, -4, 4);
  h_etaGen2 = TH1F("etaGen2", "#eta of GenJet2", 100, -4, 4);
  h_etaGen3 = TH1F("etaGen3", "#eta of GenJet3", 100, -4, 4);
  h_phiGen1 = TH1F("phiGen1", "#phi of GenJet1", 50, -M_PI, M_PI);
  h_phiGen2 = TH1F("phiGen2", "#phi of GenJet2", 50, -M_PI, M_PI);
  h_phiGen3 = TH1F("phiGen3", "#phi of GenJet3", 50, -M_PI, M_PI);

  h_ptGenL1 = TH1F("ptGenL1", "p_{T} of GenJetL1", 50, 0, 300);
  h_ptGenL12 = TH1F("ptGenL12", "p_{T} of GenJetL1 2", 50, 0, 1200);
  h_ptGenL13 = TH1F("ptGenL13", "p_{T} of GenJetL1 3", 50, 0, 6000);
  h_ptGenL2 = TH1F("ptGenL2", "p_{T} of GenJetL2", 50, 0, 300);
  h_ptGenL22 = TH1F("ptGenL22", "p_{T} of GenJetL2 2", 50, 0, 1200);
  h_ptGenL23 = TH1F("ptGenL23", "p_{T} of GenJetL2 3", 50, 0, 6000);
  h_ptGenL3 = TH1F("ptGenL3", "p_{T} of GenJetL3", 50, 0, 300);
  h_ptGenL32 = TH1F("ptGenL32", "p_{T} of GenJetL32", 50, 0, 1200);
  h_ptGenL33 = TH1F("ptGenL33", "p_{T} of GenJetL33", 50, 0, 6000);

  h_etaGenL1 = TH1F("etaGenL1", "#eta of GenJetL1", 100, -4, 4);
  h_etaGenL2 = TH1F("etaGenL2", "#eta of GenJetL2", 100, -4, 4);
  h_etaGenL3 = TH1F("etaGenL3", "#eta of GenJetL3", 100, -4, 4);
  h_phiGenL1 = TH1F("phiGenL1", "#phi of GenJetL1", 50, -M_PI, M_PI);
  h_phiGenL2 = TH1F("phiGenL2", "#phi of GenJetL2", 50, -M_PI, M_PI);
  h_phiGenL3 = TH1F("phiGenL3", "#phi of GenJetL3", 50, -M_PI, M_PI);

  h_jetEt1 = TH1F("jetEt1", "Total Jet Et", 100, 0, 3000);
  h_jetEt2 = TH1F("jetEt2", "Total Jet Et", 100, 0, 3000);
  h_jetEt3 = TH1F("jetEt3", "Total Jet Et", 100, 0, 3000);

  h_jet1Pt1 = TH1F("jet1Pt1", "Jet Pt", 100, 0, 3000);
  h_jet2Pt1 = TH1F("jet2Pt1", "Jet Pt", 100, 0, 3000);
  h_jet3Pt1 = TH1F("jet3Pt1", "Jet Pt", 100, 0, 3000);
  h_jet4Pt1 = TH1F("jet4Pt1", "Jet Pt", 100, 0, 3000);
  h_jet5Pt1 = TH1F("jet5Pt1", "Jet Pt", 100, 0, 3000);
  h_jet6Pt1 = TH1F("jet6Pt1", "Jet Pt", 100, 0, 3000);
  h_jet7Pt1 = TH1F("jet7Pt1", "Jet Pt", 100, 0, 3000);

  h_jet1Pt2 = TH1F("jet1Pt2", "Jet Pt", 100, 0, 3000);
  h_jet2Pt2 = TH1F("jet2Pt2", "Jet Pt", 100, 0, 3000);
  h_jet3Pt2 = TH1F("jet3Pt2", "Jet Pt", 100, 0, 3000);
  h_jet4Pt2 = TH1F("jet4Pt2", "Jet Pt", 100, 0, 3000);
  h_jet5Pt2 = TH1F("jet5Pt2", "Jet Pt", 100, 0, 3000);
  h_jet6Pt2 = TH1F("jet6Pt2", "Jet Pt", 100, 0, 3000);
  h_jet7Pt2 = TH1F("jet7Pt2", "Jet Pt", 100, 0, 3000);

  h_jet1Pt3 = TH1F("jet1Pt3", "Jet Pt", 100, 0, 3000);
  h_jet2Pt3 = TH1F("jet2Pt3", "Jet Pt", 100, 0, 3000);
  h_jet3Pt3 = TH1F("jet3Pt3", "Jet Pt", 100, 0, 3000);
  h_jet4Pt3 = TH1F("jet4Pt3", "Jet Pt", 100, 0, 3000);
  h_jet5Pt3 = TH1F("jet5Pt3", "Jet Pt", 100, 0, 3000);
  h_jet6Pt3 = TH1F("jet6Pt3", "Jet Pt", 100, 0, 3000);
  h_jet7Pt3 = TH1F("jet7Pt3", "Jet Pt", 100, 0, 3000);

  h_jet1Pt4 = TH1F("jet1Pt4", "Jet Pt", 100, 0, 3000);
  h_jet2Pt4 = TH1F("jet2Pt4", "Jet Pt", 100, 0, 3000);
  h_jet3Pt4 = TH1F("jet3Pt4", "Jet Pt", 100, 0, 3000);
  h_jet4Pt4 = TH1F("jet4Pt4", "Jet Pt", 100, 0, 3000);
  h_jet5Pt4 = TH1F("jet5Pt4", "Jet Pt", 100, 0, 3000);
  h_jet6Pt4 = TH1F("jet6Pt4", "Jet Pt", 100, 0, 3000);
  h_jet7Pt4 = TH1F("jet7Pt4", "Jet Pt", 100, 0, 3000);

  h_totMissEt1 = TH1F("totMissEt1", "Total Unclustered Et", 100, 0, 500);
  h_totMissEt2 = TH1F("totMissEt2", "Total Unclustered Et", 100, 0, 500);
  h_totMissEt3 = TH1F("totMissEt3", "Total Unclustered Et", 100, 0, 500);

  h_missEt1 = TH1F("missEt1", "Unclustered Et", 100, 0, 50);
  h_missEt2 = TH1F("missEt2", "Unclustered Et", 100, 0, 50);
  h_missEt3 = TH1F("missEt3", "Unclustered Et", 100, 0, 50);

  h_missEt1s = TH1F("missEt1s", "Unclustered Et", 100, 0, 2);
  h_missEt2s = TH1F("missEt2s", "Unclustered Et", 100, 0, 2);
  h_missEt3s = TH1F("missEt3s", "Unclustered Et", 100, 0, 2);

  ParMatch1 = TH1F("ParMatch1", "Number of Matched Jets 1", 10, 0, 10);
  ParMatch2 = TH1F("ParMatch2", "Number of Matched Jets 2", 10, 0, 10);
  ParMatch3 = TH1F("ParMatch3", "Number of Matched Jets 3", 10, 0, 10);
}

void myFastSimVal::analyze(const Event& evt, const EventSetup& es) {
  int EtaOk10, EtaOk13, EtaOk40;

  double ZpM, ZpMG, ZpMM;
  double LeadMass1, LeadMass2, LeadMass3, LeadMass4;
  double LeadMassP1, LeadMassP2, LeadMassP3;

  float minJetPt = 30.;
  float minJetPt10 = 10.;
  int jetInd, allJetInd;
  int usedInd = -1;
  //  double matchedDelR = 0.1;
  double matchedDelR = 0.3;

  ZpMG = 0;
  LeadMass1 = -1;
  LeadMass2 = -1;
  LeadMass3 = -1;

  math::XYZTLorentzVector p4tmp[2], p4cortmp[2];
  nEvent++;

  // ********************************
  // **** Get the CaloJet1 collection
  // ********************************

  Handle<CaloJetCollection> caloJets1;
  evt.getByLabel(CaloJetAlgorithm1, caloJets1);

  // Count Jets above Pt cut
  for (int istep = 0; istep < 100; ++istep) {
    int njet = 0;
    float ptStep = (istep * (5000. / 100.));
    for (CaloJetCollection::const_iterator cal = caloJets1->begin(); cal != caloJets1->end(); ++cal) {
      if (cal->pt() > ptStep)
        njet++;
    }

    hf_nJet1.Fill(ptStep, njet);
  }

  // Count Jets above Pt cut
  for (int istep = 0; istep < 100; ++istep) {
    int njet = 0;
    float ptStep = (istep * (200. / 100.));
    for (CaloJetCollection::const_iterator cal = caloJets1->begin(); cal != caloJets1->end(); ++cal) {
      if (cal->pt() > ptStep)
        njet++;
    }

    hf_nJet1s.Fill(ptStep, njet);
  }

  // Count Jets above Pt cut
  for (int istep = 0; istep < 100; ++istep) {
    int njet = 0;
    float ptStep = (istep * (3000. / 100.));
    for (CaloJetCollection::const_iterator cal = caloJets1->begin(); cal != caloJets1->end(); ++cal) {
      if (cal->pt() > ptStep)
        njet++;
    }

    hf_nJet11.Fill(ptStep, njet);
  }

  //Loop over the two leading CaloJets and fill some histograms
  jetInd = 0;
  allJetInd = 0;
  EtaOk10 = 0;
  EtaOk13 = 0;
  EtaOk40 = 0;

  double highestPt;
  double nextPt;

  highestPt = 0.0;
  nextPt = 0.0;

  for (CaloJetCollection::const_iterator cal = caloJets1->begin(); cal != caloJets1->end(); ++cal) {
    //    double scale = corrector->correction (*cal);
    double scale = 1.0;
    double corPt = scale * cal->pt();
    //    double corPt = cal->pt();

    if (corPt > highestPt) {
      nextPt = highestPt;
      p4cortmp[1] = p4cortmp[0];
      highestPt = corPt;
      p4cortmp[0] = scale * cal->p4();
    } else if (corPt > nextPt) {
      nextPt = corPt;
      p4cortmp[1] = scale * cal->p4();
    }

    /***
    std::cout << ">>> Corr Jet: corPt = " 
	      << corPt << ", scale = " << scale 
	      << " pt = " << cal->pt()
	      << " highestPt = " << highestPt 
	      << " nextPt = "    << nextPt 
	      << std::endl;  
    ****/

    allJetInd++;
    if (allJetInd == 1) {
      h_jet1Pt1.Fill(cal->pt());
      p4tmp[0] = cal->p4();
      if (fabs(cal->eta()) < 1.0)
        EtaOk10++;
      if (fabs(cal->eta()) < 1.3)
        EtaOk13++;
      if (fabs(cal->eta()) < 4.0)
        EtaOk40++;
    }
    if (allJetInd == 2) {
      h_jet2Pt1.Fill(cal->pt());
      p4tmp[1] = cal->p4();
      if (fabs(cal->eta()) < 1.0)
        EtaOk10++;
      if (fabs(cal->eta()) < 1.3)
        EtaOk13++;
      if (fabs(cal->eta()) < 4.0)
        EtaOk40++;
    }
    if ((allJetInd == 1) || (allJetInd == 2)) {
      h_ptCalL1.Fill(cal->pt());
      h_ptCalL12.Fill(cal->pt());
      h_ptCalL13.Fill(cal->pt());

      h_etaCalL1.Fill(cal->eta());
      h_phiCalL1.Fill(cal->phi());
    }

    if (allJetInd == 3)
      h_jet3Pt1.Fill(cal->pt());
    if (allJetInd == 4)
      h_jet4Pt1.Fill(cal->pt());
    if (allJetInd == 5)
      h_jet5Pt1.Fill(cal->pt());
    if (allJetInd == 6)
      h_jet6Pt1.Fill(cal->pt());
    if (allJetInd == 7)
      h_jet7Pt1.Fill(cal->pt());

    if (fabs(cal->eta()) < 1.3) {
      h_lowPtCal11.Fill(cal->pt());
      if (cal->pt() > 10.) {
        h_lowPtCal1c11.Fill(cal->pt());
      }
    }
    if ((fabs(cal->eta()) > 1.3) && (fabs(cal->eta()) < 3.)) {
      h_lowPtCal12.Fill(cal->pt());
      if (cal->pt() > 10.) {
        h_lowPtCal1c12.Fill(cal->pt());
      }
    }
    if (fabs(cal->eta()) > 3.0) {
      h_lowPtCal13.Fill(cal->pt());
      if (cal->pt() > 10.) {
        h_lowPtCal1c13.Fill(cal->pt());
      }
    }

    if (cal->pt() > minJetPt) {
      //    std::cout << "CALO JET1 #" << jetInd << std::endl << cal->print() << std::endl;
      h_ptCal1.Fill(cal->pt());
      h_ptCal12.Fill(cal->pt());
      h_ptCal13.Fill(cal->pt());

      h_etaCal1.Fill(cal->eta());
      h_phiCal1.Fill(cal->phi());
      jetInd++;
    }
  }

  //  h_nCalJets1.Fill( caloJets1->size() );
  h_nCalJets1.Fill(jetInd);
  if (jetInd > 1) {
    LeadMass1 = (p4tmp[0] + p4tmp[1]).mass();
    dijetMass1.Fill(LeadMass1);
    dijetMass12.Fill(LeadMass1);
    dijetMass13.Fill(LeadMass1);
    if (EtaOk10 == 2) {
      dijetMass101.Fill(LeadMass1);
      dijetMass102.Fill(LeadMass1);
      dijetMass103.Fill(LeadMass1);
      dijetMass_700_101.Fill(LeadMass1);
      dijetMass_2000_101.Fill(LeadMass1);
      dijetMass_5000_101.Fill(LeadMass1);
    }
    if (EtaOk13 == 2) {
      dijetMass131.Fill(LeadMass1);
      dijetMass132.Fill(LeadMass1);
      dijetMass133.Fill(LeadMass1);
      dijetMass_700_131.Fill(LeadMass1);
      dijetMass_2000_131.Fill(LeadMass1);
      dijetMass_5000_131.Fill(LeadMass1);
    }
    if (EtaOk40 == 2) {
      dijetMass401.Fill(LeadMass1);
      dijetMass402.Fill(LeadMass1);
      dijetMass403.Fill(LeadMass1);
      dijetMass_700_401.Fill(LeadMass1);
      dijetMass_2000_401.Fill(LeadMass1);
      dijetMass_5000_401.Fill(LeadMass1);
    }

    LeadMass1 = (p4cortmp[0] + p4cortmp[1]).mass();

    /****
    if (LeadMass1 < 30.) {
      std::cout << " XXX Low Mass " 
		<< (p4tmp[0]+p4tmp[1]).mass() 
		<< " / " 
		<< (p4cortmp[0]+p4cortmp[1]).mass() 
		<< std::endl;

      std::cout << " p4 1 = " << p4tmp[0]
		<< " p4 2 = " << p4tmp[1]
		<< " p4 cor 1 = " << p4cortmp[0]
		<< " p4 cor 2 = " << p4cortmp[0]
		<< endl;

    }
    ****/

    /****
    dijetMassCor1.Fill( LeadMass1 );    
    dijetMassCor_700_1.Fill( LeadMass1 );    
    dijetMassCor_2000_1.Fill( LeadMass1 );    
    dijetMassCor_5000_1.Fill( LeadMass1 );    

    if (EtaOk10 == 2) {
      dijetMassCor101.Fill( LeadMass1 );  
      dijetMassCor_700_101.Fill( LeadMass1 );  
      dijetMassCor_2000_101.Fill( LeadMass1 );  
      dijetMassCor_5000_101.Fill( LeadMass1 );  
    }
    if (EtaOk13 == 2) {
      dijetMassCor131.Fill( LeadMass1 );  
      dijetMassCor_700_131.Fill( LeadMass1 );  
      dijetMassCor_2000_131.Fill( LeadMass1 );  
      dijetMassCor_5000_131.Fill( LeadMass1 );  
    }
    if (EtaOk40 == 2) {
      dijetMassCor401.Fill( LeadMass1 ); 
      dijetMassCor_700_401.Fill( LeadMass1 ); 
      dijetMassCor_2000_401.Fill( LeadMass1 ); 
      dijetMassCor_5000_401.Fill( LeadMass1 ); 
    }
    ****/
  }

  // ********************************
  // **** Get the CaloJet2 collection
  // ********************************
  Handle<CaloJetCollection> caloJets2;
  evt.getByLabel(CaloJetAlgorithm2, caloJets2);

  // Count Jets above Pt cut
  for (int istep = 0; istep < 100; ++istep) {
    int njet = 0;
    float ptStep = (istep * (5000. / 100.));

    for (CaloJetCollection::const_iterator cal = caloJets2->begin(); cal != caloJets2->end(); ++cal)
      if (cal->pt() > ptStep)
        njet++;

    hf_nJet2.Fill(ptStep, njet);
  }

  for (int istep = 0; istep < 100; ++istep) {
    int njet = 0;
    float ptStep = (istep * (200. / 100.));

    for (CaloJetCollection::const_iterator cal = caloJets2->begin(); cal != caloJets2->end(); ++cal)
      if (cal->pt() > ptStep)
        njet++;

    hf_nJet2s.Fill(ptStep, njet);
  }

  for (int istep = 0; istep < 100; ++istep) {
    int njet = 0;
    float ptStep = (istep * (3000. / 100.));

    for (CaloJetCollection::const_iterator cal = caloJets2->begin(); cal != caloJets2->end(); ++cal)
      if (cal->pt() > ptStep)
        njet++;

    hf_nJet21.Fill(ptStep, njet);
  }

  //Loop over the two leading CaloJets and fill some histograms
  jetInd = 0;
  allJetInd = 0;
  for (CaloJetCollection::const_iterator cal = caloJets2->begin(); cal != caloJets2->end(); ++cal) {
    allJetInd++;
    if (allJetInd == 1) {
      h_jet1Pt2.Fill(cal->pt());
      p4tmp[0] = cal->p4();
    }
    if (allJetInd == 2) {
      h_jet2Pt2.Fill(cal->pt());
      p4tmp[1] = cal->p4();
    }
    if ((allJetInd == 1) || (allJetInd == 2)) {
      h_ptCalL2.Fill(cal->pt());
      h_ptCalL22.Fill(cal->pt());
      h_ptCalL23.Fill(cal->pt());

      h_etaCalL2.Fill(cal->eta());
      h_phiCalL2.Fill(cal->phi());
    }
    if (allJetInd == 3)
      h_jet3Pt2.Fill(cal->pt());
    if (allJetInd == 4)
      h_jet4Pt2.Fill(cal->pt());
    if (allJetInd == 5)
      h_jet5Pt2.Fill(cal->pt());
    if (allJetInd == 6)
      h_jet6Pt2.Fill(cal->pt());
    if (allJetInd == 7)
      h_jet7Pt2.Fill(cal->pt());

    if (fabs(cal->eta()) < 1.3) {
      h_lowPtCal21.Fill(cal->pt());
      if (cal->pt() > 10.) {
        h_lowPtCal2c11.Fill(cal->pt());
      }
    }
    if ((fabs(cal->eta()) > 1.3) && (fabs(cal->eta()) < 3.)) {
      h_lowPtCal22.Fill(cal->pt());
      if (cal->pt() > 10.) {
        h_lowPtCal2c12.Fill(cal->pt());
      }
    }
    if (fabs(cal->eta()) > 3.0) {
      h_lowPtCal23.Fill(cal->pt());
      if (cal->pt() > 10.) {
        h_lowPtCal2c13.Fill(cal->pt());
      }
    }

    if (cal->pt() > minJetPt) {
      h_ptCal2.Fill(cal->pt());
      h_ptCal22.Fill(cal->pt());
      h_ptCal23.Fill(cal->pt());

      h_etaCal2.Fill(cal->eta());
      h_phiCal2.Fill(cal->phi());
      jetInd++;
    }
  }
  //  h_nCalJets2.Fill( caloJets2->size() );
  h_nCalJets2.Fill(jetInd);
  if (jetInd > 1) {
    LeadMass2 = (p4tmp[0] + p4tmp[1]).mass();
    dijetMass2.Fill(LeadMass2);
    dijetMass22.Fill(LeadMass2);
    dijetMass23.Fill(LeadMass2);

    dijetMassCor1.Fill(LeadMass2);
    dijetMassCor_700_1.Fill(LeadMass2);
    dijetMassCor_2000_1.Fill(LeadMass2);
    dijetMassCor_5000_1.Fill(LeadMass2);

    if (EtaOk10 == 2) {
      dijetMassCor101.Fill(LeadMass2);
      dijetMassCor_700_101.Fill(LeadMass2);
      dijetMassCor_2000_101.Fill(LeadMass2);
      dijetMassCor_5000_101.Fill(LeadMass2);
    }
    if (EtaOk13 == 2) {
      dijetMassCor131.Fill(LeadMass2);
      dijetMassCor_700_131.Fill(LeadMass2);
      dijetMassCor_2000_131.Fill(LeadMass2);
      dijetMassCor_5000_131.Fill(LeadMass2);
    }
    if (EtaOk40 == 2) {
      dijetMassCor401.Fill(LeadMass2);
      dijetMassCor_700_401.Fill(LeadMass2);
      dijetMassCor_2000_401.Fill(LeadMass2);
      dijetMassCor_5000_401.Fill(LeadMass2);
    }
  }

  // ********************************
  // **** Get the CaloJet3 collection
  // ********************************
  Handle<CaloJetCollection> caloJets3;
  evt.getByLabel(CaloJetAlgorithm3, caloJets3);

  //Loop over the two leading CaloJets and fill some histograms
  jetInd = 0;
  allJetInd = 0;

  // Count Jets above Pt cut
  for (int istep = 0; istep < 100; ++istep) {
    int njet = 0;
    float ptStep = (istep * (5000. / 100.));

    for (CaloJetCollection::const_iterator cal = caloJets3->begin(); cal != caloJets3->end(); ++cal)
      if (cal->pt() > ptStep)
        njet++;

    hf_nJet3.Fill(ptStep, njet);
  }

  for (int istep = 0; istep < 100; ++istep) {
    int njet = 0;
    float ptStep = (istep * (200. / 100.));

    for (CaloJetCollection::const_iterator cal = caloJets3->begin(); cal != caloJets3->end(); ++cal)
      if (cal->pt() > ptStep)
        njet++;

    hf_nJet3s.Fill(ptStep, njet);
  }

  for (int istep = 0; istep < 100; ++istep) {
    int njet = 0;
    float ptStep = (istep * (3000. / 100.));

    for (CaloJetCollection::const_iterator cal = caloJets3->begin(); cal != caloJets3->end(); ++cal)
      if (cal->pt() > ptStep)
        njet++;

    hf_nJet31.Fill(ptStep, njet);
  }

  for (CaloJetCollection::const_iterator cal = caloJets3->begin(); cal != caloJets3->end(); ++cal) {
    allJetInd++;
    if (allJetInd == 1) {
      h_jet1Pt3.Fill(cal->pt());
      p4tmp[0] = cal->p4();
    }
    if (allJetInd == 2) {
      h_jet2Pt3.Fill(cal->pt());
      p4tmp[1] = cal->p4();
    }
    if ((allJetInd == 1) || (allJetInd == 2)) {
      h_ptCalL3.Fill(cal->pt());
      h_ptCalL32.Fill(cal->pt());
      h_ptCalL33.Fill(cal->pt());

      h_etaCalL3.Fill(cal->eta());
      h_phiCalL3.Fill(cal->phi());
    }
    if (allJetInd == 3)
      h_jet3Pt3.Fill(cal->pt());
    if (allJetInd == 4)
      h_jet4Pt3.Fill(cal->pt());
    if (allJetInd == 5)
      h_jet5Pt3.Fill(cal->pt());
    if (allJetInd == 6)
      h_jet6Pt3.Fill(cal->pt());
    if (allJetInd == 7)
      h_jet7Pt3.Fill(cal->pt());

    if (fabs(cal->eta()) < 1.3) {
      h_lowPtCal31.Fill(cal->pt());
      if (cal->pt() > 10.) {
        h_lowPtCal3c11.Fill(cal->pt());
      }
    }
    if ((fabs(cal->eta()) > 1.3) && (fabs(cal->eta()) < 3.)) {
      h_lowPtCal32.Fill(cal->pt());
      if (cal->pt() > 10.) {
        h_lowPtCal3c12.Fill(cal->pt());
      }
    }
    if (fabs(cal->eta()) > 3.0) {
      h_lowPtCal33.Fill(cal->pt());
      if (cal->pt() > 10.) {
        h_lowPtCal3c13.Fill(cal->pt());
      }
    }

    if (cal->pt() > minJetPt) {
      //    std::cout << "CALO JET3 #" << jetInd << std::endl << cal->print() << std::endl;
      h_ptCal3.Fill(cal->pt());
      h_ptCal32.Fill(cal->pt());
      h_ptCal33.Fill(cal->pt());

      h_etaCal3.Fill(cal->eta());
      h_phiCal3.Fill(cal->phi());
      jetInd++;
    }
  }
  //  h_nCalJets3.Fill( caloJets3->size() );
  h_nCalJets3.Fill(jetInd);
  if (jetInd > 1) {
    LeadMass3 = (p4tmp[0] + p4tmp[1]).mass();
    dijetMass3.Fill(LeadMass3);
    dijetMass32.Fill(LeadMass3);
    dijetMass33.Fill(LeadMass3);
  }

  // ********************************
  // **** Get the CaloJet4 collection
  // ********************************

  Handle<CaloJetCollection> caloJets4;
  evt.getByLabel(CaloJetAlgorithm4, caloJets4);

  //Loop over the two leading CaloJets and fill some histograms
  jetInd = 0;
  allJetInd = 0;

  // Count Jets above Pt cut
  for (int istep = 0; istep < 100; ++istep) {
    int njet = 0;
    float ptStep = (istep * (5000. / 100.));

    for (CaloJetCollection::const_iterator cal = caloJets4->begin(); cal != caloJets4->end(); ++cal)
      if (cal->pt() > ptStep)
        njet++;

    hf_nJet4.Fill(ptStep, njet);
  }

  for (int istep = 0; istep < 100; ++istep) {
    int njet = 0;
    float ptStep = (istep * (200. / 100.));

    for (CaloJetCollection::const_iterator cal = caloJets4->begin(); cal != caloJets4->end(); ++cal)
      if (cal->pt() > ptStep)
        njet++;

    hf_nJet4s.Fill(ptStep, njet);
  }

  for (int istep = 0; istep < 100; ++istep) {
    int njet = 0;
    float ptStep = (istep * (3000. / 100.));

    for (CaloJetCollection::const_iterator cal = caloJets4->begin(); cal != caloJets4->end(); ++cal)
      if (cal->pt() > ptStep)
        njet++;

    hf_nJet41.Fill(ptStep, njet);
  }

  for (CaloJetCollection::const_iterator cal = caloJets4->begin(); cal != caloJets4->end(); ++cal) {
    allJetInd++;
    if (allJetInd == 1) {
      h_jet1Pt4.Fill(cal->pt());
      p4tmp[0] = cal->p4();
    }
    if (allJetInd == 2) {
      h_jet2Pt4.Fill(cal->pt());
      p4tmp[1] = cal->p4();
    }
    if ((allJetInd == 1) || (allJetInd == 2)) {
      h_ptCalL4.Fill(cal->pt());
      h_ptCalL42.Fill(cal->pt());
      h_ptCalL43.Fill(cal->pt());

      h_etaCalL4.Fill(cal->eta());
      h_phiCalL4.Fill(cal->phi());
    }
    if (allJetInd == 3)
      h_jet3Pt4.Fill(cal->pt());
    if (allJetInd == 4)
      h_jet4Pt4.Fill(cal->pt());
    if (allJetInd == 5)
      h_jet5Pt4.Fill(cal->pt());
    if (allJetInd == 6)
      h_jet6Pt4.Fill(cal->pt());
    if (allJetInd == 7)
      h_jet7Pt4.Fill(cal->pt());

    if (fabs(cal->eta()) < 1.3) {
      h_lowPtCal41.Fill(cal->pt());
      if (cal->pt() > 10.) {
        h_lowPtCal4c11.Fill(cal->pt());
      }
    }
    if ((fabs(cal->eta()) > 1.3) && (fabs(cal->eta()) < 3.)) {
      h_lowPtCal42.Fill(cal->pt());
      if (cal->pt() > 10.) {
        h_lowPtCal4c12.Fill(cal->pt());
      }
    }
    if (fabs(cal->eta()) > 3.0) {
      h_lowPtCal43.Fill(cal->pt());
      if (cal->pt() > 10.) {
        h_lowPtCal4c13.Fill(cal->pt());
      }
    }

    if (cal->pt() > minJetPt) {
      //    std::cout << "CALO JET4 #" << jetInd << std::endl << cal->print() << std::endl;
      h_ptCal4.Fill(cal->pt());
      h_ptCal42.Fill(cal->pt());
      h_ptCal43.Fill(cal->pt());

      h_etaCal4.Fill(cal->eta());
      h_phiCal4.Fill(cal->phi());
      jetInd++;
    }
  }
  //  h_nCalJets4.Fill( caloJets4->size() );
  h_nCalJets4.Fill(jetInd);
  if (jetInd > 1) {
    LeadMass4 = (p4tmp[0] + p4tmp[1]).mass();
    dijetMass4.Fill(LeadMass4);
    dijetMass42.Fill(LeadMass4);
    dijetMass43.Fill(LeadMass4);
  }

  // *********************************************
  // *********************************************

  //**** Get the GenJet1 collection
  Handle<GenJetCollection> genJets1;
  evt.getByLabel(GenJetAlgorithm1, genJets1);

  //Loop over the two leading GenJets and fill some histograms
  jetInd = 0;
  allJetInd = 0;
  for (GenJetCollection::const_iterator gen = genJets1->begin(); gen != genJets1->end(); ++gen) {
    allJetInd++;
    if (allJetInd == 1) {
      p4tmp[0] = gen->p4();
    }
    if (allJetInd == 2) {
      p4tmp[1] = gen->p4();
    }

    if ((allJetInd == 1) || (allJetInd == 2)) {
      h_ptGenL1.Fill(gen->pt());
      h_ptGenL12.Fill(gen->pt());
      h_ptGenL13.Fill(gen->pt());

      h_etaGenL1.Fill(gen->eta());
      h_phiGenL1.Fill(gen->phi());
    }

    if (gen->pt() > minJetPt) {
      // std::cout << "GEN JET1 #" << jetInd << std::endl << gen->print() << std::endl;
      h_ptGen1.Fill(gen->pt());
      h_ptGen12.Fill(gen->pt());
      h_ptGen13.Fill(gen->pt());

      h_etaGen1.Fill(gen->eta());
      h_phiGen1.Fill(gen->phi());
      jetInd++;
    }
  }

  LeadMassP1 = (p4tmp[0] + p4tmp[1]).mass();
  dijetMassP1.Fill(LeadMassP1);
  if (EtaOk10 == 2) {
    dijetMassP101.Fill(LeadMassP1);
    dijetMassP_700_101.Fill(LeadMassP1);
    dijetMassP_2000_101.Fill(LeadMassP1);
    dijetMassP_5000_101.Fill(LeadMassP1);
  }
  if (EtaOk13 == 2) {
    dijetMassP131.Fill(LeadMassP1);
    dijetMassP_700_131.Fill(LeadMassP1);
    dijetMassP_2000_131.Fill(LeadMassP1);
    dijetMassP_5000_131.Fill(LeadMassP1);
  }
  if (EtaOk40 == 2) {
    dijetMassP401.Fill(LeadMassP1);
    dijetMassP_5000_401.Fill(LeadMassP1);
    dijetMassP_5000_401.Fill(LeadMassP1);
    dijetMassP_5000_401.Fill(LeadMassP1);
  }

  //  h_nGenJets1.Fill( genJets1->size() );
  h_nGenJets1.Fill(jetInd);

  //**** Get the GenJet2 collection
  Handle<GenJetCollection> genJets2;
  evt.getByLabel(GenJetAlgorithm2, genJets2);

  //Loop over the two leading GenJets and fill some histograms
  jetInd = 0;
  allJetInd = 0;
  for (GenJetCollection::const_iterator gen = genJets2->begin(); gen != genJets2->end(); ++gen) {
    allJetInd++;
    if (allJetInd == 1) {
      p4tmp[0] = gen->p4();
    }
    if (allJetInd == 2) {
      p4tmp[1] = gen->p4();
    }
    if ((allJetInd == 1) || (allJetInd == 2)) {
      h_ptGenL2.Fill(gen->pt());
      h_ptGenL22.Fill(gen->pt());
      h_ptGenL23.Fill(gen->pt());

      h_etaGenL2.Fill(gen->eta());
      h_phiGenL2.Fill(gen->phi());
    }

    if (gen->pt() > minJetPt) {
      // std::cout << "GEN JET2 #" << jetInd << std::endl << gen->print() << std::endl;
      h_ptGen2.Fill(gen->pt());
      h_ptGen22.Fill(gen->pt());
      h_ptGen23.Fill(gen->pt());

      h_etaGen2.Fill(gen->eta());
      h_phiGen2.Fill(gen->phi());
      jetInd++;
    }
  }

  LeadMassP2 = (p4tmp[0] + p4tmp[1]).mass();
  dijetMassP2.Fill(LeadMassP2);

  //  h_nGenJets2.Fill( genJets2->size() );
  h_nGenJets2.Fill(jetInd);

  //**** Get the GenJet3 collection
  Handle<GenJetCollection> genJets3;
  evt.getByLabel(GenJetAlgorithm3, genJets3);

  //Loop over the two leading GenJets and fill some histograms
  jetInd = 0;
  allJetInd = 0;
  for (GenJetCollection::const_iterator gen = genJets3->begin(); gen != genJets3->end(); ++gen) {
    allJetInd++;
    if (allJetInd == 1) {
      p4tmp[0] = gen->p4();
    }
    if (allJetInd == 2) {
      p4tmp[1] = gen->p4();
    }
    if ((allJetInd == 1) || (allJetInd == 2)) {
      h_ptGenL3.Fill(gen->pt());
      h_ptGenL32.Fill(gen->pt());
      h_ptGenL33.Fill(gen->pt());

      h_etaGenL3.Fill(gen->eta());
      h_phiGenL3.Fill(gen->phi());
    }

    if (gen->pt() > minJetPt) {
      // std::cout << "GEN JET3 #" << jetInd << std::endl << gen->print() << std::endl;
      h_ptGen3.Fill(gen->pt());
      h_ptGen32.Fill(gen->pt());
      h_ptGen33.Fill(gen->pt());

      h_etaGen3.Fill(gen->eta());
      h_phiGen3.Fill(gen->phi());
      jetInd++;
    }
  }

  LeadMassP3 = (p4tmp[0] + p4tmp[1]).mass();
  dijetMassP3.Fill(LeadMassP3);

  //  h_nGenJets3.Fill( genJets3->size() );
  h_nGenJets3.Fill(jetInd);

  // *********************
  // MidPoint Jet Matching

  Handle<GenJetCollection> genJets;
  Handle<CaloJetCollection> caloJets;

  //  evt.getByLabel( "midPointCone5GenJets",  genJets );
  //  evt.getByLabel( "midPointCone5CaloJets", caloJets );
  evt.getByLabel(GenJetAlgorithm1, genJets);
  evt.getByLabel(CaloJetAlgorithm1, caloJets);

  int maxJets = MAXJETS;

  jetInd = 0;
  double dRmin[MAXJETS];
  math::XYZTLorentzVector p4gen[MAXJETS], p4cal[MAXJETS], p4par[MAXJETS], p4Zp[MAXJETS], p4part[MAXJETS];

  int used[MAXJETS];
  int nj;

  for (int i = 0; i < maxJets; ++i)
    used[i] = 0;

  //  cout << ">>>>>>>>> " << endl;

  for (GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd < maxJets; ++gen) {
    p4gen[jetInd] = gen->p4();  //Gen 4-vector
    dRmin[jetInd] = 1000.0;

    nj = 0;
    usedInd = -1;

    for (CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end(); ++cal) {
      double delR = deltaR(cal->eta(), cal->phi(), gen->eta(), gen->phi());

      if ((delR < dRmin[jetInd]) && (delR < matchedDelR) && (used[nj] == 0)) {
        dRmin[jetInd] = delR;       // delta R of match
        p4cal[jetInd] = cal->p4();  // Matched Cal 4-vector
        usedInd = nj;
      }

      nj++;
    }

    if (fabs(p4gen[jetInd].eta()) < 1.3) {
      matchedAllPt11.Fill(p4gen[jetInd].pt());
    }
    if ((fabs(p4gen[jetInd].eta()) > 1.3) && (fabs(p4gen[jetInd].eta()) < 3.)) {
      matchedAllPt12.Fill(p4gen[jetInd].pt());
    }
    if (fabs(p4gen[jetInd].eta()) > 3.) {
      matchedAllPt13.Fill(p4gen[jetInd].pt());
    }

    if (usedInd != -1) {
      used[usedInd] = 1;

      if (p4cal[jetInd].pt() > minJetPt10)
        hf_PtResponse1.Fill(p4cal[jetInd].eta(), p4cal[jetInd].pt() / p4gen[jetInd].pt());

      if (fabs(p4gen[jetInd].eta()) < 1.3) {
        matchedPt11.Fill(p4gen[jetInd].pt());
      }
      if ((fabs(p4gen[jetInd].eta()) > 1.3) && (fabs(p4gen[jetInd].eta()) < 3.)) {
        matchedPt12.Fill(p4gen[jetInd].pt());
      }
      if (fabs(p4gen[jetInd].eta()) > 3.) {
        matchedPt13.Fill(p4gen[jetInd].pt());
      }
      /***
      cout << " >>>SISCone "     
	   << jetInd                 << " " 
	   << p4gen[jetInd].pt()     << " " 
	   << p4gen[jetInd].eta()    << " " 
	   << p4gen[jetInd].phi()    << " " 
	   << p4cal[jetInd].pt()     << " " 
	   << p4cal[jetInd].eta()    << " " 
	   << p4cal[jetInd].phi()    
	   << endl;
      ***/

      double dpt = p4cal[jetInd].pt() - p4gen[jetInd].pt();
      if (fabs(p4gen[jetInd].eta()) < 1.3) {
        if ((p4gen[jetInd].pt() > 20) && (p4gen[jetInd].pt() < 40)) {
          dPt20Frac1.Fill(dpt / p4gen[jetInd].pt());
        }
        if ((p4gen[jetInd].pt() > 40) && (p4gen[jetInd].pt() < 60)) {
          dPt40Frac1.Fill(dpt / p4gen[jetInd].pt());
        }
        if ((p4gen[jetInd].pt() > 80) && (p4gen[jetInd].pt() < 100)) {
          dPt80Frac1.Fill(dpt / p4gen[jetInd].pt());
        }
        if ((p4gen[jetInd].pt() > 100) && (p4gen[jetInd].pt() < 120)) {
          dPt100Frac1.Fill(dpt / p4gen[jetInd].pt());
        }
      }
      if ((p4gen[jetInd].pt() > minJetPt10) && (fabs(p4gen[jetInd].eta()) < 1.3)) {
        dR1.Fill(dRmin[jetInd]);
        double dphi = deltaPhi(p4cal[jetInd].phi(), p4gen[jetInd].phi());
        dPhi1.Fill(dphi);
        double deta = p4cal[jetInd].eta() - p4gen[jetInd].eta();
        dEta1.Fill(deta);
        dPt1.Fill(dpt);
        dPtFrac1.Fill(dpt / p4gen[jetInd].pt());

        if (((dpt / p4gen[jetInd].pt()) < -0.5) && (fabs(dpt) > 90.)) {
          cout << " deltaR min = " << dRmin[jetInd] << " Ind = " << jetInd << " / " << usedInd << " / " << used[nj]
               << " Del pt / frac  = " << dpt << " / " << dpt / p4gen[jetInd].pt()
               << " cal/gen pt   = " << p4cal[jetInd].pt() << " / " << p4gen[jetInd].pt()
               << " cal/gen eta  = " << p4cal[jetInd].eta() << " / " << p4gen[jetInd].eta()
               << " cal/gen phi  = " << p4cal[jetInd].phi() << " / " << p4gen[jetInd].phi() << endl;
        }
      }

      jetInd++;
    }
  }

  // *********************
  // Seedless Jet Matching

  //  Handle<GenJetCollection>  genJets;
  //  Handle<CaloJetCollection> caloJets;

  //  evt.getByLabel( "sisCone5GenJets",  genJets );
  //  evt.getByLabel( "sisCone5CaloJets", caloJets );
  evt.getByLabel(GenJetAlgorithm2, genJets);
  evt.getByLabel(CaloJetAlgorithm2, caloJets);

  // int maxJets = 20;
  jetInd = 0;
  //  double dRmin[20];
  //  math::XYZTLorentzVector p4jet[20], p4gen[20], p4cal[20], p4cor[20];

  for (int i = 0; i < maxJets; ++i)
    used[i] = 0;
  for (GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd < maxJets; ++gen) {
    p4gen[jetInd] = gen->p4();  //Gen 4-vector
    dRmin[jetInd] = 1000.0;

    nj = 0;
    usedInd = -1;

    for (CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end(); ++cal) {
      double delR = deltaR(cal->eta(), cal->phi(), gen->eta(), gen->phi());

      if ((delR < dRmin[jetInd]) && (delR < matchedDelR) && (used[nj] == 0)) {
        dRmin[jetInd] = delR;       // delta R of match
        p4cal[jetInd] = cal->p4();  // Matched Cal 4-vector
        usedInd = nj;
      }
      nj++;
    }

    if (fabs(p4gen[jetInd].eta()) < 1.3) {
      matchedAllPt21.Fill(p4gen[jetInd].pt());
    }
    if ((fabs(p4gen[jetInd].eta()) > 1.3) && (fabs(p4gen[jetInd].eta()) < 3.)) {
      matchedAllPt22.Fill(p4gen[jetInd].pt());
    }
    if (fabs(p4gen[jetInd].eta()) > 3.) {
      matchedAllPt23.Fill(p4gen[jetInd].pt());
    }

    if (usedInd != -1) {
      used[usedInd] = 1;

      if (p4cal[jetInd].pt() > minJetPt10)
        hf_PtResponse2.Fill(p4cal[jetInd].eta(), p4cal[jetInd].pt() / p4gen[jetInd].pt());

      if (fabs(p4gen[jetInd].eta()) < 1.3) {
        matchedPt21.Fill(p4gen[jetInd].pt());
      }
      if ((fabs(p4gen[jetInd].eta()) > 1.3) && (fabs(p4gen[jetInd].eta()) < 3.)) {
        matchedPt22.Fill(p4gen[jetInd].pt());
      }
      if (fabs(p4gen[jetInd].eta()) > 3.) {
        matchedPt23.Fill(p4gen[jetInd].pt());
      }
      /***
      cout << " >>>IterCone "     
	   << jetInd                 << " " 
	   << p4gen[jetInd].pt()     << " " 
	   << p4gen[jetInd].eta()    << " " 
	   << p4gen[jetInd].phi()    << " " 
	   << p4cal[jetInd].pt()     << " " 
	   << p4cal[jetInd].eta()    << " " 
	   << p4cal[jetInd].phi()    
	   << endl;
      ***/

      double dpt = p4cal[jetInd].pt() - p4gen[jetInd].pt();
      if (fabs(p4gen[jetInd].eta()) < 1.3) {
        if ((p4gen[jetInd].pt() > 20) && (p4gen[jetInd].pt() < 40)) {
          dPt20Frac2.Fill(dpt / p4gen[jetInd].pt());
        }
        if ((p4gen[jetInd].pt() > 40) && (p4gen[jetInd].pt() < 60)) {
          dPt40Frac2.Fill(dpt / p4gen[jetInd].pt());
        }
        if ((p4gen[jetInd].pt() > 80) && (p4gen[jetInd].pt() < 100)) {
          dPt80Frac2.Fill(dpt / p4gen[jetInd].pt());
        }
        if ((p4gen[jetInd].pt() > 100) && (p4gen[jetInd].pt() < 120)) {
          dPt100Frac2.Fill(dpt / p4gen[jetInd].pt());
        }
      }
      if ((p4gen[jetInd].pt() > minJetPt10) && (fabs(p4gen[jetInd].eta()) < 1.3)) {
        dR2.Fill(dRmin[jetInd]);
        double dphi = deltaPhi(p4cal[jetInd].phi(), p4gen[jetInd].phi());
        dPhi2.Fill(dphi);
        double deta = p4cal[jetInd].eta() - p4gen[jetInd].eta();
        dEta2.Fill(deta);
        dPt2.Fill(dpt);
        dPtFrac2.Fill(dpt / p4gen[jetInd].pt());
      }

      jetInd++;
    }
  }

  // *********************
  // Kt Jet Matching

  //  Handle<GenJetCollection>  genJets;
  //  Handle<CaloJetCollection> caloJets;

  //  evt.getByLabel( "sisCone5GenJets",  genJets );
  //  evt.getByLabel( "sisCone5CaloJets", caloJets );
  evt.getByLabel(GenJetAlgorithm3, genJets);
  evt.getByLabel(CaloJetAlgorithm3, caloJets);

  // int maxJets = 20;
  jetInd = 0;
  //  double dRmin[20];
  //  math::XYZTLorentzVector p4jet[20], p4gen[20], p4cal[20], p4cor[20];

  for (int i = 0; i < maxJets; ++i)
    used[i] = 0;
  for (GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd < maxJets; ++gen) {
    p4gen[jetInd] = gen->p4();  //Gen 4-vector
    dRmin[jetInd] = 1000.0;

    nj = 0;
    usedInd = -1;

    for (CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end(); ++cal) {
      double delR = deltaR(cal->eta(), cal->phi(), gen->eta(), gen->phi());

      if ((delR < dRmin[jetInd]) && (delR < matchedDelR) && (used[nj] == 0)) {
        dRmin[jetInd] = delR;       // delta R of match
        p4cal[jetInd] = cal->p4();  // Matched Cal 4-vector
        usedInd = nj;
      }
      nj++;
    }

    if (fabs(p4gen[jetInd].eta()) < 1.3) {
      matchedAllPt31.Fill(p4gen[jetInd].pt());
    }
    if ((fabs(p4gen[jetInd].eta()) > 1.3) && (fabs(p4gen[jetInd].eta()) < 3.)) {
      matchedAllPt32.Fill(p4gen[jetInd].pt());
    }
    if (fabs(p4gen[jetInd].eta()) > 3.) {
      matchedAllPt33.Fill(p4gen[jetInd].pt());
    }

    if (usedInd != -1) {
      used[usedInd] = 1;

      if (p4cal[jetInd].pt() > minJetPt10)
        hf_PtResponse3.Fill(p4cal[jetInd].eta(), p4cal[jetInd].pt() / p4gen[jetInd].pt());

      if (fabs(p4gen[jetInd].eta()) < 1.3) {
        matchedPt31.Fill(p4gen[jetInd].pt());
      }
      if ((fabs(p4gen[jetInd].eta()) > 1.3) && (fabs(p4gen[jetInd].eta()) < 3.)) {
        matchedPt32.Fill(p4gen[jetInd].pt());
      }
      if (fabs(p4gen[jetInd].eta()) > 3.) {
        matchedPt33.Fill(p4gen[jetInd].pt());
      }
      /***
      cout << " >>>MidPoint "     
	   << jetInd                 << " " 
	   << p4gen[jetInd].pt()     << " " 
	   << p4gen[jetInd].eta()    << " " 
	   << p4gen[jetInd].phi()    << " " 
	   << p4cal[jetInd].pt()     << " " 
	   << p4cal[jetInd].eta()    << " " 
	   << p4cal[jetInd].phi()    
	   << endl;
      ***/

      double dpt = p4cal[jetInd].pt() - p4gen[jetInd].pt();
      if (fabs(p4gen[jetInd].eta()) < 1.3) {
        if ((p4gen[jetInd].pt() > 20) && (p4gen[jetInd].pt() < 40)) {
          dPt20Frac3.Fill(dpt / p4gen[jetInd].pt());
        }
        if ((p4gen[jetInd].pt() > 40) && (p4gen[jetInd].pt() < 60)) {
          dPt40Frac3.Fill(dpt / p4gen[jetInd].pt());
        }
        if ((p4gen[jetInd].pt() > 80) && (p4gen[jetInd].pt() < 100)) {
          dPt80Frac3.Fill(dpt / p4gen[jetInd].pt());
        }
        if ((p4gen[jetInd].pt() > 100) && (p4gen[jetInd].pt() < 120)) {
          dPt100Frac3.Fill(dpt / p4gen[jetInd].pt());
        }
      }

      if ((p4gen[jetInd].pt() > minJetPt10) && (fabs(p4gen[jetInd].eta()) < 1.3)) {
        dR3.Fill(dRmin[jetInd]);
        double dphi = deltaPhi(p4cal[jetInd].phi(), p4gen[jetInd].phi());
        dPhi3.Fill(dphi);
        double deta = p4cal[jetInd].eta() - p4gen[jetInd].eta();
        dEta3.Fill(deta);
        dPt3.Fill(dpt);
        dPtFrac3.Fill(dpt / p4gen[jetInd].pt());
      }

      jetInd++;
    }
  }

  // *********************
  // Jet Algorithm 4

  evt.getByLabel(GenJetAlgorithm4, genJets);
  evt.getByLabel(CaloJetAlgorithm4, caloJets);

  // int maxJets = 20;
  jetInd = 0;
  //  double dRmin[20];
  //  math::XYZTLorentzVector p4jet[20], p4gen[20], p4cal[20], p4cor[20];

  for (int i = 0; i < maxJets; ++i)
    used[i] = 0;
  for (GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() && jetInd < maxJets; ++gen) {
    p4gen[jetInd] = gen->p4();  //Gen 4-vector
    dRmin[jetInd] = 1000.0;

    nj = 0;
    usedInd = -1;

    for (CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end(); ++cal) {
      double delR = deltaR(cal->eta(), cal->phi(), gen->eta(), gen->phi());

      if ((delR < dRmin[jetInd]) && (delR < matchedDelR) && (used[nj] == 0)) {
        dRmin[jetInd] = delR;       // delta R of match
        p4cal[jetInd] = cal->p4();  // Matched Cal 4-vector
        usedInd = nj;
      }
      nj++;
    }

    if (fabs(p4gen[jetInd].eta()) < 1.3) {
      matchedAllPt41.Fill(p4gen[jetInd].pt());
    }
    if ((fabs(p4gen[jetInd].eta()) > 1.3) && (fabs(p4gen[jetInd].eta()) < 3.)) {
      matchedAllPt42.Fill(p4gen[jetInd].pt());
    }
    if (fabs(p4gen[jetInd].eta()) > 3.) {
      matchedAllPt43.Fill(p4gen[jetInd].pt());
    }

    if (usedInd != -1) {
      used[usedInd] = 1;

      if (p4cal[jetInd].pt() > minJetPt10)
        hf_PtResponse4.Fill(p4cal[jetInd].eta(), p4cal[jetInd].pt() / p4gen[jetInd].pt());

      if (fabs(p4gen[jetInd].eta()) < 1.3) {
        matchedPt41.Fill(p4gen[jetInd].pt());
      }
      if ((fabs(p4gen[jetInd].eta()) > 1.3) && (fabs(p4gen[jetInd].eta()) < 3.)) {
        matchedPt42.Fill(p4gen[jetInd].pt());
      }
      if (fabs(p4gen[jetInd].eta()) > 3.) {
        matchedPt43.Fill(p4gen[jetInd].pt());
      }

      /***
      cout << " >>>KtClus "     
	   << jetInd                 << " " 
	   << p4gen[jetInd].pt()     << " " 
	   << p4gen[jetInd].eta()    << " " 
	   << p4gen[jetInd].phi()    << " " 
	   << p4cal[jetInd].pt()     << " " 
	   << p4cal[jetInd].eta()    << " " 
	   << p4cal[jetInd].phi()    
	   << endl;
      ***/

      double dpt = p4cal[jetInd].pt() - p4gen[jetInd].pt();
      if (fabs(p4gen[jetInd].eta()) < 1.3) {
        if ((p4gen[jetInd].pt() > 20) && (p4gen[jetInd].pt() < 40)) {
          dPt20Frac4.Fill(dpt / p4gen[jetInd].pt());
        }
        if ((p4gen[jetInd].pt() > 40) && (p4gen[jetInd].pt() < 60)) {
          dPt40Frac4.Fill(dpt / p4gen[jetInd].pt());
        }
        if ((p4gen[jetInd].pt() > 80) && (p4gen[jetInd].pt() < 100)) {
          dPt80Frac4.Fill(dpt / p4gen[jetInd].pt());
        }
        if ((p4gen[jetInd].pt() > 100) && (p4gen[jetInd].pt() < 120)) {
          dPt100Frac4.Fill(dpt / p4gen[jetInd].pt());
        }
      }

      if ((p4gen[jetInd].pt() > minJetPt10) && (fabs(p4gen[jetInd].eta()) < 1.3)) {
        dR4.Fill(dRmin[jetInd]);
        double dphi = deltaPhi(p4cal[jetInd].phi(), p4gen[jetInd].phi());
        dPhi4.Fill(dphi);
        double deta = p4cal[jetInd].eta() - p4gen[jetInd].eta();
        dEta4.Fill(deta);
        dPt4.Fill(dpt);
        dPtFrac4.Fill(dpt / p4gen[jetInd].pt());
      }

      jetInd++;
    }
  }

  // *********************
  // MidPoint - Seedless Jet Matching

  Handle<CaloJetCollection> calo1Jets;
  Handle<CaloJetCollection> calo2Jets;
  Handle<CaloJetCollection> calo3Jets;

  evt.getByLabel(CaloJetAlgorithm1, calo1Jets);
  evt.getByLabel(CaloJetAlgorithm2, calo2Jets);
  evt.getByLabel(CaloJetAlgorithm3, calo3Jets);

  jetInd = 0;

  for (int i = 0; i < maxJets; ++i)
    used[i] = 0;
  for (CaloJetCollection::const_iterator cal1 = calo1Jets->begin(); cal1 != calo1Jets->end() && jetInd < maxJets;
       ++cal1) {
    p4gen[jetInd] = cal1->p4();  //Gen 4-vector
    dRmin[jetInd] = 1000.0;

    nj = 0;
    for (CaloJetCollection::const_iterator cal2 = calo2Jets->begin(); cal2 != calo2Jets->end(); ++cal2) {
      double delR = deltaR(cal1->eta(), cal1->phi(), cal2->eta(), cal2->phi());
      if ((delR < dRmin[jetInd]) && (used[nj] == 0)) {
        dRmin[jetInd] = delR;        // delta R of match
        p4cal[jetInd] = cal2->p4();  // Matched Cal 4-vector
        usedInd = nj;
      }
      nj++;
    }
    used[usedInd] = 1;

    if (p4gen[jetInd].pt() > minJetPt) {
      dR12.Fill(dRmin[jetInd]);
      double dphi = deltaPhi(p4cal[jetInd].phi(), p4gen[jetInd].phi());
      dPhi12.Fill(dphi);
      double deta = p4cal[jetInd].eta() - p4gen[jetInd].eta();
      dEta12.Fill(deta);
      double dpt = p4cal[jetInd].pt() - p4gen[jetInd].pt();
      dPt12.Fill(dpt);
    }

    jetInd++;
  }

  // ******************************************
  // ******************************************

  Handle<CandidateCollection> genParticles;
  evt.getByLabel("genParticleCandidates", genParticles);

  // *********************
  // Partons (Z')

  int nPart = 0;
  for (size_t i = 0; i < genParticles->size(); i++) {
    const Candidate& p = (*genParticles)[i];
    //    int Status =  p.status();
    //    bool ParticleIsStable = Status==1;
    int id = p.pdgId();

    if (id == 32) {
      if (p.numberOfDaughters() != 0) {
        /***
	cout << "Z': part = "   << i << " id = " << id 
	     << " daughters = " << p.numberOfDaughters() 
	     << " mass = "      << p.mass()
	     << endl;
	***/
        ZpMG = p.mass();
        ZpMassGen.Fill(ZpMG);
        if (EtaOk10 == 2) {
          ZpMassGen10.Fill(ZpMG);
          ZpMassGen_700_10.Fill(ZpMG);
          ZpMassGen_2000_10.Fill(ZpMG);
          ZpMassGen_5000_10.Fill(ZpMG);
        }
        if (EtaOk13 == 2) {
          ZpMassGen13.Fill(ZpMG);
          ZpMassGen_700_13.Fill(ZpMG);
          ZpMassGen_2000_13.Fill(ZpMG);
          ZpMassGen_5000_13.Fill(ZpMG);
        }
        if (EtaOk40 == 2) {
          ZpMassGen40.Fill(ZpMG);
          ZpMassGen_700_40.Fill(ZpMG);
          ZpMassGen_2000_40.Fill(ZpMG);
          ZpMassGen_5000_40.Fill(ZpMG);
        }
      }

      for (int id1 = 0, nd1 = p.numberOfDaughters(); id1 < nd1; ++id1) {
        const Candidate* d1 = p.daughter(id1);

        if (abs(d1->pdgId()) != 32) {
          math::XYZTLorentzVector momentum = d1->p4();
          p4Zp[nPart] = momentum = d1->p4();
          nPart++;
        }
      }
    }
  }

  // *********************
  // Match jets to Zp
  int genInd = 0;

  if (nPart == 2) {
    ZpM = (p4Zp[0] + p4Zp[1]).mass();
    ZpMass.Fill(ZpM);

    if (EtaOk10 == 2) {
      ZpMass_700_10.Fill(ZpM);
      ZpMass_2000_10.Fill(ZpM);
      ZpMass_5000_10.Fill(ZpM);
    }
    if (EtaOk13 == 2) {
      ZpMass_700_13.Fill(ZpM);
      ZpMass_2000_13.Fill(ZpM);
      ZpMass_5000_13.Fill(ZpM);
    }
    if (EtaOk40 == 2) {
      ZpMass_700_40.Fill(ZpM);
      ZpMass_2000_40.Fill(ZpM);
      ZpMass_5000_40.Fill(ZpM);
    }

    int usedInd;

    // ***********
    // **** Calor1
    usedInd = -1;
    jetInd = 0;

    for (int i = 0; i < maxJets; ++i)
      used[i] = 0;
    for (int i = 0; i < 2; ++i) {
      dRmin[jetInd] = 1000.0;

      int nj = 0;
      for (CaloJetCollection::const_iterator cal1 = calo1Jets->begin(); cal1 != calo1Jets->end() && jetInd < maxJets;
           ++cal1) {
        double delR = deltaR(cal1->eta(), cal1->phi(), p4Zp[i].eta(), p4Zp[i].phi());

        //	if ( (delR < dRmin[jetInd]) && (delR < matchedDelR) && (used[nj] == 0) ) {
        if ((delR < dRmin[jetInd]) && (used[nj] == 0)) {
          dRmin[jetInd] = delR;        // delta R of match
          p4cal[jetInd] = cal1->p4();  // Matched Cal 4-vector
          usedInd = nj;
          genInd = i;
        }

        /****
	cout << "Delta R = " << delR
	     << " deltaR min = " << dRmin[jetInd]
	     << " Ind = " << jetInd << " / " << nj << " / " << used[nj]
	     << " cal1 eta  = " << cal1->eta()
	     << " p4par eta = " << p4Zp[i].eta()
	     << " cal1 phi  = " << cal1->phi()
	     << " p4par phi = " << p4Zp[i].phi()
	     << endl;
	cout << "    " 
	     << " p4par = " << p4Zp[i]
	     << " p4cal = " << cal1->p4()
	     << endl;
	***/

        nj++;
      }

      // Found matched jet
      if (usedInd != -1) {
        used[usedInd] = 1;
        jetInd++;
      }
    }

    ZpMM = (p4cal[0] + p4cal[1]).mass();
    ZpMassMatched1.Fill(ZpMM);

    if ((ZpMG != 0) && (EtaOk40 == 2)) {
      ZpMassRes401.Fill((ZpMM - ZpMG) / ZpMG);

      ZpMassResL401.Fill((LeadMass1 - ZpMG) / ZpMG);
      ZpMassResL402.Fill((LeadMass2 - ZpMG) / ZpMG);
      ZpMassResL403.Fill((LeadMass3 - ZpMG) / ZpMG);

      ZpMassResRL401.Fill(LeadMass1 / ZpMG);
      ZpMassResRL402.Fill(LeadMass2 / ZpMG);
      ZpMassResRL403.Fill(LeadMass3 / ZpMG);

      ZpMassResRLoP401.Fill(LeadMass1 / LeadMassP1);
      ZpMassResRLoP402.Fill(LeadMass2 / LeadMassP2);
      ZpMassResRLoP403.Fill(LeadMass3 / LeadMassP2);

      ZpMassResPRL401.Fill(LeadMassP1 / ZpMG);
      ZpMassResPRL402.Fill(LeadMassP2 / ZpMG);
      ZpMassResPRL403.Fill(LeadMassP3 / ZpMG);
    }

    if ((ZpMG != 0) && (EtaOk10 == 2)) {
      ZpMassRes101.Fill((ZpMM - ZpMG) / ZpMG);

      ZpMassResL101.Fill((LeadMass1 - ZpMG) / ZpMG);
      ZpMassResL102.Fill((LeadMass2 - ZpMG) / ZpMG);
      ZpMassResL103.Fill((LeadMass3 - ZpMG) / ZpMG);

      ZpMassResRL101.Fill(LeadMass1 / ZpMG);
      ZpMassResRL102.Fill(LeadMass2 / ZpMG);
      ZpMassResRL103.Fill(LeadMass3 / ZpMG);

      ZpMassResRLoP101.Fill(LeadMass1 / LeadMassP1);
      ZpMassResRLoP102.Fill(LeadMass2 / LeadMassP2);
      ZpMassResRLoP103.Fill(LeadMass3 / LeadMassP2);

      ZpMassResPRL101.Fill(LeadMassP1 / ZpMG);
      ZpMassResPRL102.Fill(LeadMassP2 / ZpMG);
      ZpMassResPRL103.Fill(LeadMassP3 / ZpMG);
    }

    if ((ZpMG != 0) && (EtaOk13 == 2)) {
      ZpMassRes131.Fill((ZpMM - ZpMG) / ZpMG);

      ZpMassResL131.Fill((LeadMass1 - ZpMG) / ZpMG);
      ZpMassResL132.Fill((LeadMass2 - ZpMG) / ZpMG);
      ZpMassResL133.Fill((LeadMass3 - ZpMG) / ZpMG);

      ZpMassResRL131.Fill(LeadMass1 / ZpMG);
      ZpMassResRL132.Fill(LeadMass2 / ZpMG);
      ZpMassResRL133.Fill(LeadMass3 / ZpMG);

      ZpMassResRLoP131.Fill(LeadMass1 / LeadMassP1);
      ZpMassResRLoP132.Fill(LeadMass2 / LeadMassP2);
      ZpMassResRLoP133.Fill(LeadMass3 / LeadMassP2);

      ZpMassResPRL131.Fill(LeadMassP1 / ZpMG);
      ZpMassResPRL132.Fill(LeadMassP2 / ZpMG);
      ZpMassResPRL133.Fill(LeadMassP3 / ZpMG);
    }

    // ***********
    // **** Calor2
    usedInd = -1;
    jetInd = 0;

    for (int i = 0; i < maxJets; ++i)
      used[i] = 0;
    for (int i = 0; i < 2; ++i) {
      dRmin[jetInd] = 1000.0;

      int nj = 0;
      for (CaloJetCollection::const_iterator cal2 = calo2Jets->begin(); cal2 != calo2Jets->end() && jetInd < maxJets;
           ++cal2) {
        double delR = deltaR(cal2->eta(), cal2->phi(), p4Zp[i].eta(), p4Zp[i].phi());

        if ((delR < dRmin[jetInd]) && (delR < matchedDelR) && (used[nj] == 0)) {
          dRmin[jetInd] = delR;        // delta R of match
          p4cal[jetInd] = cal2->p4();  // Matched Cal 4-vector
          usedInd = nj;
        }

        /****	
	cout << "Delta R = " << delR
	     << " deltaR min = " << dRmin[jetInd]
	     << " Ind = " << jetInd << " / " << nj << " / " << used[nj]
	     << " p4par = " << p4par[i]
	     << " p4cal = " << cal1->p4()
	     << " cal1 eta  = " << cal1->eta()
	     << " p4par eta = " << p4par[i].eta()
	     << endl;
	****/

        nj++;
      }

      // Found matched jet
      if (usedInd != -1) {
        used[usedInd] = 1;
        jetInd++;
      }
    }

    ZpMM = (p4cal[0] + p4cal[1]).mass();
    ZpMassMatched2.Fill(ZpMM);
    ZpMassRes402.Fill((ZpMM - ZpM) / ZpM);

    // ***********
    // **** Calor3
    usedInd = -1;
    jetInd = 0;

    for (int i = 0; i < maxJets; ++i)
      used[i] = 0;
    for (int i = 0; i < 2; ++i) {
      dRmin[jetInd] = 1000.0;

      int nj = 0;
      for (CaloJetCollection::const_iterator cal3 = calo3Jets->begin(); cal3 != calo3Jets->end() && jetInd < maxJets;
           ++cal3) {
        double delR = deltaR(cal3->eta(), cal3->phi(), p4Zp[i].eta(), p4Zp[i].phi());

        if ((delR < dRmin[jetInd]) && (delR < matchedDelR) && (used[nj] == 0)) {
          dRmin[jetInd] = delR;        // delta R of match
          p4cal[jetInd] = cal3->p4();  // Matched Cal 4-vector
          usedInd = nj;
        }

        /****	
	cout << "Delta R = " << delR
	     << " deltaR min = " << dRmin[jetInd]
	     << " Ind = " << jetInd << " / " << nj << " / " << used[nj]
	     << " p4par = " << p4par[i]
	     << " p4cal = " << cal1->p4()
	     << " cal1 eta  = " << cal1->eta()
	     << " p4par eta = " << p4par[i].eta()
	     << endl;
	****/

        nj++;
      }

      // Found matched jet
      if (usedInd != -1) {
        used[usedInd] = 1;
        jetInd++;
      }
    }

    ZpMM = (p4cal[0] + p4cal[1]).mass();
    ZpMassMatched3.Fill(ZpMM);
    ZpMassRes403.Fill((ZpMM - ZpM) / ZpM);

  } else {
    cout << "Z' (3): nPart = " << nPart << endl;
  }

  // *********************
  // Partons (ttbar) Jet Matching

  //  cout << ">>> Begin MC list" << endl;
  int nJet = 0;

  int ii = 1;
  int jj = 4;

  for (size_t i = 0; i < genParticles->size(); i++) {
    const Candidate& p = (*genParticles)[i];
    //    int Status =  p.status();
    //    bool ParticleIsStable = Status==1;
    int id = p.pdgId();

    // Top Quarks
    if (abs(id) == 6) {
      cout << "TOP: id = " << id << " mass = " << p.mass() << endl;

      topMassParton.Fill(p.mass());

      if (id == 6)
        tMassGen.Fill(p.mass());
      if (id == -6)
        tbarMassGen.Fill(p.mass());

      for (size_t id1 = 0, nd1 = p.numberOfDaughters(); id1 < nd1; ++id1) {
        const Candidate* d1 = p.daughter(id1);

        // b - quark
        if (abs(d1->pdgId()) == 5) {
          math::XYZTLorentzVector momentum = d1->p4();
          p4par[nJet++] = momentum = d1->p4();

          cout << "Daughter1: id = " << d1->pdgId() << " daughters = " << d1->numberOfDaughters()
               << " mother 1   = " << (d1->mother())->pdgId() << " Momentum " << momentum << " GeV/c" << endl;

          if ((d1->mother())->pdgId() == 6) {
            p4part[0] = momentum = d1->p4();
            cout << ">>> part0 = " << p4part[0] << endl;
          }
          if ((d1->mother())->pdgId() == -6) {
            p4part[3] = momentum = d1->p4();
            cout << ">>> part3 = " << p4part[3] << endl;
          }
        }

        // W
        // Check for fully hadronic decay

        if (abs(d1->pdgId()) == 24) {
          for (size_t id2 = 0, nd2 = d1->numberOfDaughters(); id2 < nd2; ++id2) {
            const Candidate* d2 = d1->daughter(id2);

            if (abs(d2->pdgId()) < 9) {
              math::XYZVector vertex(d2->vx(), d2->vy(), d2->vz());
              math::XYZTLorentzVector momentum = d2->p4();
              p4par[nJet++] = momentum = d2->p4();

              if ((d1->mother())->pdgId() == 6) {
                p4part[ii] = momentum = d2->p4();
                cout << ">>> part" << ii << " = " << p4part[ii] << endl;
                ii++;
              }
              if ((d1->mother())->pdgId() == -6) {
                p4part[jj] = momentum = d2->p4();
                cout << ">>> part" << jj << " = " << p4part[jj] << endl;
                jj++;
              }

              cout << "Daughter2: id = " << d2->pdgId() << " daughters = " << d2->numberOfDaughters()
                   << " mother 2   = " << (d2->mother())->pdgId() << " Momentum " << momentum << " GeV/c" << endl;
            }
          }
        }

        //	if ( pdgId == d->pdgId() && d->status() == 1 ) {
        //	}
      }
    }
  }
  //  cout << ">>> N Jets = " << nJet << endl;

  if (nJet == 6) {
    double tmass = (p4part[0] + p4part[1] + p4part[2]).mass();
    double tbarmass = (p4part[3] + p4part[4] + p4part[5]).mass();

    tMass.Fill(tmass);
    tbarMass.Fill(tbarmass);

    cout << ">>> T Mass = " << tmass << " / " << tbarmass << endl;

    double mindR = 1000.;
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        if (j > i) {
          double delR = deltaR(p4par[i].eta(), p4par[i].phi(), p4par[j].eta(), p4par[j].phi());
          if (delR < mindR)
            mindR = delR;
          dRParton.Fill(delR);
        }
      }
    }
    dRPartonMin.Fill(mindR);

    int usedInd;
    usedInd = -1;
    jetInd = 0;

    for (int i = 0; i < maxJets; ++i)
      used[i] = 0;
    for (int i = 0; i < 6; ++i) {
      dRmin[jetInd] = 1000.0;

      int nj = 0;
      for (CaloJetCollection::const_iterator cal1 = calo1Jets->begin(); cal1 != calo1Jets->end() && jetInd < maxJets;
           ++cal1) {
        double delR = deltaR(cal1->eta(), cal1->phi(), p4par[i].eta(), p4par[i].phi());

        if ((delR < dRmin[jetInd]) && (delR < matchedDelR) && (used[nj] == 0)) {
          dRmin[jetInd] = delR;        // delta R of match
          p4cal[jetInd] = cal1->p4();  // Matched Cal 4-vector
          usedInd = nj;
          genInd = i;
        }

        /****	
	cout << "Delta R = " << delR
	     << " deltaR min = " << dRmin[jetInd]
	     << " Ind = " << jetInd << " / " << nj << " / " << used[nj]
	     << " p4par = " << p4par[i]
	     << " p4cal = " << cal1->p4()
	     << " cal1 eta  = " << cal1->eta()
	     << " p4par eta = " << p4par[i].eta()
	     << endl;
	****/

        nj++;
      }

      // Found matched jet
      if (usedInd != -1) {
        used[usedInd] = 1;

        dRPar1.Fill(dRmin[jetInd]);
        double dphi = deltaPhi(p4cal[jetInd].phi(), p4par[genInd].phi());
        dPhiPar1.Fill(dphi);
        double deta = p4cal[jetInd].eta() - p4par[genInd].eta();
        dEtaPar1.Fill(deta);
        double dpt = p4cal[jetInd].pt() - p4par[genInd].pt();
        dPtPar1.Fill(dpt);
        jetInd++;
      }
    }
    ParMatch1.Fill(jetInd);
    if (jetInd == 6) {
      topMass1.Fill((p4cal[0] + p4cal[1] + p4cal[2]).mass());
      topMass1.Fill((p4cal[3] + p4cal[4] + p4cal[5]).mass());
    }

    /***
    cout << "Collection Size = " <<  calo1Jets->size() 
	 << " / " << jetInd
	 << endl;
    ***/

    // ***********************
    jetInd = 0;
    usedInd = -1;

    for (int i = 0; i < maxJets; ++i)
      used[i] = 0;
    for (int i = 0; i < 6; ++i) {
      dRmin[jetInd] = 1000.0;

      int nj = 0;
      for (CaloJetCollection::const_iterator cal2 = calo2Jets->begin(); cal2 != calo2Jets->end() && jetInd < maxJets;
           ++cal2) {
        double delR = deltaR(cal2->eta(), cal2->phi(), p4par[i].eta(), p4par[i].phi());

        if ((delR < dRmin[jetInd]) && (delR < matchedDelR) && (used[nj] == 0)) {
          dRmin[jetInd] = delR;        // delta R of match
          p4cal[jetInd] = cal2->p4();  // Matched Cal 4-vector
          usedInd = nj;
          genInd = i;
        }

        /****
	cout << "Delta R = " << delR
	     << " deltaR min = " << dRmin[jetInd]
	     << " Ind = " << jetInd << " / " << nj << " / " << used[nj]
	     << " cal2 eta  = " << cal2->eta()
	     << " p4par eta = " << p4par[i].eta()
	     << endl;
	****/

        nj++;
      }
      if (usedInd != -1) {
        used[usedInd] = 1;

        dRPar2.Fill(dRmin[jetInd]);
        double dphi = deltaPhi(p4cal[jetInd].phi(), p4par[genInd].phi());
        dPhiPar2.Fill(dphi);
        double deta = p4cal[jetInd].eta() - p4par[genInd].eta();
        dEtaPar2.Fill(deta);
        double dpt = p4cal[jetInd].pt() - p4par[genInd].pt();
        dPtPar2.Fill(dpt);

        jetInd++;
      }
    }
    ParMatch2.Fill(jetInd);
    if (jetInd == 6) {
      topMass2.Fill((p4cal[0] + p4cal[1] + p4cal[2]).mass());
      topMass2.Fill((p4cal[3] + p4cal[4] + p4cal[5]).mass());
    }

    /***
    cout << "Collection Size = " <<  calo2Jets->size() 
	 << " / " << jetInd
	 << endl;
    ***/

    // ***********************
    jetInd = 0;
    usedInd = -1;

    for (int i = 0; i < maxJets; ++i)
      used[i] = 0;
    for (int i = 0; i < 6; ++i) {
      dRmin[jetInd] = 1000.0;

      int nj = 0;
      for (CaloJetCollection::const_iterator cal3 = calo3Jets->begin(); cal3 != calo3Jets->end() && jetInd < maxJets;
           ++cal3) {
        double delR = deltaR(cal3->eta(), cal3->phi(), p4par[i].eta(), p4par[i].phi());

        if ((delR < dRmin[jetInd]) && (delR < matchedDelR) && (used[nj] == 0)) {
          dRmin[jetInd] = delR;        // delta R of match
          p4cal[jetInd] = cal3->p4();  // Matched Cal 4-vector
          usedInd = nj;
          genInd = i;
        }
        /****
	cout << "Delta R = " << delR
	     << " deltaR min = " << dRmin[jetInd]
	     << " Ind = " << jetInd << " / " << nj << " / " << used[nj]
	     << " cal3 eta  = " << cal3->eta()
	     << " p4par eta = " << p4par[i].eta()
	     << endl;
	****/

        nj++;
      }
      if (usedInd != -1) {
        used[usedInd] = 1;

        dRPar3.Fill(dRmin[jetInd]);
        double dphi = deltaPhi(p4cal[jetInd].phi(), p4par[genInd].phi());
        dPhiPar3.Fill(dphi);
        double deta = p4cal[jetInd].eta() - p4par[genInd].eta();
        dEtaPar3.Fill(deta);
        double dpt = p4cal[jetInd].pt() - p4par[genInd].pt();
        dPtPar3.Fill(dpt);

        jetInd++;
      }
    }
    ParMatch3.Fill(jetInd);
    if (jetInd == 6) {
      topMass3.Fill((p4cal[0] + p4cal[1] + p4cal[2]).mass());
      topMass3.Fill((p4cal[3] + p4cal[4] + p4cal[5]).mass());
    }

    /***
    cout << "Collection Size = " <<  calo3Jets->size() 
	 << " / " << jetInd
	 << endl;
    ***/
  }

  int nTow1, nTow2, nTow3, nTow4;

  Handle<CaloJetCollection> jets;

  // *********************
  // Jet Properties
  // *********************

  // --- Loop over jets and make a list of all the used towers
  evt.getByLabel(CaloJetAlgorithm1, jets);
  int jjet = 0;
  for (CaloJetCollection::const_iterator ijet = jets->begin(); ijet != jets->end(); ijet++) {
    jjet++;

    float hadEne = ijet->hadEnergyInHB() + ijet->hadEnergyInHO() + ijet->hadEnergyInHE() + ijet->hadEnergyInHF();
    float emEne = ijet->emEnergyInEB() + ijet->emEnergyInEE() + ijet->emEnergyInHF();
    float had = ijet->energyFractionHadronic();

    float j_et = ijet->et();

    if (fabs(ijet->eta()) < 1.3) {
      totEneLeadJetEta1_1.Fill(hadEne + emEne);
      hadEneLeadJetEta1_1.Fill(hadEne);
      emEneLeadJetEta1_1.Fill(emEne);

      totEneLeadJetEta1_2.Fill(hadEne + emEne);
      hadEneLeadJetEta1_2.Fill(hadEne);
      emEneLeadJetEta1_2.Fill(emEne);

      if (ijet->pt() > minJetPt10)
        hadFracEta11.Fill(had);
    }
    if ((fabs(ijet->eta()) > 1.3) && (fabs(ijet->eta()) < 3.)) {
      totEneLeadJetEta2_1.Fill(hadEne + emEne);
      hadEneLeadJetEta2_1.Fill(hadEne);
      emEneLeadJetEta2_1.Fill(emEne);

      totEneLeadJetEta2_2.Fill(hadEne + emEne);
      hadEneLeadJetEta2_2.Fill(hadEne);
      emEneLeadJetEta2_2.Fill(emEne);

      if (ijet->pt() > minJetPt10)
        hadFracEta21.Fill(had);
    }
    if (fabs(ijet->eta()) > 3.) {
      totEneLeadJetEta3_1.Fill(hadEne + emEne);
      hadEneLeadJetEta3_1.Fill(hadEne);
      emEneLeadJetEta3_1.Fill(emEne);

      totEneLeadJetEta3_2.Fill(hadEne + emEne);
      hadEneLeadJetEta3_2.Fill(hadEne);
      emEneLeadJetEta3_2.Fill(emEne);

      if (ijet->pt() > minJetPt10)
        hadFracEta31.Fill(had);
    }

    if (jjet == 1) {
      hadFracLeadJet1.Fill(had);
      hadEneLeadJet1.Fill(hadEne);
      hadEneLeadJet12.Fill(hadEne);
      hadEneLeadJet13.Fill(hadEne);
      emEneLeadJet1.Fill(emEne);
      emEneLeadJet12.Fill(emEne);
      emEneLeadJet13.Fill(emEne);
    }

    const std::vector<CaloTowerPtr> jetCaloRefs = ijet->getCaloConstituents();
    int nConstituents = jetCaloRefs.size();

    if (jjet == 1) {
      nTow1 = nTow2 = nTow3 = nTow4 = 0;
      for (int i = 0; i < nConstituents; i++) {
        float et = jetCaloRefs[i]->et();

        if (et > 0.5)
          nTow1++;
        if (et > 1.0)
          nTow2++;
        if (et > 1.5)
          nTow3++;
        if (et > 2.0)
          nTow4++;

        hf_TowerJetEt1.Fill(et / j_et);
      }

      nTowersLeadJetPt1.Fill(nTow1);
      nTowersLeadJetPt2.Fill(nTow2);
      nTowersLeadJetPt3.Fill(nTow3);
      nTowersLeadJetPt4.Fill(nTow4);
    }

    if ((jjet == 1) && (fabs(ijet->eta()) < 1.3)) {
      nTowersLeadJet1.Fill(nConstituents);

      for (int i = 0; i < nConstituents; i++) {
        float t_et = jetCaloRefs[i]->et();
        double delR = deltaR(ijet->eta(), ijet->phi(), jetCaloRefs[i]->eta(), jetCaloRefs[i]->phi());
        hf_TowerDelR1.Fill(delR, t_et / j_et);
        hf_TowerDelR12.Fill(delR, t_et / j_et);
        TowerEtLeadJet1.Fill(t_et);
        TowerEtLeadJet12.Fill(t_et);
        TowerEtLeadJet13.Fill(t_et);
      }
    }

    if ((jjet == 2) && (fabs(ijet->eta()) < 1.3)) {
      nTowersSecondJet1.Fill(nConstituents);
    }
  }

  // *********************
  // Unclustered Energy
  // *********************

  double SumEtNotJets(0);
  double SumEtJets(0);

  double sumTowerAllEx(0);
  double sumTowerAllEy(0);

  std::vector<CaloTowerPtr> UsedTowerList;
  std::vector<CaloTower> TowerUsedInJets;
  std::vector<CaloTower> TowerNotUsedInJets;

  // *********************
  // Towers

  Handle<CaloTowerCollection> caloTowers;
  evt.getByLabel("towerMaker", caloTowers);

  nTow1 = nTow2 = nTow3 = nTow4 = 0;

  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;
  //  double sum_ez = 0.0;

  // --- Loop over towers and make a lists of used and unused towers
  for (CaloTowerCollection::const_iterator tower = caloTowers->begin(); tower != caloTowers->end(); tower++) {
    Double_t et = tower->et();

    if (et > 0.5)
      nTow1++;
    if (et > 1.0)
      nTow2++;
    if (et > 1.5)
      nTow3++;
    if (et > 2.0)
      nTow4++;

    if (et > 0.5) {
      // ********
      double phix = tower->phi();
      //      double theta = tower->theta();
      //      double e     = tower->energy();
      //      double et    = e*sin(theta);
      //      double et    = tower->emEt() + tower->hadEt();
      double et = tower->et();

      //      sum_ez += e*cos(theta);
      sum_et += et;
      sum_ex += et * cos(phix);
      sum_ey += et * sin(phix);
      // ********

      Double_t phi = tower->phi();

      sumTowerAllEx += et * cos(phi);
      sumTowerAllEy += et * sin(phi);
    }
  }

  SumEt1.Fill(sum_et);
  SumEt12.Fill(sum_et);
  SumEt13.Fill(sum_et);

  MET1.Fill(sqrt(sum_ex * sum_ex + sum_ey * sum_ey));
  MET12.Fill(sqrt(sum_ex * sum_ex + sum_ey * sum_ey));
  MET13.Fill(sqrt(sum_ex * sum_ex + sum_ey * sum_ey));

  //  met->mex   = -sum_ex;
  //  met->mey   = -sum_ey;
  //  met->mez   = -sum_ez;
  //  met->met   = sqrt( sum_ex*sum_ex + sum_ey*sum_ey );
  // cout << "MET = " << met->met << endl;
  //  met->sumet = sum_et;
  //  met->phi   = atan2( -sum_ey, -sum_ex );

  hf_sumTowerAllEx.Fill(sumTowerAllEx);
  hf_sumTowerAllEy.Fill(sumTowerAllEy);

  nTowers1.Fill(nTow1);
  nTowers2.Fill(nTow2);
  nTowers3.Fill(nTow3);
  nTowers4.Fill(nTow4);

  // *********************
  // MidPoint
  //
  UsedTowerList.clear();
  TowerUsedInJets.clear();
  TowerNotUsedInJets.clear();

  // --- Loop over jets and make a list of all the used towers
  evt.getByLabel(CaloJetAlgorithm1, jets);
  for (CaloJetCollection::const_iterator ijet = jets->begin(); ijet != jets->end(); ijet++) {
    //    if (jetPt>5.0) {

    const std::vector<CaloTowerPtr> jetCaloRefs = ijet->getCaloConstituents();
    int nConstituents = jetCaloRefs.size();
    for (int i = 0; i < nConstituents; i++) {
      UsedTowerList.push_back(jetCaloRefs[i]);
    }

    //    }
  }

  int NTowersUsed = UsedTowerList.size();

  // --- Loop over towers and make a lists of used and unused towers
  for (CaloTowerCollection::const_iterator tower = caloTowers->begin(); tower != caloTowers->end(); tower++) {
    const CaloTower& t = *tower;
    Double_t et = tower->et();

    if (et > 0) {
      bool used = false;

      for (int i = 0; i < NTowersUsed; i++) {
        if (tower->id() == UsedTowerList[i]->id()) {
          used = true;
          break;
        }
      }

      if (used) {
        TowerUsedInJets.push_back(t);
      } else {
        TowerNotUsedInJets.push_back(t);
      }
    }
  }

  int nUsed = TowerUsedInJets.size();
  int nNotUsed = TowerNotUsedInJets.size();

  SumEtJets = 0;
  SumEtNotJets = 0;

  for (int i = 0; i < nUsed; i++) {
    SumEtJets += TowerUsedInJets[i].et();
  }
  h_jetEt1.Fill(SumEtJets);

  for (int i = 0; i < nNotUsed; i++) {
    if (TowerNotUsedInJets[i].et() > 0.5)
      SumEtNotJets += TowerNotUsedInJets[i].et();
    h_missEt1.Fill(TowerNotUsedInJets[i].et());
    h_missEt1s.Fill(TowerNotUsedInJets[i].et());
  }
  h_totMissEt1.Fill(SumEtNotJets);

  // *********************
  // SISCone
  //
  UsedTowerList.clear();
  TowerUsedInJets.clear();
  TowerNotUsedInJets.clear();

  // --- Loop over jets and make a list of all the used towers
  evt.getByLabel(CaloJetAlgorithm2, jets);
  for (CaloJetCollection::const_iterator ijet = jets->begin(); ijet != jets->end(); ijet++) {
    //    if (jetPt>5.0) {

    const std::vector<CaloTowerPtr> jetCaloRefs = ijet->getCaloConstituents();
    int nConstituents = jetCaloRefs.size();
    for (int i = 0; i < nConstituents; i++) {
      UsedTowerList.push_back(jetCaloRefs[i]);
    }

    //    }
  }

  //  Handle<CaloTowerCollection> caloTowers;
  //  evt.getByLabel( "towerMaker", caloTowers );

  NTowersUsed = UsedTowerList.size();

  // --- Loop over towers and make a lists of used and unused towers
  for (CaloTowerCollection::const_iterator tower = caloTowers->begin(); tower != caloTowers->end(); tower++) {
    const CaloTower& t = *tower;
    Double_t et = tower->et();

    if (et > 0) {
      bool used = false;

      for (int i = 0; i < NTowersUsed; i++) {
        if (tower->id() == UsedTowerList[i]->id()) {
          used = true;
          break;
        }
      }

      if (used) {
        TowerUsedInJets.push_back(t);
      } else {
        TowerNotUsedInJets.push_back(t);
      }
    }
  }

  nUsed = TowerUsedInJets.size();
  nNotUsed = TowerNotUsedInJets.size();

  SumEtJets = 0;
  SumEtNotJets = 0;

  for (int i = 0; i < nUsed; i++) {
    SumEtJets += TowerUsedInJets[i].et();
  }
  h_jetEt2.Fill(SumEtJets);

  for (int i = 0; i < nNotUsed; i++) {
    if (TowerNotUsedInJets[i].et() > 0.5)
      SumEtNotJets += TowerNotUsedInJets[i].et();
    h_missEt2.Fill(TowerNotUsedInJets[i].et());
    h_missEt2s.Fill(TowerNotUsedInJets[i].et());
  }
  h_totMissEt2.Fill(SumEtNotJets);

  // *********************
  // KtClus
  //
  UsedTowerList.clear();
  TowerUsedInJets.clear();
  TowerNotUsedInJets.clear();

  // --- Loop over jets and make a list of all the used towers
  evt.getByLabel(CaloJetAlgorithm3, jets);
  for (CaloJetCollection::const_iterator ijet = jets->begin(); ijet != jets->end(); ijet++) {
    //    if (jetPt>5.0) {

    const std::vector<CaloTowerPtr> jetCaloRefs = ijet->getCaloConstituents();
    int nConstituents = jetCaloRefs.size();
    for (int i = 0; i < nConstituents; i++) {
      UsedTowerList.push_back(jetCaloRefs[i]);
    }

    //    }
  }

  //  Handle<CaloTowerCollection> caloTowers;
  //  evt.getByLabel( "towerMaker", caloTowers );

  NTowersUsed = UsedTowerList.size();

  // --- Loop over towers and make a lists of used and unused towers
  for (CaloTowerCollection::const_iterator tower = caloTowers->begin(); tower != caloTowers->end(); tower++) {
    const CaloTower& t = *tower;
    Double_t et = tower->et();

    if (et > 0) {
      bool used = false;

      for (int i = 0; i < NTowersUsed; i++) {
        if (tower->id() == UsedTowerList[i]->id()) {
          used = true;
          break;
        }
      }

      if (used) {
        TowerUsedInJets.push_back(t);
      } else {
        TowerNotUsedInJets.push_back(t);
      }
    }
  }

  nUsed = TowerUsedInJets.size();
  nNotUsed = TowerNotUsedInJets.size();

  SumEtJets = 0;
  SumEtNotJets = 0;

  for (int i = 0; i < nUsed; i++) {
    SumEtJets += TowerUsedInJets[i].et();
  }
  h_jetEt3.Fill(SumEtJets);

  for (int i = 0; i < nNotUsed; i++) {
    if (TowerNotUsedInJets[i].et() > 0.5)
      SumEtNotJets += TowerNotUsedInJets[i].et();
    h_missEt3.Fill(TowerNotUsedInJets[i].et());
    h_missEt3s.Fill(TowerNotUsedInJets[i].et());
  }
  h_totMissEt3.Fill(SumEtNotJets);
}

void myFastSimVal::endJob() {
  //Write out the histogram file.
  m_file->Write();
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(myFastSimVal);
