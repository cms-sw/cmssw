#ifndef HLTrigger_HLTanalyzers_HLTBJet_h
#define HLTrigger_HLTanalyzers_HLTBJet_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

class TTree;

class HLTBJet {
public:
  HLTBJet();
  ~HLTBJet();
  
  void setup(const edm::ParameterSet & config, TTree * tree);
  void analyze(const edm::Event & event, const edm::EventSetup & setup, TTree * tree);

private:
  void analyzeLifetime(
      const edm::View<reco::Jet>   & lifetimeBjetL2, 
      const reco::JetTagCollection & lifetimeBjetL25, 
      const reco::JetTagCollection & lifetimeBjetL3);

  void analyzeSoftmuon(
      const edm::View<reco::Jet>   & softmuonBjetL2, 
      const reco::JetTagCollection & softmuonBjetL25, 
      const reco::JetTagCollection & softmuonBjetL3);

  void analyzePerformance(
      const edm::View<reco::Jet>   & performanceBjetL2, 
      const reco::JetTagCollection & performanceBjetL25, 
      const reco::JetTagCollection & performanceBjetL3);

  // labels for input HLT collections
  edm::InputTag m_globalEnergy;
  edm::InputTag m_lifetimeBjetL2;
  edm::InputTag m_lifetimeBjetL25;
  edm::InputTag m_lifetimeBjetL3;
  edm::InputTag m_softmuonBjetL2;
  edm::InputTag m_softmuonBjetL25;
  edm::InputTag m_softmuonBjetL3;
  edm::InputTag m_performanceBjetL2;
  edm::InputTag m_performanceBjetL25;
  edm::InputTag m_performanceBjetL3;

  // set of variables for lifetime-based b-tag
  int m_lifetimeBJets;
  float * m_lifetimeBJetL2Energy;
  float * m_lifetimeBJetL2ET;
  float * m_lifetimeBJetL2Eta;
  float * m_lifetimeBJetL2Phi;
  float * m_lifetimeBJetL25Discriminator;
  float * m_lifetimeBJetL3Discriminator;

  // set of variables for soft-muon-based b-tag
  int m_softmuonBJets;
  float * m_softmuonBJetL2Energy;
  float * m_softmuonBJetL2ET;
  float * m_softmuonBJetL2Eta;
  float * m_softmuonBJetL2Phi;
  int   * m_softmuonBJetL25Discriminator;       // do not optimize
  float * m_softmuonBJetL3Discriminator;
  
  // set of variables for b-tagging performance measurements
  int m_performanceBJets;
  float * m_performanceBJetL2Energy;
  float * m_performanceBJetL2ET;
  float * m_performanceBJetL2Eta;
  float * m_performanceBJetL2Phi;
  int   * m_performanceBJetL25Discriminator;    // do not optimize 
  int   * m_performanceBJetL3Discriminator;     // do not optimize
};

#endif // HLTrigger_HLTanalyzers_HLTBJet_h
