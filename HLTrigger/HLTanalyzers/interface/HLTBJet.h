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
  void clear(void);
  void analyze(const edm::Handle<edm::View<reco::Jet> >  & rawBJets,
               const edm::Handle<edm::View<reco::Jet> >  & correctedBJets, 
               const edm::Handle<edm::View<reco::Jet> >  & correctedBJetsL1FastJet, 
               const edm::Handle<edm::View<reco::Jet> >  & pfBJets, 
               const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL25,
               const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL3,
               const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL25L1FastJet,
               const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL3L1FastJet,
               const edm::Handle<reco::JetTagCollection> & lifetimePFBJetsL3,
               const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL25SingleTrack,
               const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL3SingleTrack,
               const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL25SingleTrackL1FastJet,
               const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL3SingleTrackL1FastJet,
               const edm::Handle<reco::JetTagCollection> & performanceBJetsL25,
               const edm::Handle<reco::JetTagCollection> & performanceBJetsL3,
               const edm::Handle<reco::JetTagCollection> & performanceBJetsL25L1FastJet,
               const edm::Handle<reco::JetTagCollection> & performanceBJetsL3L1FastJet,
               TTree * tree);

private:
  void analyseJets(
      const edm::View<reco::Jet>   & jets);
  
  void analyseCorrectedJets(
      const edm::View<reco::Jet>   & jets);
  
  void analyseCorrectedJetsL1FastJet(
      const edm::View<reco::Jet>   & jets);
  
  void analysePFJets(
      const edm::View<reco::Jet>   & jets);
  
  void analyseLifetime(
      const edm::View<reco::Jet>   & jets, 
      const reco::JetTagCollection & tagsL25, 
      const reco::JetTagCollection & tagsL3);

  void analyseLifetimeL1FastJet(
      const edm::View<reco::Jet>   & jets, 
      const reco::JetTagCollection & tagsL25, 
      const reco::JetTagCollection & tagsL3);

  void analyseLifetimePF(
      const edm::View<reco::Jet>   & jets, 
      const reco::JetTagCollection & tagsL3);

  void analyseLifetimeSingleTrack(
      const edm::View<reco::Jet>   & jets, 
      const reco::JetTagCollection & tagsL25, 
      const reco::JetTagCollection & tagsL3);

  void analyseLifetimeSingleTrackL1FastJet(
      const edm::View<reco::Jet>   & jets, 
      const reco::JetTagCollection & tagsL25, 
      const reco::JetTagCollection & tagsL3);

  void analysePerformance(
      const edm::View<reco::Jet>   & jets, 
      const reco::JetTagCollection & tagsL25, 
      const reco::JetTagCollection & tagsL3);

  void analysePerformanceL1FastJet(
      const edm::View<reco::Jet>   & jets, 
      const reco::JetTagCollection & tagsL25, 
      const reco::JetTagCollection & tagsL3);

  // set of variables for uncorrected L2 jets
  int NohBJetL2;
  float * ohBJetL2Energy;
  float * ohBJetL2Et;
  float * ohBJetL2Pt;
  float * ohBJetL2Eta;
  float * ohBJetL2Phi;
              
  // set of variables for corrected L2 jets
  int NohBJetL2Corrected;
  float * ohBJetL2CorrectedEnergy;
  float * ohBJetL2CorrectedEt;
  float * ohBJetL2CorrectedPt;
  float * ohBJetL2CorrectedEta;
  float * ohBJetL2CorrectedPhi;
  
  // set of variables for corrected L2 jets L1FastJet
  int NohBJetL2CorrectedL1FastJet;
  float * ohBJetL2CorrectedEnergyL1FastJet;
  float * ohBJetL2CorrectedEtL1FastJet;
  float * ohBJetL2CorrectedPtL1FastJet;
  float * ohBJetL2CorrectedEtaL1FastJet;
  float * ohBJetL2CorrectedPhiL1FastJet;
  
  // set of variables for uncorrected L2 PF jets
  int NohpfBJetL2;
  float * ohpfBJetL2Energy;
  float * ohpfBJetL2Et;
  float * ohpfBJetL2Pt;
  float * ohpfBJetL2Eta;
  float * ohpfBJetL2Phi;
              
  // set of variables for lifetime-based b-tag
  float * ohBJetIPL25Tag;
  float * ohBJetIPL3Tag;

  // set of variables for lifetime-based b-tag L1FastJet
  float * ohBJetIPL25TagL1FastJet;
  float * ohBJetIPL3TagL1FastJet;

  // set of variables for lifetime-based b-tag PF jets
  float * ohpfBJetIPL3Tag;
  
  // set of variables for lifetime-based b-tag Single Track
  float * ohBJetIPL25TagSingleTrack;
  float * ohBJetIPL3TagSingleTrack;
  float * ohBJetIPL25TagSingleTrackL1FastJet;
  float * ohBJetIPL3TagSingleTrackL1FastJet;
  
  // set of variables for b-tagging performance measurements
  int   * ohBJetPerfL25Tag;         // do not optimize 
  int   * ohBJetPerfL3Tag;          // do not optimize
  // set of variables for b-tagging performance measurements L1FastJet
  int   * ohBJetPerfL25TagL1FastJet;         // do not optimize 
  int   * ohBJetPerfL3TagL1FastJet;          // do not optimize
};

#endif // HLTrigger_HLTanalyzers_HLTBJet_h
