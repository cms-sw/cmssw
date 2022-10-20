#ifndef JetAnalyzer_HeavyIons_matching_H
#define JetAnalyzer_HeavyIons_matching_H

//
// Jet Tester class for heavy ion jets. for DQM jet analysis monitoring
// For CMSSW_7_5_X, especially reading background subtracted jets
// author: Raghav Kunnawalkam Elayavalli,
//         April 6th 2015
//         Rutgers University, email: raghav.k.e at CERN dot CH
//
// The logic for the matching is taken from Pawan Kumar Netrakanti's macro analysis level macro available here
// https://github.com/pawannetrakanti/UserCode/blob/master/JetRAA/jetmatch.C
//

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/CandidateWithRef.h"

// include the basic jet for the PuPF jets.
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
// include the pf candidates
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
// include the voronoi subtraction
#include "DataFormats/HeavyIonEvent/interface/VoronoiBackground.h"
#include "RecoHI/HiJetAlgos/interface/UEParameters.h"
// include the centrality variables
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"

#include "RecoJets/JetProducers/interface/JetIDHelper.h"
#include "DQMOffline/JetMET/interface/JetMETDQMDCSFilter.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"
#include "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include <utility>

class JetAnalyzer_HeavyIons_matching : public DQMEDAnalyzer {
public:
  explicit JetAnalyzer_HeavyIons_matching(const edm::ParameterSet &);
  ~JetAnalyzer_HeavyIons_matching() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  // calojet1token = vscalo
  // calojet2token = pucalo

private:
  edm::InputTag mInputJet1Collection;
  edm::InputTag mInputJet2Collection;

  std::string mOutputFile;
  std::string JetType1;
  std::string JetType2;
  double mRecoJetPtThreshold;
  double mRecoDelRMatch;
  double mRecoJetEtaCut;

  //Tokens
  edm::EDGetTokenT<reco::CaloJetCollection> caloJet1Token_;
  edm::EDGetTokenT<reco::CaloJetCollection> caloJet2Token_;
  edm::EDGetTokenT<reco::PFJetCollection> pfJetsToken_;
  edm::EDGetTokenT<reco::BasicJetCollection> basicJetsToken_;
  edm::EDGetTokenT<reco::JPTJetCollection> jptJetsToken_;

  MonitorElement *mpT_ratio_Jet1Jet2;
  MonitorElement *mpT_Jet1_matched;
  MonitorElement *mpT_Jet2_matched;
  MonitorElement *mpT_Jet1_unmatched;
  MonitorElement *mpT_Jet2_unmatched;

  MonitorElement *mHadEnergy_Jet1_unmatched;
  MonitorElement *mEmEnergy_Jet1_unmatched;
  MonitorElement *mChargedHadronEnergy_Jet1_unmatched;
  MonitorElement *mNeutralHadronEnergy_Jet1_unmatched;
  MonitorElement *mChargedEmEnergy_Jet1_unmatched;
  MonitorElement *mNeutralEmEnergy_Jet1_unmatched;
  MonitorElement *mChargedMuEnergy_Jet1_unmatched;

  MonitorElement *mChargedHadEnergyFraction_Jet1_unmatched;
  MonitorElement *mNeutralHadEnergyFraction_Jet1_unmatched;
  MonitorElement *mPhotonEnergyFraction_Jet1_unmatched;
  MonitorElement *mElectronEnergyFraction_Jet1_unmatched;
  MonitorElement *mMuonEnergyFraction_Jet1_unmatched;

  MonitorElement *mHadEnergy_Jet2_unmatched;
  MonitorElement *mEmEnergy_Jet2_unmatched;
  MonitorElement *mChargedHadronEnergy_Jet2_unmatched;
  MonitorElement *mNeutralHadronEnergy_Jet2_unmatched;
  MonitorElement *mChargedEmEnergy_Jet2_unmatched;
  MonitorElement *mNeutralEmEnergy_Jet2_unmatched;
  MonitorElement *mChargedMuEnergy_Jet2_unmatched;

  MonitorElement *mChargedHadEnergyFraction_Jet2_unmatched;
  MonitorElement *mNeutralHadEnergyFraction_Jet2_unmatched;
  MonitorElement *mPhotonEnergyFraction_Jet2_unmatched;
  MonitorElement *mElectronEnergyFraction_Jet2_unmatched;
  MonitorElement *mMuonEnergyFraction_Jet2_unmatched;

  struct MyJet {
    int id;
    float pt;
    float eta;
    float phi;
  };

  typedef std::pair<MyJet, MyJet> ABJetPair;

  struct CompareMatchedJets {
    //! A-B jet match
    bool operator()(const ABJetPair &A1, const ABJetPair &A2) const {
      MyJet jet1_pair1 = A1.first;   //! Jet1 1st pair
      MyJet jet2_pair1 = A1.second;  //! Jet2 1st pair
      MyJet jet1_pair2 = A2.first;   //! Jet1 2nd pair
      MyJet jet2_pair2 = A2.second;  //! Jet2 2nd pair
      float delr1 = deltaRR(jet1_pair1.eta, jet1_pair1.phi, jet2_pair1.eta, jet2_pair1.phi);
      float delr2 = deltaRR(jet1_pair2.eta, jet1_pair2.phi, jet2_pair2.eta, jet2_pair2.phi);

      return ((delr1 < delr2) && (jet1_pair1.pt > jet1_pair2.pt));
    }
  };

  typedef std::multiset<ABJetPair, CompareMatchedJets> ABMatchedJets;
  typedef std::multiset<ABJetPair>::iterator ABItr;

  static float deltaRR(float eta1, float phi1, float eta2, float phi2) {
    float deta = eta1 - eta2;
    float dphi = fabs(phi1 - phi2);
    if (dphi > M_PI)
      dphi -= 2 * M_PI;
    float dr = sqrt(deta * deta + dphi * dphi);
    return dr;
  }
};

#endif
