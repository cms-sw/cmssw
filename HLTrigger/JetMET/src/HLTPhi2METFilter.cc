/** \class HLTPhi2METFilter
 *
 *
 *  \author Dominique J. Mangeol
 *
 */

#include "HLTrigger/JetMET/interface/HLTPhi2METFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/METReco/interface/CaloMET.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// constructors and destructor
//
HLTPhi2METFilter::HLTPhi2METFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  inputJetTag_ = iConfig.getParameter<edm::InputTag>("inputJetTag");
  inputMETTag_ = iConfig.getParameter<edm::InputTag>("inputMETTag");
  minDPhi_ = iConfig.getParameter<double>("minDeltaPhi");
  maxDPhi_ = iConfig.getParameter<double>("maxDeltaPhi");
  minEtjet1_ = iConfig.getParameter<double>("minEtJet1");
  minEtjet2_ = iConfig.getParameter<double>("minEtJet2");
  m_theJetToken = consumes<reco::CaloJetCollection>(inputJetTag_);
  m_theMETToken = consumes<trigger::TriggerFilterObjectWithRefs>(inputMETTag_);
}

HLTPhi2METFilter::~HLTPhi2METFilter() = default;

void HLTPhi2METFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputJetTag", edm::InputTag("iterativeCone5CaloJets"));
  desc.add<edm::InputTag>("inputMETTag", edm::InputTag("hlt1MET60"));
  desc.add<double>("minDeltaPhi", 0.377);
  desc.add<double>("minEtJet2", 60.);
  descriptions.add("hltPhi2METFilter", desc);
}

// ------------ method called to produce the data  ------------
bool HLTPhi2METFilter::hltFilter(edm::Event& iEvent,
                                 const edm::EventSetup& iSetup,
                                 trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  // The filter object
  if (saveTags()) {
    filterproduct.addCollectionTag(inputJetTag_);
    filterproduct.addCollectionTag(inputMETTag_);
  }

  Handle<CaloJetCollection> recocalojets;
  iEvent.getByToken(m_theJetToken, recocalojets);
  Handle<trigger::TriggerFilterObjectWithRefs> metcal;
  iEvent.getByToken(m_theMETToken, metcal);

  // look at all candidates,  check cuts and add to filter object
  int n(0);

  VRcalomet vrefMET;
  metcal->getObjects(TriggerMET, vrefMET);
  CaloMETRef metRef = vrefMET.at(0);
  CaloJetRef ref1, ref2;

  if (recocalojets->size() > 1) {
    // events with two or more jets

    double etjet1 = 0.;
    double etjet2 = 0.;
    double phijet2 = 0.;
    // double etmiss  = vrefMET.at(0)->et();
    double phimiss = vrefMET.at(0)->phi();
    int countjets = 0;

    for (auto recocalojet = recocalojets->begin(); recocalojet <= (recocalojets->begin() + 1); recocalojet++) {
      if (countjets == 0) {
        etjet1 = recocalojet->et();
        ref1 = CaloJetRef(recocalojets, distance(recocalojets->begin(), recocalojet));
      }
      if (countjets == 1) {
        etjet2 = recocalojet->et();
        phijet2 = recocalojet->phi();
        ref2 = CaloJetRef(recocalojets, distance(recocalojets->begin(), recocalojet));
      }
      countjets++;
    }
    double Dphi = std::abs(phimiss - phijet2);
    if (Dphi > M_PI)
      Dphi = 2.0 * M_PI - Dphi;
    if (etjet1 > minEtjet1_ && etjet2 > minEtjet2_ && Dphi >= minDPhi_ && Dphi <= maxDPhi_) {
      filterproduct.addObject(TriggerMET, metRef);
      filterproduct.addObject(TriggerJet, ref1);
      filterproduct.addObject(TriggerJet, ref2);
      n++;
    }

  }  // events with two or more jets

  // filter decision
  bool accept(n >= 1);

  return accept;
}
