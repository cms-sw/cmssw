/** \class HLTFatJetMassFilter
*
*
*  \author Maurizio Pierini
*
*/

#include <vector>

#include "HLTrigger/JetMET/interface/HLTFatJetMassFilter.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include "DataFormats/Math/interface/deltaR.h"

//
// constructors and destructor
//
template<typename jetType>
HLTFatJetMassFilter<jetType>::HLTFatJetMassFilter(const edm::ParameterSet& iConfig) : 
  HLTFilter(iConfig),
  inputJetTag_  (iConfig.template getParameter< edm::InputTag > ("inputJetTag")),
  minMass_      (iConfig.template getParameter<double> ("minMass")),
  fatJetDeltaR_ (iConfig.template getParameter<double> ("fatJetDeltaR")),
  maxDeltaEta_  (iConfig.template getParameter<double> ("maxDeltaEta")),
  maxJetEta_    (iConfig.template getParameter<double> ("maxJetEta")),
  minJetPt_     (iConfig.template getParameter<double> ("minJetPt")),
  triggerType_  (iConfig.template getParameter<int> ("triggerType"))
{
  m_theJetToken = consumes<std::vector<jetType>>(inputJetTag_);
  LogDebug("") << "HLTFatJetMassFilter: Input/minMass/fatJetDeltaR/maxDeltaEta/maxJetEta/minJetPt/triggerType : "
	       << inputJetTag_.encode() << " "
	       << minMass_ << " " 
	       << fatJetDeltaR_ << " "
	       << maxDeltaEta_ << " "
	       << maxJetEta_ << " "
	       << minJetPt_ << " "
	       << triggerType_;
}

template<typename jetType>
HLTFatJetMassFilter<jetType>::~HLTFatJetMassFilter(){}

template<typename jetType>
void 
HLTFatJetMassFilter<jetType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltCollection"));
  desc.add<double>("minMass",0.0);
  desc.add<double>("fatJetDeltaR",1.1);
  desc.add<double>("maxDeltaEta",10.0);
  desc.add<double>("maxJetEta",3.0);
  desc.add<double>("minJetPt",30.0);
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add(defaultModuleLabel<HLTFatJetMassFilter<jetType>>(), desc);
}

// ------------ method called to produce the data  ------------
template<typename jetType>
bool
HLTFatJetMassFilter<jetType>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace std;
  using namespace edm;
  using namespace reco;

  typedef vector<jetType> JetCollection;
  typedef Ref<JetCollection> JetRef;

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputJetTag_);
  
  // All jets
  Handle<JetCollection> objects;
  iEvent.getByToken ( m_theJetToken,objects);
  
  // Selected jets
  CaloJetCollection recojets;
  typename JetCollection::const_iterator i ( objects->begin() );
  for(;i != objects->end(); i++){
    if(std::abs(i->eta()) < maxJetEta_ && i->pt() >= minJetPt_){ 
      reco::CaloJet jet(i->p4(), i->vertex(), reco::CaloJet::Specific()); 
      recojets.push_back(jet);
    }
  }
  
  // events with at least two jets
  if(recojets.size() < 2) return false;

  math::PtEtaPhiMLorentzVector j1(0.1, 0., 0., 0.);
  math::PtEtaPhiMLorentzVector j2(0.1, 0., 0., 0.);
  double jetPt1 = 0.;
  double jetPt2 = 0.;
  
  // look for the two highest-pT jet
  CaloJetCollection::const_iterator recojet ( recojets.begin() );
  for (; recojet != recojets.end() ; recojet++) {
    if(recojet->pt() > jetPt1) {
      // downgrade the 1st jet to 2nd jet
      j2 = j1;
      jetPt2 = j2.pt();
      // promote this jet to 1st jet
      j1 = recojet->p4();
      jetPt1 = recojet->pt();
    } else if(recojet->pt() > jetPt2) {
      // promote this jet to 2nd jet
      j2 = recojet->p4();
      jetPt2 = recojet->pt();
    }
  }
  
  // apply DeltaEta cut
  double DeltaEta = std::abs(j1.eta() - j2.eta());
  if(DeltaEta > maxDeltaEta_) return false;
  
  math::PtEtaPhiMLorentzVector fj1;
  math::PtEtaPhiMLorentzVector fj2;
  
  // apply radiation recovery
  for ( recojet = recojets.begin() ; recojet != recojets.end() ; recojet++) {
    double DeltaR1sq = reco::deltaR2(*recojet, j1);
    double DeltaR2sq = reco::deltaR2(*recojet, j2);
    if(DeltaR1sq < DeltaR2sq && DeltaR1sq < fatJetDeltaR_*fatJetDeltaR_) {
      fj1 += recojet->p4();
    } else if(DeltaR2sq < fatJetDeltaR_*fatJetDeltaR_) {
      fj2 += recojet->p4();
    }
  }

  // Apply mass cut
  fj1 += fj2;
  if(fj1.mass() < minMass_) return false;

  return true;
}
