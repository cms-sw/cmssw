/** \class HLTFatJetMassFilter
*
*
*  \author Maurizio Pierini
*
*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HLTrigger/JetMET/interface/HLTFatJetMassFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <vector>


//
// constructors and destructor
//
HLTFatJetMassFilter::HLTFatJetMassFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
  inputJetTag_  = iConfig.getParameter< edm::InputTag > ("inputJetTag");
  minMass_      = iConfig.getParameter<double> ("minMass");
  fatJetDeltaR_ = iConfig.getParameter<double> ("fatJetDeltaR");
  maxDeltaEta_  = iConfig.getParameter<double> ("maxDeltaEta");
  maxJetEta_    = iConfig.getParameter<double> ("maxJetEta");
  minJetPt_     = iConfig.getParameter<double> ("minJetPt");
}

HLTFatJetMassFilter::~HLTFatJetMassFilter(){}

void HLTFatJetMassFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltCollection"));
  desc.add<double>("minMass",0.0);
  desc.add<double>("fatJetDeltaR",1.1);
  desc.add<double>("maxDeltaEta",10.0);
  desc.add<double>("maxJetEta",3.0);
  desc.add<double>("minJetPt",30.0);
  descriptions.add("hltFatJetMassFilter",desc);
}

// ------------ method called to produce the data  ------------
bool
  HLTFatJetMassFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace std;
  using namespace edm;
  using namespace reco;

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputJetTag_);

  // All jets 
  Handle<CaloJetCollection> allrecocalojets;
  iEvent.getByLabel(inputJetTag_,allrecocalojets);

  // Selected jets          
  CaloJetCollection recocalojets;
  CaloJetCollection::const_iterator aBegin ( allrecocalojets->begin() );
  CaloJetCollection::const_iterator aEnd ( allrecocalojets->end() );
  for (CaloJetCollection::const_iterator allrecojet = aBegin ; allrecojet != aEnd; allrecojet++) {
    if(fabs(allrecojet->eta()) < maxJetEta_ && allrecojet->pt() >= minJetPt_) {
      recocalojets.push_back(*allrecojet);
    }
  }

  // events with at least two jets
  if(recocalojets.size() < 2) return false;

  math::PtEtaPhiMLorentzVector j1(0.1, 0., 0., 0.);
  math::PtEtaPhiMLorentzVector j2(0.1, 0., 0., 0.);
  double jetPt1 = 0.;
  double jetPt2 = 0.;
  // look for the two highest-pT jet
  for (CaloJetCollection::const_iterator recocalojet = recocalojets.begin();
       recocalojet != recocalojets.end(); recocalojet++) {
    if(recocalojet->pt() > jetPt1) {
      // downgrade the 1st jet to 2nd jet
      j2 = j1;
      jetPt2 = j2.pt();
      // promote this jet to 1st jet
      j1 = recocalojet->p4();
      jetPt1 = recocalojet->pt();
    } else if(recocalojet->pt() > jetPt2) {
      // promote this jet to 2nd jet
      j2 = recocalojet->p4();
      jetPt2 = recocalojet->pt();
    }
  }

  // apply DeltaEta cut
  double DeltaEta = fabs(j1.eta() - j2.eta());
  if(DeltaEta > maxDeltaEta_) return false;

  math::PtEtaPhiMLorentzVector fj1;
  math::PtEtaPhiMLorentzVector fj2;
  
  // apply radiation recovery
  for (CaloJetCollection::const_iterator recocalojet = recocalojets.begin();
       recocalojet != recocalojets.end(); recocalojet++) {
    double DeltaR1 = sqrt(pow(recocalojet->phi()-j1.phi(), 2.)+pow(recocalojet->eta()-j1.eta(),2.));
    double DeltaR2 = sqrt(pow(recocalojet->phi()-j2.phi(), 2.)+pow(recocalojet->eta()-j2.eta(),2.));
    if(DeltaR1 < DeltaR2 && DeltaR1 < fatJetDeltaR_) {
      fj1 += recocalojet->p4();
    } else if(DeltaR2 < fatJetDeltaR_) {
      fj2 += recocalojet->p4();
    }
  }

  // Apply mass cut
  fj1 += fj2;
  if(fj1.mass() < minMass_) return false;

  return true;
}

DEFINE_FWK_MODULE(HLTFatJetMassFilter);
