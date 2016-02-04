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
HLTFatJetMassFilter::HLTFatJetMassFilter(const edm::ParameterSet& iConfig)
{
  inputJetTag_  = iConfig.getParameter< edm::InputTag > ("inputJetTag");
  saveTags_     = iConfig.getParameter<bool>("saveTags");
  minMass_      = iConfig.getParameter<double> ("minMass");
  fatJetDeltaR_ = iConfig.getParameter<double> ("fatJetDeltaR");
  maxDeltaEta_  = iConfig.getParameter<double> ("maxDeltaEta");

  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTFatJetMassFilter::~HLTFatJetMassFilter(){}

void HLTFatJetMassFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltMCJetCorJetIcone5HF07"));
  desc.add<bool>("saveTags",false);
  desc.add<double>("minMass",0.0);
  desc.add<double>("fatJetDeltaR",1.1);
  desc.add<double>("maxDeltaEta",10.0);
  descriptions.add("hltFatJetMassFilter",desc);
}

// ------------ method called to produce the data  ------------
bool
  HLTFatJetMassFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  // The filter object
  auto_ptr<trigger::TriggerFilterObjectWithRefs> filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if (saveTags_) filterobject->addCollectionTag(inputJetTag_);


  Handle<CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputJetTag_,recocalojets);

  // events with at least two jets
  if(recocalojets->size() < 2) return false;

  math::PtEtaPhiMLorentzVector j1(0.1, 0., 0., 0.);
  math::PtEtaPhiMLorentzVector j2(0.1, 0., 0., 0.);
  double jetPt1 = 0.;
  double jetPt2 = 0.;
  // look for the two highest-pT jet
  for (CaloJetCollection::const_iterator recocalojet = recocalojets->begin();
       recocalojet != recocalojets->end(); recocalojet++) {
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
  for (CaloJetCollection::const_iterator recocalojet = recocalojets->begin();
       recocalojet != recocalojets->end(); recocalojet++) {
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

  // put filter object into the Event
  iEvent.put(filterobject);

  return true;
}

DEFINE_FWK_MODULE(HLTFatJetMassFilter);
