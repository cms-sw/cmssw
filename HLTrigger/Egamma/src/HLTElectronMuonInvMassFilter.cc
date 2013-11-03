/** \class HLTElectronMuonInvMassFilter
 *
 *  Original Author: Massimiliano Chiorboli
 *  Institution: INFN, Italy
 *  Contact: Massimiliano.Chiorboli@cern.ch
 *  Date: July 6, 2011
 */

#include "HLTrigger/Egamma/interface/HLTElectronMuonInvMassFilter.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
//
// constructors and destructor
//
HLTElectronMuonInvMassFilter::HLTElectronMuonInvMassFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
  eleCandTag_             = iConfig.getParameter< edm::InputTag > ("elePrevCandTag");
  muonCandTag_            = iConfig.getParameter< edm::InputTag > ("muonPrevCandTag");
  lowerMassCut_           = iConfig.getParameter<double> ("lowerMassCut");
  upperMassCut_           = iConfig.getParameter<double> ("upperMassCut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  relaxed_ = iConfig.getUntrackedParameter<bool> ("electronRelaxed",true) ;
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("ElectronL1IsoCand");
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("ElectronL1NonIsoCand");
  MuonCollTag_= iConfig.getParameter< edm::InputTag > ("MuonCand");
  eleCandToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(eleCandTag_);
  muonCandToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(muonCandTag_);
}


HLTElectronMuonInvMassFilter::~HLTElectronMuonInvMassFilter(){}

void
HLTElectronMuonInvMassFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("elePrevCandTag",edm::InputTag("hltL1NonIsoHLTCaloIdTTrkIdVLSingleElectronEt8NoCandDphiFilter"));
  desc.add<edm::InputTag>("muonPrevCandTag",edm::InputTag("hltL1Mu0HTT50L3Filtered3"));
  desc.add<double>("lowerMassCut",4.0);
  desc.add<double>("upperMassCut",999999.0);
  desc.add<int>("ncandcut",1);
  desc.addUntracked<bool>("electronRelaxed",true);
  desc.add<edm::InputTag>("ElectronL1IsoCand",edm::InputTag("hltPixelMatchElectronsActivity"));
  desc.add<edm::InputTag>("ElectronL1NonIsoCand",edm::InputTag("hltPixelMatchElectronsActivity"));
  desc.add<edm::InputTag>("MuonCand",edm::InputTag("hltL3MuonCandidates"));
  descriptions.add("hltElectronMuonInvMassFilter",desc);
}

// ------------ method called to produce the data  ------------
bool
HLTElectronMuonInvMassFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  // The filter object
  using namespace trigger;

  double const MuMass = 0.106;
  double const MuMass2 = MuMass*MuMass;

  if (saveTags()) {
    filterproduct.addCollectionTag(L1IsoCollTag_);
    if (relaxed_) filterproduct.addCollectionTag(L1NonIsoCollTag_);
    filterproduct.addCollectionTag(MuonCollTag_);
  }

  edm::Handle<trigger::TriggerFilterObjectWithRefs> EleFromPrevFilter;
  iEvent.getByToken (eleCandToken_,EleFromPrevFilter);

  edm::Handle<trigger::TriggerFilterObjectWithRefs> MuonFromPrevFilter;
  iEvent.getByToken (muonCandToken_,MuonFromPrevFilter);

  std::vector<TLorentzVector> pElectron;
  std::vector<double> eleCharge;

  std::vector<TLorentzVector> pMuon;
  std::vector<double> muonCharge;

  Ref< ElectronCollection > refele;
  vector< Ref< ElectronCollection > > electrons;
  EleFromPrevFilter->getObjects(TriggerElectron, electrons);

  vector<RecoChargedCandidateRef> l3muons;
  MuonFromPrevFilter->getObjects(TriggerMuon,l3muons);

  for(unsigned int i=0; i<l3muons.size(); i++) {
    TrackRef tk = l3muons[i]->get<TrackRef>();
    //     TrackRef tk = l3muons[i].track();
    double muonEnergy = sqrt(tk->momentum().Mag2()+MuMass2);
    TLorentzVector pThisMuon(tk->px(), tk->py(),
			     tk->pz(), muonEnergy );
    pMuon.push_back( pThisMuon );
    muonCharge.push_back( tk->charge() );
  }

  for (unsigned int i=0; i<electrons.size(); i++) {
    refele = electrons[i];
    TLorentzVector pThisEle(refele->px(), refele->py(),
			    refele->pz(), refele->energy() );
    pElectron.push_back( pThisEle );
    eleCharge.push_back( refele->charge() );
  }

  int nEleMuPairs = 0;
  for(unsigned int i=0; i<electrons.size(); i++) {
    for(unsigned int j=0; j<l3muons.size(); j++) {
      TLorentzVector p1 = pElectron.at(i);
      TLorentzVector p2 = pMuon.at(j);
      TLorentzVector pTot = p1 + p2;
      double mass = pTot.M();
      if(mass>=lowerMassCut_ && mass<=upperMassCut_){
	nEleMuPairs++;
	filterproduct.addObject(TriggerElectron, electrons[i]);
	filterproduct.addObject(TriggerMuon, l3muons[j]);
      }
    }
  }

  // filter decision
  bool accept(nEleMuPairs>=ncandcut_);
  return accept;
}
