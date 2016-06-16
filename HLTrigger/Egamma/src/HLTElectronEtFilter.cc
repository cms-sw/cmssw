/** \class HLTElectronEtFilter
 *
 *
 *  \author Alessio Ghezzi
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronEtFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

//
// constructors and destructor
//
HLTElectronEtFilter::HLTElectronEtFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  candTag_   = iConfig.getParameter< edm::InputTag > ("candTag");
  EtEB_      = iConfig.getParameter<double> ("EtCutEB");
  EtEE_      = iConfig.getParameter<double> ("EtCutEE");

  ncandcut_  = iConfig.getParameter<int> ("ncandcut");

  l1EGTag_   = iConfig.getParameter< edm::InputTag > ("l1EGCand");

  candToken_ =  consumes<trigger::TriggerFilterObjectWithRefs> (candTag_);
}

HLTElectronEtFilter::~HLTElectronEtFilter(){}

void
HLTElectronEtFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag", edm::InputTag("hltElectronPixelMatchFilter"));
  desc.add<double>("EtCutEB", 0.0);
  desc.add<double>("EtCutEE", 0.0);
  desc.add<int>("ncandcut", 1);
  desc.add<edm::InputTag>("l1EGCand", edm::InputTag("hltL1IsoRecoEcalCandidate"));
  descriptions.add("hltElectronEtFilter", desc);
}

// ------------ method called to produce the data  ------------
bool HLTElectronEtFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace trigger;
  if (saveTags()) {
    filterproduct.addCollectionTag(l1EGTag_);
  }

  // Ref to Candidate object to be recorded in filter object
  reco::ElectronRef ref;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken (candToken_, PrevFilterOutput);

  std::vector<edm::Ref<reco::ElectronCollection> > elecands;
  PrevFilterOutput->getObjects(TriggerElectron, elecands);



  // look at all photons, check cuts and add to filter object
  int n = 0;

  for (unsigned int i=0; i<elecands.size(); i++) {

    ref = elecands[i];
    float Pt = ref->pt();
    float Eta = fabs(ref->eta());

    if ( (Eta < 1.479 && Pt > EtEB_) || (Eta >= 1.479 && Pt > EtEE_) ) {
      n++;
      filterproduct.addObject(TriggerElectron, ref);
    }

  }
  // filter decision
  bool accept(n>=ncandcut_);

  return accept;
}
