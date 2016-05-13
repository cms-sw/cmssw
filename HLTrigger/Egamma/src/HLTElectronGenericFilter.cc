/** \class HLTElectronGenericFilter
 *
 *
 *  \author Roberto Covarelli (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronGenericFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

//
// constructors and destructor
//
HLTElectronGenericFilter::HLTElectronGenericFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  candTag_      = iConfig.getParameter< edm::InputTag > ("candTag");
  varTag_       = iConfig.getParameter< edm::InputTag > ("varTag");
  lessThan_     = iConfig.getParameter<bool> ("lessThan");
  thrRegularEB_ = iConfig.getParameter<double> ("thrRegularEB");
  thrRegularEE_ = iConfig.getParameter<double> ("thrRegularEE");
  thrOverPtEB_  = iConfig.getParameter<double> ("thrOverPtEB");
  thrOverPtEE_  = iConfig.getParameter<double> ("thrOverPtEE");
  thrTimesPtEB_ = iConfig.getParameter<double> ("thrTimesPtEB");
  thrTimesPtEE_ = iConfig.getParameter<double> ("thrTimesPtEE");
  ncandcut_     = iConfig.getParameter<int> ("ncandcut");
  l1EGTag_      = iConfig.getParameter< edm::InputTag > ("l1EGCand");

  candToken_    =  consumes<trigger::TriggerFilterObjectWithRefs> (candTag_);
  varToken_     = consumes<reco::ElectronIsolationMap>(varTag_);
}

HLTElectronGenericFilter::~HLTElectronGenericFilter(){}

void
HLTElectronGenericFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag", edm::InputTag("hltSingleElectronOneOEMinusOneOPFilter"));
  desc.add<edm::InputTag>("varTag", edm::InputTag("hltSingleElectronTrackIsol"));
  desc.add<bool>("lessThan", true);
  desc.add<double>("thrRegularEB", 0.0);
  desc.add<double>("thrRegularEE", 0.0);
  desc.add<double>("thrOverPtEB", -1.0);
  desc.add<double>("thrOverPtEE", -1.0);
  desc.add<double>("thrTimesPtEB", -1.0);
  desc.add<double>("thrTimesPtEE", -1.0);
  desc.add<int>("ncandcut", 1);
  desc.add<edm::InputTag>("l1EGCand", edm::InputTag("hltPixelMatchElectronsL1Iso"));
  descriptions.add("hltElectronGenericFilter", desc);
}

// ------------ method called to produce the data  ------------
bool
HLTElectronGenericFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
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


  //get hold of isolated association map
  edm::Handle<reco::ElectronIsolationMap> depMap;
  iEvent.getByToken (varToken_,depMap);

  // look at all photons, check cuts and add to filter object
  int n = 0;

  for (unsigned int i=0; i<elecands.size(); i++) {

    ref = elecands[i];
    reco::ElectronIsolationMap::const_iterator mapi = (*depMap).find( ref );

    float vali = mapi->val;
    float Pt = ref->pt();
    float Eta = fabs(ref->eta());

    if ( lessThan_ ) {
      if ( (Eta < 1.479 && vali <= thrRegularEB_) || (Eta >= 1.479 && vali <= thrRegularEE_) ) {
	n++;
	filterproduct.addObject(TriggerElectron, ref);
	continue;
      }
      if (Pt > 0. && (thrOverPtEB_ > 0. || thrOverPtEE_ > 0. || thrTimesPtEB_ > 0. || thrTimesPtEE_ > 0.) ) {
	if ((Eta < 1.479 && vali/Pt <= thrOverPtEB_) || (Eta >= 1.479 && vali/Pt <= thrOverPtEE_) ) {
	  n++;
	  filterproduct.addObject(TriggerElectron, ref);
	  continue;
	}
	if ((Eta < 1.479 && vali*Pt <= thrTimesPtEB_) || (Eta >= 1.479 && vali*Pt <= thrTimesPtEE_) ) {
	  n++;
	  filterproduct.addObject(TriggerElectron, ref);
	}
      }
    } else {
      if ( (Eta < 1.479 && vali >= thrRegularEB_) || (Eta >= 1.479 && vali >= thrRegularEE_) ) {
	n++;
	filterproduct.addObject(TriggerElectron, ref);
	continue;
      }
      if (Pt > 0. && (thrOverPtEB_ > 0. || thrOverPtEE_ > 0. || thrTimesPtEB_ > 0. || thrTimesPtEE_ > 0.) ) {
	if ((Eta < 1.479 && vali/Pt >= thrOverPtEB_) || (Eta >= 1.479 && vali/Pt >= thrOverPtEE_) ) {
	  n++;
	  filterproduct.addObject(TriggerElectron, ref);
	  continue;
	}
	if ((Eta < 1.479 && vali*Pt >= thrTimesPtEB_) || (Eta >= 1.479 && vali*Pt >= thrTimesPtEE_) ) {
	  n++;
	  filterproduct.addObject(TriggerElectron, ref);
	}
      }
    }
  }

  // filter decision
  bool accept(n>=ncandcut_);

  return accept;
}

