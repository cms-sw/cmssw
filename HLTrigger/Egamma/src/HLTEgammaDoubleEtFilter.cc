/** \class HLTEgammaDoubleEtFilter
 *
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaDoubleEtFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

class EgammaHLTEtSortCriterium{
public:
  bool operator() (edm::Ref<reco::RecoEcalCandidateCollection> lhs, edm::Ref<reco::RecoEcalCandidateCollection> rhs){
    return lhs->et() > rhs->et();
  }
};

//
// constructors and destructor
//
HLTEgammaDoubleEtFilter::HLTEgammaDoubleEtFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
  candTag_   = iConfig.getParameter< edm::InputTag > ("candTag");
  etcut1_    = iConfig.getParameter<double> ("etcut1");
  etcut2_    = iConfig.getParameter<double> ("etcut2");
  npaircut_  = iConfig.getParameter<int> ("npaircut");
  l1EGTag_   = iConfig.getParameter< edm::InputTag > ("l1EGCand");
  candToken_ = consumes<trigger::TriggerFilterObjectWithRefs> (candTag_);
}

HLTEgammaDoubleEtFilter::~HLTEgammaDoubleEtFilter(){}

void
HLTEgammaDoubleEtFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
   edm::ParameterSetDescription desc;
   makeHLTFilterDescription(desc);
   desc.add<edm::InputTag>("candTag", edm::InputTag("hltTrackIsolFilter"));
   desc.add<edm::InputTag>("l1EGCand", edm::InputTag("hltL1IsoRecoEcalCandidate"));
   desc.add<double>("etcut1", 30.0);
   desc.add<double>("etcut2", 20.0);
   desc.add<int>("npaircut", 1);
   descriptions.add("hltEgammaDoubleEtFilter", desc);
}

// ------------ method called to produce the data  ------------
bool
HLTEgammaDoubleEtFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace trigger;

  // The filter object
  if (saveTags()) {
    filterproduct.addCollectionTag(l1EGTag_);
  }
  // Ref to Candidate object to be recorded in filter object
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken (candToken_, PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> >  mysortedrecoecalcands;
  PrevFilterOutput->getObjects(TriggerPhoton,  mysortedrecoecalcands);
  if(mysortedrecoecalcands.empty()) PrevFilterOutput->getObjects(TriggerCluster,mysortedrecoecalcands);  //we dont know if its type trigger cluster or trigger photon

  // look at all candidates,  check cuts and add to filter object
  int n(0);

  // Sort the list
  std::sort(mysortedrecoecalcands.begin(), mysortedrecoecalcands.end(), EgammaHLTEtSortCriterium());
  edm::Ref<reco::RecoEcalCandidateCollection> ref1, ref2;
  for (unsigned int i=0; i<mysortedrecoecalcands.size(); i++) {
    ref1 = mysortedrecoecalcands[i];
    if( ref1->et() >= etcut1_){
      for (unsigned int j=i+1; j<mysortedrecoecalcands.size(); j++) {
	ref2 = mysortedrecoecalcands[j];
	if( ref2->et() >= etcut2_ ){
	  filterproduct.addObject(TriggerPhoton, ref1);
	  filterproduct.addObject(TriggerPhoton, ref2);
	  n++;
	}
      }
    }
  }


  // filter decision
  bool accept(n>=npaircut_);

  return accept;
}



