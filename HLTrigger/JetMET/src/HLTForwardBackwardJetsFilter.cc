/** \class HLTForwardBackwardJetsFilter
 *
 * $Id: HLTForwardBackwardJetsFilter.cc,v 1.6 2012/01/21 14:57:01 fwyzard Exp $
 *
 *
 */

#include "HLTrigger/JetMET/interface/HLTForwardBackwardJetsFilter.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include<typeinfo>

//
// constructors and destructor
//
template<typename T>
HLTForwardBackwardJetsFilter<T>::HLTForwardBackwardJetsFilter(const edm::ParameterSet& iConfig) : 
  HLTFilter(iConfig),
  inputTag_ (iConfig.template getParameter< edm::InputTag > ("inputTag")),
  minPt_    (iConfig.template getParameter<double> ("minPt")),
  minEta_   (iConfig.template getParameter<double> ("minEta")), 
  maxEta_   (iConfig.template getParameter<double> ("maxEta")),
  triggerType_ (iConfig.template getParameter<int> ("triggerType"))
{
  LogDebug("") << "HLTForwardBackwardJetsFilter: Input/minPt/minEta/maxEta/triggerType : "
	       << inputTag_.encode() << " "
	       << minPt_ << " " 
	       << minEta_ << " "
	       << maxEta_ << " "
	       << triggerType_;
}

template<typename T>
HLTForwardBackwardJetsFilter<T>::~HLTForwardBackwardJetsFilter(){}

template<typename T>
void
HLTForwardBackwardJetsFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltIterativeCone5CaloJetsRegional"));
  desc.add<double>("minPt",15.0);
  desc.add<double>("minEta",3.0);
  desc.add<double>("maxEta",5.1);
  descriptions.add(std::string("hlt")+std::string(typeid(HLTForwardBackwardJetsFilter<T>).name()),desc);
}

// ------------ method called to produce the data  ------------
template<typename T>
bool
HLTForwardBackwardJetsFilter<T>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger; 

  typedef vector<T> TCollection;
  typedef Ref<TCollection> TRef;
  
  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputTag_);

  // get hold of collection of objects
  Handle<TCollection> objects;
  iEvent.getByLabel(inputTag_,objects);
  
  // look at all candidates,  check cuts and add to filter object
  unsigned int nplusjets(0);
  unsigned int nminusjets(0);
  
  if(objects->size() > 1){
    // events with two or more jets

    // look for jets satifying pt and eta cuts; first on the plus side, then the minus side
    typename TCollection::const_iterator jet ( objects->begin() );
    for (; jet!=objects->end(); jet++) {
      float ptjet  = jet->pt();
      float etajet = jet->eta();
      if( ptjet > minPt_ ){
	if ( etajet > minEta_ && etajet < maxEta_ ){
	  nplusjets++;
	  TRef ref = TRef(objects,distance(objects->begin(),jet));
	  filterproduct.addObject(static_cast<trigger::TriggerObjectType>(triggerType_),ref);
	}
      }
    }
    if (nplusjets > 0) {   
      typename TCollection::const_iterator jet ( objects->begin() );
      for (; jet!=objects->end(); jet++) {
	float ptjet  = jet->pt();
	float etajet = jet->eta();
	if( ptjet > minPt_ ){
	  if ( etajet < -minEta_ && etajet > -maxEta_ ){
	    nminusjets++;
	    TRef ref = TRef(objects,distance(objects->begin(),jet));
	    filterproduct.addObject(static_cast<trigger::TriggerObjectType>(triggerType_),ref);
	  }
	}
      }
    }
  } // events with two or more jets
  
  // filter decision
  bool accept(nplusjets>0 && nminusjets>0);  
  
  return accept;
}
