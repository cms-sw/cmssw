/** \class HLTForwardBackwardJetsFilter
 *
 * $Id: HLTForwardBackwardJetsFilter.cc,v 1.9 2012/03/01 16:26:28 gruen Exp $
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
  nNeg_     (iConfig.template getParameter<unsigned int>("nNeg")),
  nPos_     (iConfig.template getParameter<unsigned int>("nPos")),
  nTot_     (iConfig.template getParameter<unsigned int>("nTot")),
  triggerType_ (iConfig.template getParameter<int> ("triggerType"))
{
  LogDebug("") << "HLTForwardBackwardJetsFilter: Input/minPt/minEta/maxEta/triggerType : "
	       << inputTag_.encode() << " "
	       << minPt_ << " " 
	       << minEta_ << " "
	       << maxEta_ << " "
	       << nNeg_ << " "
	       << nPos_ << " "
	       << nTot_ << " "
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
  desc.add<unsigned int>("nNeg",1);
  desc.add<unsigned int>("nPos",1);
  desc.add<unsigned int>("nTot",0);
  desc.add<int>("triggerType",trigger::TriggerJet);
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
  unsigned int nPosJets(0);
  unsigned int nNegJets(0);
  
  typename TCollection::const_iterator jet;
  // look for jets satifying pt and eta cuts; first on the plus side, then the minus side

  for (jet=objects->begin(); jet!=objects->end(); jet++) {
    double ptjet  = jet->pt();
    double etajet = jet->eta();
    if( ptjet >= minPt_ ){
      if (( minEta_<= etajet) && (etajet <= maxEta_) ){
	nPosJets++;
	TRef ref = TRef(objects,distance(objects->begin(),jet));
	filterproduct.addObject(triggerType_,ref);
      }
      if ((-maxEta_<= etajet) && (etajet <=-minEta_) ){
	nNegJets++;
	TRef ref = TRef(objects,distance(objects->begin(),jet));
	filterproduct.addObject(triggerType_,ref);
      }
    }
  }
  
  // filter decision
  const bool accept(
		    ( nNegJets >= nNeg_ ) &&
		    ( nPosJets >= nPos_ ) &&
		    ((nNegJets+nPosJets) >= nTot_ )
		   );  
  
  return accept;
}
