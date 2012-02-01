/** \class HLTMonoJetFilter
*
*
*  \author Srimanobhas Phat
*
*/

#include "HLTrigger/JetMET/interface/HLTMonoJetFilter.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include<typeinfo>

//
// extract the candidate type
//
template<typename T, int Tid>
trigger::TriggerObjectType getObjectType(const T &) {
  return static_cast<trigger::TriggerObjectType>(Tid);
}

//
// constructors and destructor
//
template<typename T, int Tid>
HLTMonoJetFilter<T,Tid>::HLTMonoJetFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  inputJetTag_ (iConfig.template getParameter< edm::InputTag > ("inputJetTag")),
  max_PtSecondJet_ (iConfig.template getParameter<double> ("max_PtSecondJet")),
  max_DeltaPhi_ (iConfig.template getParameter<double> ("max_DeltaPhi"))
{
  LogDebug("") << "MonoJet: Input/maxPtSecondJet/maxDeltaPhi : "
	       << inputJetTag_.encode() << " "
	       << max_PtSecondJet_ << " " 
	       << max_DeltaPhi_ ;
}

template<typename T, int Tid>
HLTMonoJetFilter<T,Tid>::~HLTMonoJetFilter(){}

template<typename T, int Tid>
void 
HLTMonoJetFilter<T,Tid>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltAntiKT5ConvPFJets"));
  desc.add<double>("max_PtSecondJet",9999.);
  desc.add<double>("max_DeltaPhi",99.);
  descriptions.add(std::string("hlt")+std::string(typeid(HLTMonoJetFilter<T,Tid>).name()),desc);
}

//
// ------------ method called to produce the data  ------------
//
template<typename T, int Tid>
bool
HLTMonoJetFilter<T,Tid>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  typedef vector<T> TCollection;
  typedef Ref<TCollection> TRef;
  
  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputJetTag_);

  // Ref to Candidate object to be recorded in filter object
  TRef ref;

  // get hold of collection of objects
  Handle<TCollection> objects;
  iEvent.getByLabel (inputJetTag_,objects);

  // look at all candidates,  check cuts and add to filter object
  int n(0);

  if(objects->size() > 0){ 
    int countJet(0);
    double jet1Phi = 0.;
    double jet2Phi = 0.;
    double jet2Pt  = 0.;

    typename TCollection::const_iterator i ( objects->begin() );
    for (; i!=objects->end(); i++) {
      if(countJet==0){
	ref=TRef(objects,distance(objects->begin(),i));
	jet1Phi  = i->phi();
      }
      if(countJet==1){
	jet2Pt   = i->pt();
	jet2Phi  = i->phi();
      }
      countJet++;
      if(countJet>=2) break;
    }
  
    if(countJet==1){
      n=1;
    }
    else if(countJet>1 && jet2Pt<max_PtSecondJet_){
      n=1;
    }
    else if(countJet>1 && jet2Pt>=max_PtSecondJet_){
      double Dphi=fabs(deltaPhi(jet1Phi,jet2Phi));
      if(Dphi>=max_DeltaPhi_) n=-1;
      else n=1;
    }
    else{
      n=-1;
    }
  
    if(n==1){
      filterproduct.addObject(getObjectType<T, Tid>(*ref),ref);
    }
  }

  bool accept(n==1); 

  return accept;
}
