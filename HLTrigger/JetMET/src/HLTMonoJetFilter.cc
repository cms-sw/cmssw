/** \class HLTMonoJetFilter
*
*
*  \author Srimanobhas Phat
*
*/


#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "HLTrigger/JetMET/interface/HLTMonoJetFilter.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"


//
// constructors and destructor
//
template<typename T>
HLTMonoJetFilter<T>::HLTMonoJetFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  inputJetTag_    (iConfig.template getParameter< edm::InputTag > ("inputJetTag")),
  maxPtSecondJet_ (iConfig.template getParameter<double> ("maxPtSecondJet")),
  maxDeltaPhi_    (iConfig.template getParameter<double> ("maxDeltaPhi")),
  triggerType_    (iConfig.template getParameter<int> ("triggerType"))
{
  m_theObjectToken = consumes<std::vector<T>>(inputJetTag_);
  LogDebug("") << "HLTMonoJetFilter: Input/maxPtSecondJet/maxDeltaPhi/triggerType : "
	       << inputJetTag_.encode() << " "
	       << maxPtSecondJet_ << " " 
	       << maxDeltaPhi_ << " "
	       << triggerType_;
}

template<typename T>
HLTMonoJetFilter<T>::~HLTMonoJetFilter(){}

template<typename T>
void 
HLTMonoJetFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltAntiKT5ConvPFJets"));
  desc.add<double>("maxPtSecondJet",9999.);
  desc.add<double>("maxDeltaPhi",99.);
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add(defaultModuleLabel<HLTMonoJetFilter<T>>(), desc);
}

//
// ------------ method called to produce the data  ------------
//
template<typename T>
bool
HLTMonoJetFilter<T>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
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
  TRef ref1, ref2;

  // get hold of collection of objects
  Handle<TCollection> objects;
  iEvent.getByToken (m_theObjectToken,objects);

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
	ref1=TRef(objects,distance(objects->begin(),i));
	jet1Phi  = i->phi();
      }
      if(countJet==1){
	ref2=TRef(objects,distance(objects->begin(),i));
	jet2Pt   = i->pt();
	jet2Phi  = i->phi();
      }
      countJet++;
      if(countJet>=2) break;
    }
  
    if(countJet==1){
      n=1;
    }
    else if(countJet>1 && jet2Pt<maxPtSecondJet_){
      n=1;
    }
    else if(countJet>1 && jet2Pt>=maxPtSecondJet_){
      double Dphi=std::abs(deltaPhi(jet1Phi,jet2Phi));
      if(Dphi>=maxDeltaPhi_) n=-1;
      else n=1;
    }
    else{
      n=-1;
    }
  
    if(n==1){
      filterproduct.addObject(triggerType_,ref1);
      if(countJet>1) filterproduct.addObject(triggerType_,ref2);
    }
  }

  bool accept(n==1); 

  return accept;
}
