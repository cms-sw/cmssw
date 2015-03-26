/** \class HLTExclDiJetFilter
 *
 *
 *  \author Leonard Apanasevich
 *
 */

#include <cmath>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "HLTrigger/JetMET/interface/HLTExclDiJetFilter.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"


//
// constructors and destructor
//
template<typename T>
HLTExclDiJetFilter<T>::HLTExclDiJetFilter(const edm::ParameterSet& iConfig) :
  HLTFilter(iConfig),
  inputJetTag_ (iConfig.template getParameter<edm::InputTag> ("inputJetTag")),
  caloTowerTag_(iConfig.template getParameter<edm::InputTag> ("caloTowerTag")),
  minPtJet_    (iConfig.template getParameter<double> ("minPtJet")),
  minHFe_      (iConfig.template getParameter<double> ("minHFe")),
  HF_OR_       (iConfig.template getParameter<bool> ("HF_OR")),
  triggerType_ (iConfig.template getParameter<int> ("triggerType"))
{
  m_theJetToken = consumes<std::vector<T>>(inputJetTag_);
  m_theCaloTowerCollectionToken = consumes<CaloTowerCollection>(caloTowerTag_);
  LogDebug("") << "HLTExclDiJetFilter: Input/minPtJet/minHFe/HF_OR/triggerType : "
	       << inputJetTag_.encode() << " "
	       << caloTowerTag_.encode() << " "
	       << minPtJet_ << " "
	       << minHFe_ << " "
	       << HF_OR_ << " "
	       << triggerType_;
}

template<typename T>
HLTExclDiJetFilter<T>::~HLTExclDiJetFilter(){}

template<typename T>
void
HLTExclDiJetFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltMCJetCorJetIcone5HF07"));
  desc.add<edm::InputTag>("caloTowerTag",edm::InputTag("hltTowerMakerForAll"));
  desc.add<double>("minPtJet",30.0);
  desc.add<double>("minHFe",50.0);
  desc.add<bool>("HF_OR",false);
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add(defaultModuleLabel<HLTExclDiJetFilter<T>>(), desc);
}

// ------------ method called to produce the data  ------------
template<typename T>
bool
HLTExclDiJetFilter<T>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  typedef vector<T> TCollection;
  typedef Ref<TCollection> TRef;

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputJetTag_);

  Handle<TCollection> recojets; //recojets can be any jet collections
  iEvent.getByToken(m_theJetToken,recojets);

  // look at all candidates,  check cuts and add to filter object
  int n(0);

  double ptjet1=0., ptjet2=0.;
  double phijet1=0., phijet2=0.;

  if(recojets->size() > 1){
    // events with two or more jets

    int countjets =0;

    TRef JetRef1,JetRef2;

    typename TCollection::const_iterator recojet ( recojets->begin() );
    for (;recojet<=(recojets->begin()+1); ++recojet) {
      //
      if(countjets==0) {
	ptjet1 = recojet->pt();
        phijet1 = recojet->phi();
	
	JetRef1 = TRef(recojets,distance(recojets->begin(),recojet));
      }
      //
      if(countjets==1) {
	ptjet2 = recojet->pt();
        phijet2 = recojet->phi();

	JetRef2 = TRef(recojets,distance(recojets->begin(),recojet));
      }
      //
      ++countjets;
    }

    if(ptjet1>minPtJet_ && ptjet2>minPtJet_ ){
      double Dphi=std::abs(phijet1-phijet2);
      if(Dphi>M_PI) Dphi=2.0*M_PI-Dphi;
      if(Dphi>0.5*M_PI) {
	filterproduct.addObject(triggerType_,JetRef1);
	filterproduct.addObject(triggerType_,JetRef2);
	++n;
      }
    } // pt(jet1,jet2) > minPtJet_

  } // events with two or more jets

  // calotowers
  bool hf_accept=false;

  if(n>0) {
     double ehfp(0.);
     double ehfm(0.);

     Handle<CaloTowerCollection> o;
     iEvent.getByToken(m_theCaloTowerCollectionToken,o);
//     if( o.isValid()) {
      for( CaloTowerCollection::const_iterator cc = o->begin(); cc != o->end(); ++cc ) {
       if(std::abs(cc->ieta())>28 && cc->energy()<4.0) continue;
        if(cc->ieta()>28)  ehfp+=cc->energy();  // HF+ energy
        if(cc->ieta()<-28) ehfm+=cc->energy();  // HF- energy
      }
 //    }

     bool hf_accept_and  = (ehfp<minHFe_) && (ehfm<minHFe_);
     bool hf_accept_or  = (ehfp<minHFe_) || (ehfm<minHFe_);

     hf_accept = HF_OR_ ? hf_accept_or : hf_accept_and;

  } // n>0


////////////////////////////////////////////////////////

// filter decision
  bool accept(n>0 && hf_accept);

  return accept;
}
