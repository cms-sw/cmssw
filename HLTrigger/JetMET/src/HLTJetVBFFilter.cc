/** \class HLTJetVBFFilter
 *
 *
 *  \author Monica Vazquez Acosta (CERN)
 *  \modifier Phst Srimanobhas (srimanob@mail.cern.ch)
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
#include "HLTrigger/JetMET/interface/HLTJetVBFFilter.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

//
// constructors and destructor
//
template<typename T>
HLTJetVBFFilter<T>::HLTJetVBFFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  inputTag_       (iConfig.template getParameter< edm::InputTag > ("inputTag")),
  minPtLow_       (iConfig.template getParameter<double> ("minPtLow")),
  minPtHigh_      (iConfig.template getParameter<double> ("minPtHigh")),
  etaOpposite_    (iConfig.template getParameter<bool>   ("etaOpposite")),
  minDeltaEta_    (iConfig.template getParameter<double> ("minDeltaEta")),
  minInvMass_     (iConfig.template getParameter<double> ("minInvMass")),
  maxEta_         (iConfig.template getParameter<double> ("maxEta")),
  leadingJetOnly_ (iConfig.template getParameter<bool>   ("leadingJetOnly")),
  triggerType_    (iConfig.template getParameter<int> ("triggerType"))
{
  m_theObjectToken = consumes<std::vector<T>>(inputTag_);
  LogDebug("") << "HLTJetVBFFilter: Input/minPtLow_/minPtHigh_/triggerType : "
	       << inputTag_.encode() << " "
	       << minPtLow_  << " "
	       << minPtHigh_ << " "
	       << triggerType_;
}

template<typename T>
HLTJetVBFFilter<T>::~HLTJetVBFFilter(){}

template<typename T>
void
HLTJetVBFFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltAntiKT5ConvPFJets"));
  desc.add<double>("minPtLow",40.);
  desc.add<double>("minPtHigh",40.);
  desc.add<bool>("etaOpposite",false);
  desc.add<double>("minDeltaEta",4.0);
  desc.add<double>("minInvMass",1000.);
  desc.add<double>("maxEta",5.0);
  desc.add<bool>("leadingJetOnly",false);
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add(defaultModuleLabel<HLTJetVBFFilter<T>>(), desc);
}

//
// ------------ method called to produce the data  ------------
//
template<typename T>
bool
HLTJetVBFFilter<T>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
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
  iEvent.getByToken (m_theObjectToken,objects);

  // look at all candidates, check cuts and add to filter object
  int n(0);

  // events with two or more jets
  if(objects->size() > 1){

    double ejet1   = 0.;
    double pxjet1  = 0.;
    double pyjet1  = 0.;
    double pzjet1  = 0.;
    double ptjet1  = 0.;
    double etajet1 = 0.;

    double ejet2   = 0.;
    double pxjet2  = 0.;
    double pyjet2  = 0.;
    double pzjet2  = 0.;
    double ptjet2  = 0.;
    double etajet2 = 0.;

    // loop on all jets
    int countJet1(0);
    int countJet2(0);
    typename TCollection::const_iterator jet1 ( objects->begin() );
    for (; jet1!=objects->end(); jet1++) {
      countJet1++;
      if( leadingJetOnly_==true && countJet1>2 ) break;
      //
      if( jet1->pt() < minPtHigh_ ) break; //No need to go to the next jet (lower PT)
      if( std::abs(jet1->eta()) > maxEta_ ) continue;
      //
      countJet2 = countJet1-1;
      typename TCollection::const_iterator jet2 ( jet1+1 );
      for (; jet2!=objects->end(); jet2++) {
	countJet2++;
	if( leadingJetOnly_==true && countJet2>2 ) break;
	//
        if( jet2->pt() < minPtLow_ ) break; //No need to go to the next jet (lower PT)
	if( std::abs(jet2->eta()) > maxEta_ ) continue;
	//
        ejet1   = jet1->energy();
        pxjet1  = jet1->px();
        pyjet1  = jet1->py();
        pzjet1  = jet1->pz();
        ptjet1  = jet1->pt();
	etajet1 = jet1->eta();

        ejet2   = jet2->energy();
        pxjet2  = jet2->px();
        pyjet2  = jet2->py();
        pzjet2  = jet2->pz();
        ptjet2  = jet2->pt();
        etajet2 = jet2->eta();
        //
        float deltaetajet = etajet1 - etajet2;
        float invmassjet = sqrt( (ejet1  + ejet2)  * (ejet1  + ejet2) -
      	                         (pxjet1 + pxjet2) * (pxjet1 + pxjet2) -
                                 (pyjet1 + pyjet2) * (pyjet1 + pyjet2) -
                                 (pzjet1 + pzjet2) * (pzjet1 + pzjet2) );

        // VBF cuts
        if ( (ptjet1 > minPtHigh_) &&
	     (ptjet2 > minPtLow_) &&
             ( (etaOpposite_ == true && etajet1*etajet2 < 0) || (etaOpposite_ == false) ) &&
             (std::abs(deltaetajet) > minDeltaEta_) &&
	     (std::abs(invmassjet) > minInvMass_) ){
   	  ++n;
          TRef ref1 = TRef(objects,distance(objects->begin(),jet1));
	  TRef ref2 = TRef(objects,distance(objects->begin(),jet2));
          filterproduct.addObject(triggerType_,ref1);
          filterproduct.addObject(triggerType_,ref2);
        }// VBF cuts
	//if(n>=1) break; //Store all possible pairs
      }
      //if(n>=1) break; //Store all possible pairs
    }// loop on all jets
  }// events with two or more jets

  // filter decision
  bool accept(n>=1);

  return accept;
}
