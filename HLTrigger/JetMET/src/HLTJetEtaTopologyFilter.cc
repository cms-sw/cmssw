/** \class HLTJetEtaTopologyFilter
 *
 *
 *  \author Tomasz Fruboes
 *     based on HLTDiJetAveFilter
 */

#include "HLTrigger/JetMET/interface/HLTJetEtaTopologyFilter.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include<typeinfo>

//
// constructors and destructor
//
template<typename T>
HLTJetEtaTopologyFilter<T>::HLTJetEtaTopologyFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
    inputJetTag_ (iConfig.template getParameter< edm::InputTag > ("inputJetTag")),
    m_theJetToken(consumes<std::vector<T>>(inputJetTag_)),
    minPtJet_    (iConfig.template getParameter<double> ("minPtJet")),
    //minPtJet3_   (iConfig.template getParameter<double> ("minPtJet3")),
    jetEtaMin_     (iConfig.template getParameter<double> ("minJetEta")),
    jetEtaMax_     (iConfig.template getParameter<double> ("maxJetEta")),
    applyAbsToJet_ (iConfig.template getParameter<bool> ("applyAbsToJet")),

    triggerType_ (iConfig.template getParameter<int> ("triggerType"))
{
    LogDebug("") << "HLTJetEtaTopologyFilter: Input/minDphi/triggerType : "
        << inputJetTag_.encode() << " "
        << triggerType_;
}

template<typename T>
HLTJetEtaTopologyFilter<T>::~HLTJetEtaTopologyFilter(){}

template<typename T>
void
HLTJetEtaTopologyFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    makeHLTFilterDescription(desc);
    desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltIterativeCone5CaloJets"));
    desc.add<double>("minPtJet",50.0);
    //desc.add<double>("minPtJet3",99999.0);
    desc.add<double>("minJetEta", -1.);
    desc.add<double>("maxJetEta", 1.4);
    desc.add<bool> ("applyAbsToJet", false),
    desc.add<int>("triggerType",trigger::TriggerJet);
    descriptions.add(defaultModuleLabel<HLTJetEtaTopologyFilter<T>>(), desc);
}

// ------------ method called to produce the data  ------------
template<typename T>
bool
HLTJetEtaTopologyFilter<T>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
    using namespace std;
    using namespace edm;
    using namespace reco;
    using namespace trigger;

    typedef vector<T> TCollection;
    typedef Ref<TCollection> TRef;

    // The filter object
    if (saveTags()) filterproduct.addCollectionTag(inputJetTag_);

    // get hold of collection of objects
    Handle<TCollection> objects;
    iEvent.getByToken (m_theJetToken,objects);

    int n(0);

    if(objects->size() > 0){ // events with two or more jets
        typename TCollection::const_iterator iEnd ( objects->end() );
        typename TCollection::const_iterator iJet ( objects->begin() );
        for (;iJet != iEnd; ++iJet){
            if (iJet->pt() < minPtJet_) continue;
            float eta = iJet->eta();
            if (applyAbsToJet_) {
                    eta = abs(eta);
            }

            if ( eta < jetEtaMin_ || eta > jetEtaMax_ ) continue;
            filterproduct.addObject(triggerType_,  TRef(objects,distance(objects->begin(),iJet)));


            ++n;
        }
    } // events with one or more jets
    bool accept(n>=1);
    return accept;
}
