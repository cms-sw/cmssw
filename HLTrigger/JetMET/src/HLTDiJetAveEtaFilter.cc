/** \class HLTDiJetAveEtaFilter
 *
 *
 *  \author Tomasz Fruboes
 *     based on HLTDiJetAveFilter
 */

#include "HLTrigger/JetMET/interface/HLTDiJetAveEtaFilter.h"

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

#include<typeinfo>

//
// constructors and destructor
//
template<typename T>
HLTDiJetAveEtaFilter<T>::HLTDiJetAveEtaFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
    inputJetTag_ (iConfig.template getParameter< edm::InputTag > ("inputJetTag")),
    minPtJet_    (iConfig.template getParameter<double> ("minPtJet")),
    minPtAve_    (iConfig.template getParameter<double> ("minPtAve")),
    //minPtJet3_   (iConfig.template getParameter<double> ("minPtJet3")),
    minDphi_     (iConfig.template getParameter<double> ("minDphi")),
    tagEtaMin_     (iConfig.template getParameter<double> ("minTagEta")),
    tagEtaMax_     (iConfig.template getParameter<double> ("maxTagEta")),
    probeEtaMin_     (iConfig.template getParameter<double> ("minProbeEta")),
    probeEtaMax_     (iConfig.template getParameter<double> ("maxProbeEta")),
    triggerType_ (iConfig.template getParameter<int> ("triggerType"))
{
    m_theJetToken = consumes<std::vector<T>>(inputJetTag_);
    LogDebug("") << "HLTDiJetAveEtaFilter: Input/minPtAve/minDphi/triggerType : "
        << inputJetTag_.encode() << " "
        << minPtAve_ << " "
        //<< minPtJet3_ << " "
        << minDphi_ << " "
        << triggerType_;
}

template<typename T>
HLTDiJetAveEtaFilter<T>::~HLTDiJetAveEtaFilter(){}

template<typename T>
void
HLTDiJetAveEtaFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    makeHLTFilterDescription(desc);
    desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltIterativeCone5CaloJets"));
    desc.add<double>("minPtAve",100.0);
    desc.add<double>("minPtJet",50.0);
    //desc.add<double>("minPtJet3",99999.0);
    desc.add<double>("minDphi",-1.0);
    desc.add<double>("minTagEta", -1.);
    desc.add<double>("maxTagEta", 1.4);
    desc.add<double>("minProbeEta", 2.7);
    desc.add<double>("maxProbeEta", 5.5);
    desc.add<int>("triggerType",trigger::TriggerJet);
    descriptions.add(std::string("hlt")+std::string(typeid(HLTDiJetAveEtaFilter<T>).name()),desc);
}

// ------------ method called to produce the data  ------------
template<typename T>
bool
HLTDiJetAveEtaFilter<T>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
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

    if(objects->size() > 1){ // events with two or more jets
        typename TCollection::const_iterator iTag ( objects->begin() );
        typename TCollection::const_iterator iEnd ( objects->end() );
        for (; iTag!=iEnd; ++iTag) {
            if (iTag->pt() < minPtJet_) continue;

            // for easier trigger efficiency evaluation save all tag/probe 
            // objects passing the minPT/eta criteria (outer loop)
            float eta = std::abs(iTag->eta());
            bool isGood = false; // tag or probe
            bool isTag = false;
            if ( eta > tagEtaMin_ && eta < tagEtaMax_ ){
                isGood = true;
                isTag =  true;
            }
            if ( eta > probeEtaMin_ && eta < probeEtaMax_ ){
                isGood = true;
            }
            if (isGood){
                filterproduct.addObject(triggerType_,  TRef(objects,distance(objects->begin(),iTag)));
            }

            if (!isTag) continue;

            typename TCollection::const_iterator iProbe ( iTag );
            ++iProbe;
            for (;iProbe != iEnd; ++iProbe){
                if (iProbe->pt() < minPtJet_) continue;
                float eta2 = std::abs(iProbe->eta());
                if ( eta2 < probeEtaMin_ || eta2 > probeEtaMax_ ) continue;
                double dphi = std::abs(deltaPhi(iTag->phi(),iProbe->phi() ));
                if (dphi<minDphi_) {    
                    continue;
                }

                double ptAve = (iTag->pt() + iProbe->pt())/2;
                if (ptAve<minPtAve_ ) {
                    continue;
                }
                ++n;
            }
        }
    } // events with two or more jets
    // filter decision
    bool accept(n>=1);
    return accept;
}
