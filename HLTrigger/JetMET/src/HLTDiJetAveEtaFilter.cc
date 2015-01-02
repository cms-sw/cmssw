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
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"


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
    descriptions.add(defaultModuleLabel<HLTDiJetAveEtaFilter<T>>(), desc);
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
        typename TCollection::const_iterator iProbe ( objects->begin() );
        typename TCollection::const_iterator iEnd ( objects->end() );
        for (; iProbe!=iEnd; ++iProbe) {
            if (iProbe->pt() < minPtJet_) continue;

            // for easier trigger efficiency evaluation save all probe/tag 
            // objects passing the minPT/eta criteria (outer loop)
            float eta = std::abs(iProbe->eta());
            bool isGood = false; // probe or tag
            bool isProbe = false;
            if ( eta > probeEtaMin_ && eta < probeEtaMax_ ){
                isGood = true;
                isProbe =  true;
            }
            if ( eta > tagEtaMin_ && eta < tagEtaMax_ ){
                isGood = true;
            }
            if (isGood){
                filterproduct.addObject(triggerType_,  TRef(objects,distance(objects->begin(),iProbe)));
            }

            if (!isProbe) continue;

            typename TCollection::const_iterator iTag ( objects->begin() );
            for (;iTag != iEnd; ++iTag){
                if (iTag==iProbe) continue;
                if (iTag->pt() < minPtJet_) continue;
                float eta2 = std::abs(iTag->eta());
                if ( eta2 < tagEtaMin_ || eta2 > tagEtaMax_ ) continue;
                double dphi = std::abs(deltaPhi(iProbe->phi(),iTag->phi() ));
                if (dphi<minDphi_) {    
                    continue;
                }

                double ptAve = (iProbe->pt() + iTag->pt())/2;
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
