/** \class HLTDiJetEtaTopologyFilter
 *
 *
 *  \author Tomasz Fruboes
 *     based on HLTDiJetAveFilter
 */

#include "HLTrigger/JetMET/interface/HLTDiJetEtaTopologyFilter.h"

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
HLTDiJetEtaTopologyFilter<T>::HLTDiJetEtaTopologyFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
    inputJetTag_ (iConfig.template getParameter< edm::InputTag > ("inputJetTag")),
    m_theJetToken(consumes<std::vector<T>>(inputJetTag_)),
    minPtAve_    (iConfig.template getParameter<double> ("minPtAve")),
    atLeastOneJetAbovePT_(iConfig.template getParameter<double> ("atLeastOneJetAbovePT")),
    minPtTag_(iConfig.template getParameter<double> ("minPtTag")),
    minPtProbe_(iConfig.template getParameter<double> ("minPtProbe")),
    minDphi_     (iConfig.template getParameter<double> ("minDphi")),
    tagEtaMin_     (iConfig.template getParameter<double> ("minTagEta")),
    tagEtaMax_     (iConfig.template getParameter<double> ("maxTagEta")),
    probeEtaMin_     (iConfig.template getParameter<double> ("minProbeEta")),
    probeEtaMax_     (iConfig.template getParameter<double> ("maxProbeEta")),
    applyAbsToTag_ (iConfig.template getParameter<bool> ("applyAbsToTag")),
    applyAbsToProbe_ (iConfig.template getParameter<bool> ("applyAbsToProbe")),
    oppositeEta_ (iConfig.template getParameter<bool> ("oppositeEta")),

    triggerType_ (iConfig.template getParameter<int> ("triggerType"))
{
    LogDebug("") << "HLTDiJetEtaTopologyFilter: Input/minDphi/triggerType : "
        << inputJetTag_.encode() << " "
        //<< minPtJet3_ << " "
        << minDphi_ << " "
        << triggerType_;
}

template<typename T>
HLTDiJetEtaTopologyFilter<T>::~HLTDiJetEtaTopologyFilter(){}

template<typename T>
void
HLTDiJetEtaTopologyFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    makeHLTFilterDescription(desc);
    desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltIterativeCone5CaloJets"));
    desc.add<double>("minPtAve",0.0);
    desc.add<double>("atLeastOneJetAbovePT",0.0)->setComment("At least one jet with pt above threshold");
    desc.add<double>("minPtTag",50.0)->setComment("pt requirement on tag jet");
    desc.add<double>("minPtProbe",50.0)->setComment("pt requirement on probe jet");
    //desc.add<double>("minPtJet3",99999.0);
    desc.add<double>("minDphi",-1.0);
    desc.add<double>("minTagEta", -1.);
    desc.add<double>("maxTagEta", 1.4);
    desc.add<double>("minProbeEta", 2.7);
    desc.add<double>("maxProbeEta", 5.5);
    desc.add<bool> ("applyAbsToTag", false),
    desc.add<bool> ("applyAbsToProbe", false),
    desc.add<bool> ("oppositeEta", false),
    desc.add<int>("triggerType",trigger::TriggerJet);
    descriptions.add(defaultModuleLabel<HLTDiJetEtaTopologyFilter<T> >(), desc);
}

// ------------ method called to produce the data  ------------
template<typename T>
bool
HLTDiJetEtaTopologyFilter<T>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
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
            if (iProbe->pt() < minPtProbe_) continue;

            // for easier trigger efficiency evaluation save all probe/tag 
            // objects passing the minPT/eta criteria (outer loop)
            float eta = iProbe->eta();
            float etaForProbeTest = eta;
            float etaForTagTest = eta;
            //applyAbsToTag_ (iConfig.template getParameter<bool> ("applyAbsToTag")),
            //applyAbsToProbe_ (iConfig.template getParameter<bool> ("applyAbsToProbe")),
            if (applyAbsToProbe_) {
                etaForProbeTest = abs(etaForProbeTest);
            }
            if (applyAbsToTag_) {
                etaForTagTest = abs(etaForTagTest);
            }

            bool isGood = false; // probe or tag
            bool isProbe = false;
            if (etaForProbeTest > probeEtaMin_ && etaForProbeTest < probeEtaMax_ ){
                isGood = true;
                isProbe =  true;
            }
            if (etaForTagTest > tagEtaMin_ && etaForTagTest < tagEtaMax_ ){
                isGood = true;
            }
            if (isGood){
                filterproduct.addObject(triggerType_,  TRef(objects,distance(objects->begin(),iProbe)));
            }

            if (!isProbe) continue;

            typename TCollection::const_iterator iTag ( objects->begin() );
            for (;iTag != iEnd; ++iTag){
                if (iTag==iProbe) continue;
                if (iTag->pt() < minPtTag_) continue;
                if (std::max(iTag->pt(), iProbe->pt())<atLeastOneJetAbovePT_  ) continue;


                float eta2 = iTag->eta();
                if (applyAbsToTag_) {
                        eta2 = abs(eta2);
                }

                if ( eta2 < tagEtaMin_ || eta2 > tagEtaMax_ ) continue;
                double dphi = std::abs(deltaPhi(iProbe->phi(),iTag->phi() ));
                if (dphi<minDphi_) {    
                    continue;
                }
                if (oppositeEta_ &&  etaForProbeTest*eta2 > 0  ) {
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
