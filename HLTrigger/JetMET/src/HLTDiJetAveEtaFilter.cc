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

    // look at all candidates,  check cuts and add to filter object
    int n(0);

    //std::cout << "--- " << minPtJet_ << std::endl;
    if(objects->size() > 1){ // events with two or more jets
        std::map<int, TRef > tags; // since eta ranges can overlap
        std::map<int, TRef > probes; 
        typename TCollection::const_iterator i ( objects->begin() );
        typename TCollection::const_iterator iEnd ( objects->end() );
        int cnt = 0;
        for (; i!=iEnd; ++i) {
            ++cnt;

            if (i->pt() < minPtJet_) continue;
            float eta = std::abs(i->eta());
            bool isGood = false;
            if ( eta > tagEtaMin_ && eta < tagEtaMax_ ){
                //std::cout << "Tag: " << cnt << " " << eta << " " << i->pt() << " " << i->phi() << std::endl;
                tags[cnt] = TRef(objects,distance(objects->begin(),i));
                isGood = true;
            }
            if ( eta > probeEtaMin_ && eta < probeEtaMax_ ){
                //std::cout << "Probe: " << cnt << " " << eta << " " << i->pt() << " " << i->phi() <<  std::endl;
                probes[cnt] = TRef(objects,distance(objects->begin(),i));
                isGood = true;
            }
            if (isGood){ // for easier efficiency evaluation
                filterproduct.addObject(triggerType_,  TRef(objects,distance(objects->begin(),i)));
            }
        }
        if (probes.size() == 0) return false;

        typename std::map<int, TRef >::const_iterator iTag   = tags.begin();
        typename std::map<int, TRef >::const_iterator iTagE   = tags.end();
        typename std::map<int, TRef >::const_iterator iProbe = probes.begin();
        typename std::map<int, TRef >::const_iterator iProbeE = probes.end();
        //std::cout <<"Tags: " << tags.size() << std::endl;
        //std::cout <<"Probes: " << probes.size() << std::endl;
        for(;iTag != iTagE; ++iTag){
            iProbe = probes.begin();
            //std::cout << "With tag " << iTag -> first << std::endl;
            for(;iProbe != iProbeE; ++iProbe){
                //std::cout << "C " << iTag->first << " " << iProbe->first << std::endl;

                if (iTag->first == iProbe->first) {
                    //std::cout << "Skip same\n" ;
                    continue; // not the same jet
                }
                double dphi = std::abs(deltaPhi(iTag->second->phi(),iProbe->second->phi() ));
                if (dphi<minDphi_) {    
                    //std::cout << "skip dphi " << dphi << " < " << minDphi_ << std::endl;
                    continue;
                }
                double ptAve = (iTag->second->pt() + iProbe->second->pt())/2;
                if (ptAve<minPtAve_ ) {
                    //std::cout << "skip ave " << ptAve << " < " << minPtAve_ << std::endl;
                    continue;
                }
                //std::cout << "Good: " << ptAve << " " << dphi << std::endl;
                //std::cout << "  Tag: " << iTag->second->eta() << " " << iTag->second->pt() << std::endl;
                //std::cout << "  Probe: " << iProbe->second->eta() << " " << iProbe->second->pt() << std::endl;
                //filterproduct.addObject(triggerType_, iTag->second);
                //filterproduct.addObject(triggerType_, iProbe->second);
                ++n;
            }
        }
    } // events with two or more jets
    // filter decision
    bool accept(n>=1);
    //std::cout << "ACC: " << accept << std::endl;
    return accept;
}
