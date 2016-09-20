/** \class HLTGenericFilter
 *
 *
 *  \author Roberto Covarelli (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTGenericFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

//
// constructors and destructor
//
template<typename T1>
HLTGenericFilter<T1>::HLTGenericFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
    candTag_   = iConfig.template getParameter< edm::InputTag > ("candTag");
    varTag_    = iConfig.template getParameter< edm::InputTag > ("varTag");
    l1EGTag_   = iConfig.template getParameter< edm::InputTag > ("l1EGCand");
    
    lessThan_        = iConfig.template getParameter<bool> ("lessThan");
    useEt_           = iConfig.template getParameter<bool> ("useEt");
    thrRegularEB_    = iConfig.template getParameter<double> ("thrRegularEB");
    thrRegularEE_    = iConfig.template getParameter<double> ("thrRegularEE");
    thrOverEEB_      = iConfig.template getParameter<double> ("thrOverEEB");
    thrOverEEE_      = iConfig.template getParameter<double> ("thrOverEEE");
    thrOverE2EB_     = iConfig.template getParameter<double> ("thrOverE2EB");
    thrOverE2EE_     = iConfig.template getParameter<double> ("thrOverE2EE");
    ncandcut_        = iConfig.template getParameter<int> ("ncandcut");
    
    candToken_ = consumes<trigger::TriggerFilterObjectWithRefs> (candTag_);
    varToken_  = consumes<T1IsolationMap> (varTag_);
}

template<typename T1>
void
HLTGenericFilter<T1>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    makeHLTFilterDescription(desc);
    desc.add<edm::InputTag>("candTag", edm::InputTag("hltSingleEgammaEtFilter"));
    desc.add<edm::InputTag>("varTag", edm::InputTag("hltSingleEgammaHcalIsol"));
    desc.add<bool>("lessThan", true);
    desc.add<bool>("useEt", false);
    desc.add<double>("thrRegularEB", 0.0);
    desc.add<double>("thrRegularEE", 0.0);
    desc.add<double>("thrOverEEB", -1.0);
    desc.add<double>("thrOverEEE", -1.0);
    desc.add<double>("thrOverE2EB", -1.0);
    desc.add<double>("thrOverE2EE", -1.0);
    desc.add<int>("ncandcut", 1);
    desc.add<edm::InputTag>("l1EGCand", edm::InputTag("hltL1IsoRecoEcalCandidate"));
    descriptions.add(defaultModuleLabel<HLTGenericFilter<T1>>(), desc);
}

template<typename T1>
HLTGenericFilter<T1>::~HLTGenericFilter(){}

template<typename T1>
float HLTGenericFilter<T1>::getEnergy(T1Ref candRef) const{
    return candRef->p();
}

template<>
float HLTGenericFilter<reco::RecoEcalCandidate>::getEnergy(T1Ref candRef) const{
    return candRef->superCluster()->energy();
}

template<typename T1>
float HLTGenericFilter<T1>::getEt(T1Ref candRef) const{
    return candRef->pt();
}

template<>
float HLTGenericFilter<reco::RecoEcalCandidate>::getEt(T1Ref candRef) const{
    return candRef->superCluster()->energy() * sin (2*atan(exp(-candRef->eta())));
}


// ------------ method called to produce the data  ------------
template<typename T1>
bool
HLTGenericFilter<T1>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
    using namespace trigger;
    if (saveTags()) {
        filterproduct.addCollectionTag(l1EGTag_);
    }
    
    
    
    // Set output format
    int trigger_type = trigger::TriggerCluster;
    if (saveTags()) trigger_type = trigger::TriggerPhoton;
    
    edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
    iEvent.getByToken (candToken_, PrevFilterOutput);
    
    std::vector<T1Ref> recoCands;
    PrevFilterOutput->getObjects(TriggerCluster, recoCands);
    if(recoCands.empty()) PrevFilterOutput->getObjects(TriggerPhoton,recoCands);  //we dont know if its type trigger cluster or trigger photon
    if(recoCands.empty()) {
        PrevFilterOutput->getObjects(TriggerMuon,recoCands);  //if not a cluster and not a photon then assum it is a muon
        trigger_type = trigger::TriggerMuon;
    }
    //get hold of isolated association map
    edm::Handle<T1IsolationMap> depMap;
    iEvent.getByToken (varToken_,depMap);
    
    // look at all photons, check cuts and add to filter object
    int n = 0;
    
    for (unsigned int i=0; i<recoCands.size(); i++) {
        
        
        // Ref to Candidate object to be recorded in filter object
        T1Ref ref = recoCands[i];
        typename T1IsolationMap::const_iterator mapi = (*depMap).find( ref );
        
        float vali = mapi->val;
        float EtaSC = ref->eta();
        float energy;
        if (useEt_) energy = getEt (ref);
        else energy = getEnergy( ref );
        
        if ( lessThan_ ) {
            if ( (std::abs(EtaSC) < 1.479 && vali <= thrRegularEB_) || (std::abs(EtaSC) >= 1.479 && vali <= thrRegularEE_) ) {
                n++;
                filterproduct.addObject(trigger_type, ref);
                continue;
            }
            if (energy > 0. && (thrOverEEB_ > 0. || thrOverEEE_ > 0. || thrOverE2EB_ > 0. || thrOverE2EE_ > 0.) ) {
                if ((std::abs(EtaSC) < 1.479 && vali/energy <= thrOverEEB_) || (std::abs(EtaSC) >= 1.479 && vali/energy <= thrOverEEE_) ) {
                    n++;
                    filterproduct.addObject(trigger_type, ref);
                    continue;
                }
                if ((std::abs(EtaSC) < 1.479 && vali/(energy*energy) <= thrOverE2EB_) || (std::abs(EtaSC) >= 1.479 && vali/(energy*energy) <= thrOverE2EE_) ) {
                    n++;
                    filterproduct.addObject(trigger_type, ref);
                }
            }
        } else {
            if ( (std::abs(EtaSC) < 1.479 && vali >= thrRegularEB_) || (std::abs(EtaSC) >= 1.479 && vali >= thrRegularEE_) ) {
                n++;
                filterproduct.addObject(trigger_type, ref);
                continue;
            }
            if (energy > 0. && (thrOverEEB_ > 0. || thrOverEEE_ > 0. || thrOverE2EB_ > 0. || thrOverE2EE_ > 0.) ) {
                if ((std::abs(EtaSC) < 1.479 && vali/energy >= thrOverEEB_) || (std::abs(EtaSC) >= 1.479 && vali/energy >= thrOverEEE_) ) {
                    n++;
                    filterproduct.addObject(trigger_type, ref);
                    continue;
                }
                if ((std::abs(EtaSC) < 1.479 && vali/(energy*energy) >= thrOverE2EB_) || (std::abs(EtaSC) >= 1.479 && vali/(energy*energy) >= thrOverE2EE_) ) {
                    n++;
                    filterproduct.addObject(trigger_type, ref);
                }
            }
        }
    }
    
    // filter decision
    bool accept(n>=ncandcut_);
    
    return accept;
}

typedef HLTGenericFilter<reco::RecoEcalCandidate> HLTEgammaGenericFilter;
typedef HLTGenericFilter<reco::RecoChargedCandidate> HLTMuonGenericFilter;
DEFINE_FWK_MODULE(HLTEgammaGenericFilter);
DEFINE_FWK_MODULE(HLTMuonGenericFilter);

