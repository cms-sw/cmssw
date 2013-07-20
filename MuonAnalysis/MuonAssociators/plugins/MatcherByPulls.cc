// -*- C++ -*-
//
// Package:    MuonAnalysis/MuonAssociators
// Class:      MatcherByPulls
// 
/**\class pat::MatcherByPulls MatcherByPulls.cc MuonAnalysis/MuonAssociators/src/MatcherByPulls.cc

 Description: Matches RecoCandidates to (Gen)Particles by looking at the pulls fo the track parameters.
              Produces as output an edm::Association to GenParticles, and a ValueMap with the pull values.

 Implementation: uses MatcherByPullsAlgorithm
                 the module is in the pat namespace, but it doesn't have any PAT dependency.
*/
//
// Original Author:  Giovanni Petrucciani (SNS Pisa and CERN PH-CMG)
//         Created:  Sun Nov 16 16:14:09 CET 2008
// $Id: MatcherByPulls.cc,v 1.4 2013/02/27 20:42:45 wmtan Exp $
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "MuonAnalysis/MuonAssociators/interface/MatcherByPullsAlgorithm.h"

/*     ____ _                     _           _                 _   _             
 *    / ___| | __ _ ___ ___    __| | ___  ___| | __ _ _ __ __ _| |_(_) ___  _ __  
 *   | |   | |/ _` / __/ __|  / _` |/ _ \/ __| |/ _` | '__/ _` | __| |/ _ \| '_ \ 
 *   | |___| | (_| \__ \__ \ | (_| |  __/ (__| | (_| | | | (_| | |_| | (_) | | | |
 *    \____|_|\__,_|___/___/  \__,_|\___|\___|_|\__,_|_|  \__,_|\__|_|\___/|_| |_|
 */                                                                               
namespace pat {
    template<typename T>
    class MatcherByPulls : public edm::EDProducer {
        public:
            explicit MatcherByPulls(const edm::ParameterSet&);
            ~MatcherByPulls();

        private:
            virtual void produce(edm::Event&, const edm::EventSetup&) override;

            /// The RECO objects
            edm::InputTag src_;

            /// The MC objects to match against
            edm::InputTag matched_;

            /// Preselection cut on MC objects
            StringCutObjectSelector<reco::GenParticle>  mcSel_;

            MatcherByPullsAlgorithm algo_;
    };
}

/*     ____                _                   _             
 *    / ___|___  _ __  ___| |_ _ __ _   _  ___| |_ ___  _ __ 
 *   | |   / _ \| '_ \/ __| __| '__| | | |/ __| __/ _ \| '__|
 *   | |__| (_) | | | \__ \ |_| |  | |_| | (__| || (_) | |   
 *    \____\___/|_| |_|___/\__|_|   \__,_|\___|\__\___/|_|   
 *                                                           
 */  
template<typename T>
pat::MatcherByPulls<T>::MatcherByPulls(const edm::ParameterSet &iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    matched_(iConfig.getParameter<edm::InputTag>("matched")),
    mcSel_(iConfig.getParameter<std::string>("matchedSelector")),
    algo_(iConfig)
{
    produces<edm::Association<std::vector<reco::GenParticle> > >();
    produces<edm::ValueMap<float> >("pulls"); 
}

template<typename T>
pat::MatcherByPulls<T>::~MatcherByPulls()
{
}

/*    ____                _                
 *   |  _ \ _ __ ___   __| |_   _  ___ ___ 
 *   | |_) | '__/ _ \ / _` | | | |/ __/ _ \
 *   |  __/| | | (_) | (_| | |_| | (_|  __/
 *   |_|   |_|  \___/ \__,_|\__,_|\___\___|
 *                                         
 */  

template<typename T>
void
pat::MatcherByPulls<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    typedef std::vector<reco::GenParticle> MCColl;
    edm::Handle<edm::View<T> > src; 
    edm::Handle<MCColl> cands; 
    iEvent.getByLabel(src_,     src);
    iEvent.getByLabel(matched_, cands);

    std::vector<uint8_t> candGood(cands->size(),1);
    std::transform(cands->begin(), cands->end(), candGood.begin(), mcSel_);

    std::vector<int>   matches(src->size(),-1);
    std::vector<float> pulls(src->size(),  1e39);
    for (size_t i = 0, n = src->size(); i < n; ++i) {
        const T &tk = (*src)[i];
        std::pair<int,float> m = algo_.match(tk, *cands, candGood);
        matches[i] = m.first;
        pulls[i]   = m.second;
    }

    typedef edm::Association<MCColl> MCAsso;
    std::auto_ptr<MCAsso> matchesMap(new MCAsso(edm::RefProd<MCColl>(cands)));
    MCAsso::Filler matchesFiller(*matchesMap);
    matchesFiller.insert(src, matches.begin(), matches.end());
    matchesFiller.fill();
    iEvent.put(matchesMap);

    std::auto_ptr<edm::ValueMap<float> > pullsMap(new edm::ValueMap<float>());
    edm::ValueMap<float>::Filler pullsFiller(*pullsMap);
    pullsFiller.insert(src, pulls.begin(), pulls.end());
    pullsFiller.fill();
    iEvent.put(pullsMap, "pulls");
}

//define this as a plug-in

typedef pat::MatcherByPulls<reco::RecoCandidate> MatcherByPulls;
typedef pat::MatcherByPulls<reco::Track>         TrackMatcherByPulls;
DEFINE_FWK_MODULE(MatcherByPulls);
DEFINE_FWK_MODULE(TrackMatcherByPulls);
