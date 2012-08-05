//
// $Id: L1MuonMultiMatcher.cc,v 1.4 2011/03/31 09:59:33 gpetrucc Exp $
//

/**
  \class    pat::L1MuonMultiMatcher L1MuonMultiMatcher.h "MuonAnalysis/MuonAssociators/interface/L1MuonMultiMatcher.h"
  \brief    Matcher of reconstructed objects to L1 Muons 
            
  \author   Giovanni Petrucciani
  \version  $Id: L1MuonMultiMatcher.cc,v 1.4 2011/03/31 09:59:33 gpetrucc Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/View.h"

#include "MuonAnalysis/MuonAssociators/interface/L1MuonMatcherAlgo.h"

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace pat {

  class L1MuonMultiMatcher : public edm::EDProducer {
    public:
      explicit L1MuonMultiMatcher(const edm::ParameterSet & iConfig);
      virtual ~L1MuonMultiMatcher() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

      virtual void beginRun(edm::Run & iRun, const edm::EventSetup & iSetup);
    private:
      typedef std::pair<std::string, L1MuonMatcherAlgo> MatcherPair;
      std::vector<MatcherPair> matchers_;

      /// Labels for input collections
      edm::InputTag reco_, l1_;

      /// Store extra information in a ValueMap
      template<typename Hand, typename T>
      void storeExtraInfo(edm::Event &iEvent, 
                     const Hand & handle,
                     const std::vector<T> & values,
                     unsigned int           offset,
                     const std::string    & label) const ;
  };

} // namespace

pat::L1MuonMultiMatcher::L1MuonMultiMatcher(const edm::ParameterSet & iConfig) :
    reco_(iConfig.getParameter<edm::InputTag>("src")),
    l1_(iConfig.getParameter<edm::InputTag>("matched"))
{
    
    edm::ParameterSet cfgs = iConfig.getParameter<edm::ParameterSet>("otherMatchers");
    std::vector<std::string> names = cfgs.getParameterNamesForType<edm::ParameterSet>();
    matchers_.reserve(names.size()+1);
    matchers_.push_back(MatcherPair("", L1MuonMatcherAlgo(iConfig)));
    for (std::vector<std::string>::const_iterator it = names.begin(), ed = names.end(); it != ed; ++it) {
        matchers_.push_back(MatcherPair(*it, L1MuonMatcherAlgo(cfgs.getParameter<edm::ParameterSet>(*it))));
    }
    for (unsigned int j = 0; j < matchers_.size(); ++j) {
        const std::string &lbl = matchers_[j].first;
        produces<edm::ValueMap<reco::CandidatePtr> >(lbl);
        produces<edm::ValueMap<float> >("deltaR"+lbl);
        produces<edm::ValueMap<float> >("deltaPhi"+lbl);
        produces<edm::ValueMap<int  > >("quality"+lbl);
        produces<edm::ValueMap<int  > >("bx"+lbl);
        produces<edm::ValueMap<int  > >("isolated"+lbl);
    }
}

void 
pat::L1MuonMultiMatcher::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<reco::Candidate> > reco;
    Handle<vector<l1extra::L1MuonParticle> > l1s;

    iEvent.getByLabel(reco_, reco);
    iEvent.getByLabel(l1_, l1s);

    unsigned int n = reco->size(), m = matchers_.size();
    vector<float> deltaRs(n*m, 999), deltaPhis(n*m, 999);
    vector<int> bx(n*m, -999), quality(n*m, 0), isolated(n*m, 0);
    vector<reco::CandidatePtr> matches(n*m);
    for (unsigned int i = 0; i < n; ++i) {
        const reco::Candidate &mu = (*reco)[i];
        TrajectoryStateOnSurface propagated = matchers_[0].second.extrapolate(mu);
        if (!propagated.isValid()) continue;
        for (unsigned int j = 0; j < m; ++j) {
            int k = i + n*j;
            int match = matchers_[j].second.match(propagated, *l1s, deltaRs[k], deltaPhis[k]);
            if (match != -1) {
                const l1extra::L1MuonParticle & l1 = (*l1s)[match];
                const L1MuGMTCand & gmt = l1.gmtMuonCand();
                quality[k]  = gmt.quality();
                bx[k]       = gmt.bx();
                isolated[k] = gmt.isol();
                matches[k] = edm::Ptr<reco::Candidate>(l1s, size_t(match));
            }
        }
    }
    for (unsigned int j = 0; j < m; ++j) {
        const std::string &lbl = matchers_[j].first;
        storeExtraInfo(iEvent, reco, matches,      j*n,  lbl);
        storeExtraInfo(iEvent, reco, deltaRs,      j*n,  "deltaR"+lbl);
        storeExtraInfo(iEvent, reco, deltaPhis,    j*n,  "deltaPhi"+lbl);
        storeExtraInfo(iEvent, reco, quality,      j*n,  "quality"+lbl);
        storeExtraInfo(iEvent, reco, bx,           j*n,  "bx"+lbl);
        storeExtraInfo(iEvent, reco, isolated,     j*n,  "isolated"+lbl);
    }
}

template<typename Hand, typename T>
void
pat::L1MuonMultiMatcher::storeExtraInfo(edm::Event &iEvent,
                     const Hand & handle,
                     const std::vector<T> & values,
                     unsigned int           offs,
                     const std::string    & label) const {
    using namespace edm; using namespace std;
    auto_ptr<ValueMap<T> > valMap(new ValueMap<T>());
    typename edm::ValueMap<T>::Filler filler(*valMap);
    filler.insert(handle, values.begin()+offs, values.begin()+offs + handle->size());
    filler.fill();
    iEvent.put(valMap, label);
}


void 
pat::L1MuonMultiMatcher::beginRun(edm::Run & iRun, const edm::EventSetup & iSetup) {
    for (int i = 0, n = matchers_.size(); i < n; ++i) matchers_[i].second.init(iSetup);
}


#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(L1MuonMultiMatcher);
