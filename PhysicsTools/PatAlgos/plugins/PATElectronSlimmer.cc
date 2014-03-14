//
// $Id: PATElectronSlimmer.cc,v 1.1 2011/03/24 18:45:45 mwlebour Exp $
//

/**
  \class    pat::PATElectronSlimmer PATElectronSlimmer.h "PhysicsTools/PatAlgos/interface/PATElectronSlimmer.h"
  \brief    Matcher of reconstructed objects to L1 Muons 
            
  \author   Giovanni Petrucciani
  \version  $Id: PATElectronSlimmer.cc,v 1.1 2011/03/24 18:45:45 mwlebour Exp $
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#define protected public
#include "DataFormats/PatCandidates/interface/Electron.h"
#undef protected

namespace pat {

  class PATElectronSlimmer : public edm::EDProducer {
    public:
      explicit PATElectronSlimmer(const edm::ParameterSet & iConfig);
      virtual ~PATElectronSlimmer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      edm::InputTag src_;

      bool dropSuperClusters_, dropBasicClusters_, dropPFlowClusters_, dropPreshowerClusters_, dropRecHits_;
  };

} // namespace

pat::PATElectronSlimmer::PATElectronSlimmer(const edm::ParameterSet & iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    dropSuperClusters_(iConfig.getParameter<bool>("dropSuperCluster")),
    dropBasicClusters_(iConfig.getParameter<bool>("dropBasicClusters")),
    dropPFlowClusters_(iConfig.getParameter<bool>("dropPFlowClusters")),
    dropPreshowerClusters_(iConfig.getParameter<bool>("dropPreshowerClusters")),
    dropRecHits_(iConfig.getParameter<bool>("dropRecHits"))
{
    produces<std::vector<pat::Electron> >();
}

void 
pat::PATElectronSlimmer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<pat::Electron> >      src;
    iEvent.getByLabel(src_, src);

    auto_ptr<vector<pat::Electron> >  out(new vector<pat::Electron>());
    out->reserve(src->size());

    for (View<pat::Electron>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        out->push_back(*it);
        pat::Electron & electron = out->back();
        if (dropSuperClusters_) electron.superCluster_.clear();
	if (dropBasicClusters_) electron.basicClusters_.clear();
	if (dropSuperClusters_ || dropPFlowClusters_) electron.pflowSuperCluster_.clear();
	if (dropBasicClusters_ || dropPFlowClusters_) electron.pflowBasicClusters_.clear();
	if (dropPreshowerClusters_) electron.preshowerClusters_.clear();
	if (dropPreshowerClusters_ || dropPFlowClusters_) electron.pflowPreshowerClusters_.clear();
        if (dropRecHits_) electron.recHits_ = EcalRecHitCollection();
    }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATElectronSlimmer);
