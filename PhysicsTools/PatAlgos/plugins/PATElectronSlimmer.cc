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
      
      /// clear mElectronArea, mPassNumber, mPileupEnergy
      bool clearElectronVars_;
      /// reset daughters to an empty vector
      bool clearDaughters_;
      bool clearTrackRefs_;
//       /// reduce GenElectron to a bare 4-vector
//       bool slimGenElectron_;
      /// drop the Calo or PF specific
      bool dropSpecific_;
//       /// drop the ElectronCorrFactors (but keep the electron corrected!)
//       bool dropElectronCorrFactors_;
  };

} // namespace

pat::PATElectronSlimmer::PATElectronSlimmer(const edm::ParameterSet & iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    clearElectronVars_(iConfig.getParameter<bool>("clearElectronVars")),
    clearDaughters_(iConfig.getParameter<bool>("clearDaughters")),
    clearTrackRefs_(iConfig.getParameter<bool>("clearTrackRefs")),
//     slimGenElectron_(iConfig.getParameter<bool>("slimGenElectron")),
    dropSpecific_(iConfig.getParameter<bool>("dropSpecific"))
//     dropElectronCorrFactors_(iConfig.getParameter<bool>("dropElectronCorrFactors"))
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
	electron.superCluster_.clear();
	electron.basicClusters_.clear();
	electron.pflowSuperCluster_.clear();
	electron.pflowBasicClusters_.clear();
	electron.recHits_=EcalRecHitCollection();
    }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATElectronSlimmer);
