//
// $Id: PATGenJetSlimmer.cc,v 1.1 2011/03/24 18:45:45 mwlebour Exp $
//

/**
  \class    pat::PATGenJetSlimmer PATGenJetSlimmer.h "PhysicsTools/PatAlgos/interface/PATGenJetSlimmer.h"
  \brief    Matcher of reconstructed objects to L1 Muons 
            
  \author   Giovanni Petrucciani
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/JetReco/interface/GenJet.h"

namespace pat {

  class PATGenJetSlimmer : public edm::EDProducer {
    public:
      explicit PATGenJetSlimmer(const edm::ParameterSet & iConfig);
      virtual ~PATGenJetSlimmer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      edm::EDGetTokenT<edm::View<reco::GenJet> > src_;
      StringCutObjectSelector<reco::GenJet> cut_;
      
      /// reset daughters to an empty vector
      bool clearDaughters_;
      /// drop the specific
      bool dropSpecific_;
  };

} // namespace

pat::PATGenJetSlimmer::PATGenJetSlimmer(const edm::ParameterSet & iConfig) :
    src_(consumes<edm::View<reco::GenJet> >(iConfig.getParameter<edm::InputTag>("src"))),
    cut_(iConfig.getParameter<std::string>("cut")),
    clearDaughters_(iConfig.getParameter<bool>("clearDaughters")),
    dropSpecific_(iConfig.getParameter<bool>("dropSpecific"))
{
    produces<std::vector<reco::GenJet> >();
}

void 
pat::PATGenJetSlimmer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<reco::GenJet> >      src;
    iEvent.getByToken(src_, src);

    auto_ptr<vector<reco::GenJet> >  out(new vector<reco::GenJet>());
    out->reserve(src->size());

    for (View<reco::GenJet>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        if (!cut_(*it)) continue;

        out->push_back(*it);
        reco::GenJet & jet = out->back();

        if (clearDaughters_) {
            jet.clearDaughters();
        }
        if (dropSpecific_) {
            jet.setSpecific( reco::GenJet::Specific() );
        }
    }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATGenJetSlimmer);
