#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "PhysicsTools/PatAlgos/interface/ObjectModifier.h"

#include <memory>

namespace pat {

  template <class T>
  class ModifiedObjectProducer : public edm::stream::EDProducer<> {
  public:
    typedef std::vector<T> Collection;
    typedef pat::ObjectModifier<T> Modifier;

    ModifiedObjectProducer(const edm::ParameterSet& conf) {
      //set our input source
      src_ = consumes<edm::View<T> >(conf.getParameter<edm::InputTag>("src"));
      //setup modifier
      const edm::ParameterSet& mod_config = conf.getParameter<edm::ParameterSet>("modifierConfig");
      modifier_ = std::make_unique<Modifier>(mod_config, consumesCollector());
      //declare products
      produces<Collection>();
    }
    ~ModifiedObjectProducer() override {}

    void produce(edm::Event& evt, const edm::EventSetup& evs) final {
      modifier_->setEventContent(evs);

      auto output = std::make_unique<Collection>();

      auto input = evt.getHandle(src_);
      output->reserve(input->size());

      modifier_->setEvent(evt);

      for (auto const& itr : *input) {
        output->push_back(itr);
        T& obj = output->back();
        modifier_->modify(obj);
      }

      evt.put(std::move(output));
    }

  private:
    edm::EDGetTokenT<edm::View<T> > src_;
    std::unique_ptr<Modifier> modifier_;
  };
}  // namespace pat

typedef pat::ModifiedObjectProducer<reco::GsfElectron> ModifiedGsfElectronProducer;
typedef pat::ModifiedObjectProducer<pat::Electron> ModifiedElectronProducer;
typedef pat::ModifiedObjectProducer<pat::Photon> ModifiedPhotonProducer;
typedef pat::ModifiedObjectProducer<pat::Muon> ModifiedMuonProducer;
typedef pat::ModifiedObjectProducer<pat::Tau> ModifiedTauProducer;
typedef pat::ModifiedObjectProducer<pat::Jet> ModifiedJetProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ModifiedGsfElectronProducer);
DEFINE_FWK_MODULE(ModifiedElectronProducer);
DEFINE_FWK_MODULE(ModifiedPhotonProducer);
DEFINE_FWK_MODULE(ModifiedMuonProducer);
DEFINE_FWK_MODULE(ModifiedTauProducer);
DEFINE_FWK_MODULE(ModifiedJetProducer);
