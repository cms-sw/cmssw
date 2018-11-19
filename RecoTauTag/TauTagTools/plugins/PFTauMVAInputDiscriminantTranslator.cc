/*
 * PFTauMVAInputDiscriminantTranslator
 *
 * Translate a list of given MVA (i.e. TaNC)
 * variables into standard PFTauDiscriminators
 * to facilitate embeddeing them into pat::Taus
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */

#include <memory>
#include <string>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/MVAComputer/interface/Variable.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "RecoTauTag/RecoTau/interface/RecoTauDiscriminantPlugins.h"

using namespace reco;

class PFTauMVAInputDiscriminantTranslator : public edm::EDProducer {
  public:
    struct DiscriminantInfo {
      PhysicsTools::AtomicId name;
      std::string collName;
      size_t index;
      float defaultValue;
      std::shared_ptr<reco::tau::RecoTauDiscriminantPlugin> plugin;
    };

    PFTauMVAInputDiscriminantTranslator(const edm::ParameterSet&);
    void produce(edm::Event&, const edm::EventSetup&) override;

  private:
    edm::InputTag pfTauSource_;
    std::vector<DiscriminantInfo> discriminators_;
};

PFTauMVAInputDiscriminantTranslator::PFTauMVAInputDiscriminantTranslator(
    const edm::ParameterSet& pset) {
  typedef std::vector<edm::ParameterSet> VPSet;
  pfTauSource_ = pset.getParameter<edm::InputTag>("pfTauSource");
  VPSet discriminants = pset.getParameter<VPSet>("discriminants");

  for(VPSet::const_iterator iDisc = discriminants.begin();
      iDisc != discriminants.end(); ++iDisc) {
    // WTF IS GOING ON HERE
    std::string name = iDisc->getParameter<std::string>("name");
    double defaultValue = (iDisc->exists("default")) ?
        iDisc->getParameter<double>("default") : 0.;
    // check if we are getting multiple indices
    bool requestMultiple = iDisc->exists("indices");
    if(requestMultiple) {
      // make a discrimiantor for each desired index
      std::vector<uint32_t> indices =
          iDisc->getParameter<std::vector<uint32_t> >("indices");
      for(std::vector<uint32_t>::const_iterator index = indices.begin();
          index != indices.end(); ++index) {
        DiscriminantInfo newDisc;
        newDisc.name = name;
        newDisc.index = *index;
        newDisc.defaultValue = defaultValue;
        // make a nice colleciton name
        std::stringstream collectionName;
        collectionName << name << *index;
        newDisc.collName = collectionName.str();
        // Build the plugin
        edm::ParameterSet fakePSet;
        newDisc.plugin.reset(
            RecoTauDiscriminantPluginFactory::get()->create(
                reco::tau::discPluginName(name), fakePSet));
        discriminators_.push_back(newDisc);
      }
    } else {
      //single discriminant
      DiscriminantInfo newDisc;
      newDisc.name = name;
      newDisc.collName = name;
      newDisc.index = 0;
      newDisc.defaultValue = defaultValue;
      // Build the plugin
      edm::ParameterSet fakePSet;
      newDisc.plugin.reset(
          RecoTauDiscriminantPluginFactory::get()->create(
              reco::tau::discPluginName(name), fakePSet));
      discriminators_.push_back(newDisc);
    }
  }
  // register products
  for(auto const& disc : discriminators_) {
    produces<PFTauDiscriminator>(disc.collName);
  }
}

void PFTauMVAInputDiscriminantTranslator::produce(edm::Event& evt,
                                                  const edm::EventSetup& es) {
  // Handle to get PFTaus to associated to
  edm::Handle<PFTauCollection> pfTaus;
  evt.getByLabel(pfTauSource_, pfTaus);

  for(auto const& disc : discriminators_) {
    // output for this discriminator
    auto output = std::make_unique<PFTauDiscriminator>(edm::RefProd<PFTauCollection>(pfTaus));
    // loop over taus
    for(size_t itau = 0; itau < pfTaus->size(); ++itau) {
      PFTauRef tauRef(pfTaus, itau);
      // discriminator result
      std::vector<double> result = (*disc.plugin)(tauRef);
      // The desired index
      double selected_result = disc.defaultValue;
      if (result.size()-1 < disc.index) {
        selected_result = result[disc.index];
      }
      output->setValue(itau, selected_result);
    }
    evt.put(std::move(output), disc.collName);
  }
}

DEFINE_FWK_MODULE(PFTauMVAInputDiscriminantTranslator);
