#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"
#include "RecoTauTag/RecoTau/plugins/PFTauSelectorDefinition.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class PFTauSelector : public ObjectSelectorStream<PFTauSelectorDefinition> {
public:
  PFTauSelector(const edm::ParameterSet& ps) : ObjectSelectorStream<PFTauSelectorDefinition>(ps) {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("PF tau selector module");

    desc.add<edm::InputTag>("src", edm::InputTag("fixedConePFTauProducer"));
    desc.add<std::string>("cut", "pt > 0");

    edm::ParameterSetDescription psd1;
    psd1.add<edm::InputTag>("discriminator");
    psd1.add<double>("selectionCut");
    edm::ParameterSet ps1;
    ps1.addParameter<edm::InputTag>("discriminator", edm::InputTag("fixedConePFTauDiscriminationByIsolation"));
    ps1.addParameter<double>("selectionCut", 0.5);
    desc.addVPSet("discriminators", psd1, {ps1});

    edm::ParameterSetDescription psd2;
    psd2.add<edm::InputTag>("discriminator");
    psd2.add<std::vector<std::string>>("rawValues");
    psd2.add<std::vector<std::string>>("workingPoints");
    psd2.add<std::vector<double>>("selectionCuts");
    desc.addVPSet("discriminatorContainers", psd2, {});

    descriptions.add("pfTauSelector", desc);
  }
};

DEFINE_FWK_MODULE(PFTauSelector);
