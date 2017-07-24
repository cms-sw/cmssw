
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/DeepFormats/interface/DeepFlavourTagInfo.h"

#include <algorithm>
#include <string>

#include "TTree.h"

class DeepFlavourExporter : public edm::one::EDAnalyzer<edm::one::SharedResources> {

  public:
	  explicit DeepFlavourExporter(const edm::ParameterSet&);
	  ~DeepFlavourExporter();

	  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:

    typedef std::vector<reco::DeepFlavourTagInfo> TagInfoCollection;
    typedef reco::JetTagCollection JetTagCollection;

    virtual void beginJob() override;
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;

	  const edm::EDGetTokenT< TagInfoCollection > tag_info_src_;
	  edm::EDGetTokenT<edm::View<pat::Jet>> jet_src_;

    // to keep data to be saved in the TTree
    std::vector<std::string> disc_names_;
    std::vector<float> disc_values_;
    deep::DeepFlavourFeatures features_;
    
    bool use_jet_src_;

    edm::Service<TFileService> fs;
    TTree* tree;
};

DeepFlavourExporter::DeepFlavourExporter(const edm::ParameterSet& iConfig) :
  tag_info_src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("tag_info_src"))),
  disc_names_((iConfig.getParameter<std::vector<std::string>>("btagDiscriminators"))),
  disc_values_(disc_names_.size(),0.0)
{
  auto jet_src_tag = iConfig.getParameter<edm::InputTag>("jet_src");
  if (jet_src_tag.label().size() == 0) {
    use_jet_src_ = false;
  } else {
    jet_src_ = consumes<edm::View<pat::Jet>>(jet_src_tag);
  } 
  usesResource("TFileService");
}


DeepFlavourExporter::~DeepFlavourExporter()
{
}

void DeepFlavourExporter::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
    edm::ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
}

void DeepFlavourExporter::beginJob()
{
  tree = fs->make<TTree>("tree", "tree");

  tree->Branch("features",&features_, 64000, 2);

  for (std::size_t i=0; i<disc_names_.size(); i++) {
    std::string disc_name = disc_names_.at(i);
    std::replace(disc_name.begin(), disc_name.end(), ':', '_');
    tree->Branch(disc_name.c_str(),&disc_values_[i]);
  }
}

void DeepFlavourExporter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(tag_info_src_, tag_infos);

  edm::Handle<edm::View<pat::Jet>> jets;
  if (use_jet_src_) {
    iEvent.getByToken(jet_src_, jets);
  }

  for (std::size_t i = 0; i<tag_infos->size(); i++) {
    const auto & tag_info = tag_infos->at(i);

    // make a copy of features (avoid const problem in branch)
    features_ = tag_info.features();
    
    if (use_jet_src_) {
      const auto & jet = jets->at(i);
      // save discriminator outputs in corresponding branches
      for (std::size_t j=0; j<disc_names_.size(); j++) {
        disc_values_.at(j) = jet.bDiscriminator(disc_names_.at(j));
      }
    }


    // fill tree per jet
    tree->Fill();
  }

}

void DeepFlavourExporter::endJob()
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepFlavourExporter);
