#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "RecoEgamma/EgammaElectronProducers/interface/LowPtGsfElectronFeatures.h"
#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
//
class LowPtGsfElectronIDProducer final : public edm::global::EDProducer<> {
public:
  explicit LowPtGsfElectronIDProducer(const edm::ParameterSet&);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  double eval(const std::string& name, const edm::Ptr<reco::GsfElectron>&, double rho, float unbiased) const;

  const edm::EDGetTokenT<edm::View<reco::GsfElectron> > gsfElectrons_;
  const edm::EDGetTokenT<double> rho_;
  const edm::EDGetTokenT<edm::ValueMap<float> > unbiased_;
  const std::vector<std::string> names_;
  const bool passThrough_;
  const double minPtThreshold_;
  const double maxPtThreshold_;
  std::vector<std::unique_ptr<const GBRForest> > models_;
  const std::vector<double> thresholds_;
};

////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronIDProducer::LowPtGsfElectronIDProducer(const edm::ParameterSet& conf)
    : gsfElectrons_(consumes<edm::View<reco::GsfElectron> >(conf.getParameter<edm::InputTag>("electrons"))),
      rho_(consumes<double>(conf.getParameter<edm::InputTag>("rho"))),
      unbiased_(consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("unbiased"))),
      names_(conf.getParameter<std::vector<std::string> >("ModelNames")),
      passThrough_(conf.getParameter<bool>("PassThrough")),
      minPtThreshold_(conf.getParameter<double>("MinPtThreshold")),
      maxPtThreshold_(conf.getParameter<double>("MaxPtThreshold")),
      thresholds_(conf.getParameter<std::vector<double> >("ModelThresholds")) {
  for (auto& weights : conf.getParameter<std::vector<std::string> >("ModelWeights")) {
    models_.push_back(createGBRForest(edm::FileInPath(weights)));
  }
  if (names_.size() != models_.size()) {
    throw cms::Exception("Incorrect configuration")
        << "'ModelNames' size (" << names_.size() << ") != 'ModelWeights' size (" << models_.size() << ").\n";
  }
  if (models_.size() != thresholds_.size()) {
    throw cms::Exception("Incorrect configuration")
        << "'ModelWeights' size (" << models_.size() << ") != 'ModelThresholds' size (" << thresholds_.size() << ").\n";
  }
  for (const auto& name : names_) {
    produces<edm::ValueMap<float> >(name);
  }
}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronIDProducer::produce(edm::StreamID, edm::Event& event, const edm::EventSetup&) const {
  // Pileup
  edm::Handle<double> rho;
  event.getByToken(rho_, rho);
  if (!rho.isValid()) {
    std::ostringstream os;
    os << "Problem accessing rho collection for low-pT electrons" << std::endl;
    edm::LogError("InvalidHandle") << os.str();
    throw cms::Exception("InvalidHandle", os.str());
  }

  // Retrieve GsfElectrons from Event
  edm::Handle<edm::View<reco::GsfElectron> > gsfElectrons;
  event.getByToken(gsfElectrons_, gsfElectrons);
  if (!gsfElectrons.isValid()) {
    std::ostringstream os;
    os << "Problem accessing low-pT gsfElectrons collection" << std::endl;
    edm::LogError("InvalidHandle") << os.str();
    throw cms::Exception("InvalidHandle", os.str());
  }

  // ElectronSeed unbiased BDT
  edm::Handle<edm::ValueMap<float> > unbiasedH;
  event.getByToken(unbiased_, unbiasedH);
  if (!unbiasedH.isValid()) {
    std::ostringstream os;
    os << "Problem accessing low-pT 'unbiased' ElectronSeed collection" << std::endl;
    edm::LogError("InvalidHandle") << os.str();
    throw cms::Exception("InvalidHandle", os.str());
  }

  // Iterate through Electrons, evaluate BDT, and store result
  std::vector<std::vector<float> > output;
  for (unsigned int iname = 0; iname < names_.size(); ++iname) {
    output.emplace_back(gsfElectrons->size(), -999.);
  }
  for (unsigned int iele = 0; iele < gsfElectrons->size(); iele++) {
    edm::Ptr<reco::GsfElectron> ele(gsfElectrons, iele);

    if (ele->core().isNull()) {
      continue;
    }
    const auto& gsf = ele->core()->gsfTrack();  // reco::GsfTrackRef
    if (gsf.isNull()) {
      continue;
    }
    float unbiased = (*unbiasedH)[gsf];

    //if ( !passThrough_ && ( ele->pt() < minPtThreshold_ ) ) { continue; }
    for (unsigned int iname = 0; iname < names_.size(); ++iname) {
      output[iname][iele] = eval(names_[iname], ele, *rho, unbiased);
    }
  }

  // Create and put ValueMap in Event
  for (unsigned int iname = 0; iname < names_.size(); ++iname) {
    auto ptr = std::make_unique<edm::ValueMap<float> >(edm::ValueMap<float>());
    edm::ValueMap<float>::Filler filler(*ptr);
    filler.insert(gsfElectrons, output[iname].begin(), output[iname].end());
    filler.fill();
    event.put(std::move(ptr), names_[iname]);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
//
double LowPtGsfElectronIDProducer::eval(const std::string& name,
                                        const edm::Ptr<reco::GsfElectron>& ele,
                                        double rho,
                                        float unbiased) const {
  auto iter = std::find(names_.begin(), names_.end(), name);
  if (iter != names_.end()) {
    int index = std::distance(names_.begin(), iter);
    std::vector<float> inputs = lowptgsfeleid::features(ele, rho, unbiased);
    return models_.at(index)->GetResponse(inputs.data());
  } else {
    throw cms::Exception("Unknown model name") << "'Name given: '" << name << "'. Check against configuration file.\n";
  }
  return 0.;
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronIDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("electrons", edm::InputTag("lowPtGsfElectrons"));
  desc.add<edm::InputTag>("unbiased", edm::InputTag("lowPtGsfElectronSeedValueMaps:unbiased"));
  desc.add<edm::InputTag>("rho", edm::InputTag("fixedGridRhoFastjetAllTmp"));
  desc.add<std::vector<std::string> >("ModelNames", {""});
  desc.add<std::vector<std::string> >(
      "ModelWeights",
      {"RecoEgamma/ElectronIdentification/data/LowPtElectrons/RunII_Autumn18_LowPtElectrons_mva_id.xml.gz"});
  desc.add<std::vector<double> >("ModelThresholds", {-10.});
  desc.add<bool>("PassThrough", false);
  desc.add<double>("MinPtThreshold", 0.5);
  desc.add<double>("MaxPtThreshold", 15.);
  descriptions.add("lowPtGsfElectronID", desc);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LowPtGsfElectronIDProducer);
