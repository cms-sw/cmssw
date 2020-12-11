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
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
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
  double eval(
      const std::string& name, const edm::Ptr<reco::GsfElectron>&, double rho, float unbiased, float field_z) const;

  const edm::EDGetTokenT<edm::View<reco::GsfElectron> > electrons_;
  const edm::EDGetTokenT<double> rho_;
  const edm::EDGetTokenT<edm::ValueMap<float> > unbiased_;
  const std::vector<std::string> names_;
  const bool passThrough_;
  const double minPtThreshold_;
  const double maxPtThreshold_;
  std::vector<std::unique_ptr<const GBRForest> > models_;
  const std::vector<double> thresholds_;
  const std::string version_;
};

////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronIDProducer::LowPtGsfElectronIDProducer(const edm::ParameterSet& conf)
    : electrons_(consumes<edm::View<reco::GsfElectron> >(conf.getParameter<edm::InputTag>("electrons"))),
      rho_(consumes<double>(conf.getParameter<edm::InputTag>("rho"))),
      unbiased_(consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("unbiased"))),
      names_(conf.getParameter<std::vector<std::string> >("ModelNames")),
      passThrough_(conf.getParameter<bool>("PassThrough")),
      minPtThreshold_(conf.getParameter<double>("MinPtThreshold")),
      maxPtThreshold_(conf.getParameter<double>("MaxPtThreshold")),
      thresholds_(conf.getParameter<std::vector<double> >("ModelThresholds")),
      version_(conf.getParameter<std::string>("Version")) {
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
  if (version_ != "V0" && version_ != "V1") {
    throw cms::Exception("Incorrect configuration") << "Unknown Version: " << version_ << "\n";
  }
  for (const auto& name : names_) {
    produces<edm::ValueMap<float> >(name);
  }
}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronIDProducer::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  // Get z-component of B field
  edm::ESHandle<MagneticField> field;
  setup.get<IdealMagneticFieldRecord>().get(field);
  math::XYZVector zfield(field->inTesla(GlobalPoint(0, 0, 0)));

  // Pileup
  edm::Handle<double> rho;
  event.getByToken(rho_, rho);
  if (!rho.isValid()) {
    std::ostringstream os;
    os << "Problem accessing rho collection for low-pT electrons" << std::endl;
    throw cms::Exception("InvalidHandle", os.str());
  }

  // Retrieve GsfElectrons from Event
  edm::Handle<edm::View<reco::GsfElectron> > electrons;
  event.getByToken(electrons_, electrons);
  if (!electrons.isValid()) {
    std::ostringstream os;
    os << "Problem accessing low-pT electrons collection" << std::endl;
    throw cms::Exception("InvalidHandle", os.str());
  }

  // ElectronSeed unbiased BDT
  edm::Handle<edm::ValueMap<float> > unbiasedH;
  event.getByToken(unbiased_, unbiasedH);

  // Iterate through Electrons, evaluate BDT, and store result
  std::vector<std::vector<float> > output;
  for (unsigned int iname = 0; iname < names_.size(); ++iname) {
    output.emplace_back(electrons->size(), -999.);
  }
  for (unsigned int iele = 0; iele < electrons->size(); iele++) {
    edm::Ptr<reco::GsfElectron> ele(electrons, iele);

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
      output[iname][iele] = eval(names_[iname], ele, *rho, unbiased, zfield.z());
    }
  }

  // Create and put ValueMap in Event
  for (unsigned int iname = 0; iname < names_.size(); ++iname) {
    auto ptr = std::make_unique<edm::ValueMap<float> >(edm::ValueMap<float>());
    edm::ValueMap<float>::Filler filler(*ptr);
    filler.insert(electrons, output[iname].begin(), output[iname].end());
    filler.fill();
    event.put(std::move(ptr), names_[iname]);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
//
double LowPtGsfElectronIDProducer::eval(
    const std::string& name, const edm::Ptr<reco::GsfElectron>& ele, double rho, float unbiased, float field_z) const {
  auto iter = std::find(names_.begin(), names_.end(), name);
  if (iter != names_.end()) {
    int index = std::distance(names_.begin(), iter);
    std::vector<float> inputs;
    if (version_ == "V0") {
      inputs = lowptgsfeleid::features_V0(*ele, rho, unbiased);
    } else if (version_ == "V1") {
      inputs = lowptgsfeleid::features_V1(*ele, rho, unbiased, field_z);
    }
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
  desc.add<edm::InputTag>("rho", edm::InputTag("fixedGridRhoFastjetAll"));
  desc.add<std::vector<std::string> >("ModelNames", {""});
  desc.add<std::vector<std::string> >(
      "ModelWeights", {"RecoEgamma/ElectronIdentification/data/LowPtElectrons/LowPtElectrons_ID_2020Sept15.root"});
  desc.add<std::vector<double> >("ModelThresholds", {-99.});
  desc.add<bool>("PassThrough", false);
  desc.add<double>("MinPtThreshold", 0.5);
  desc.add<double>("MaxPtThreshold", 15.);
  desc.add<std::string>("Version", "V1");
  descriptions.add("lowPtGsfElectronID", desc);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LowPtGsfElectronIDProducer);
