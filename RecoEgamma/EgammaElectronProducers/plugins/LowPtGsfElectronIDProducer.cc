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
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
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
  double eval(const GBRForest& model,
              const reco::GsfElectron&,
              double rho,
              float unbiased,
              float field_z,
              const reco::Track* trk = nullptr) const;

  template <typename EL, typename FID, typename FTRK>
  void doWork(double rho, float bz, EL const& electrons, FID&& idFunctor, FTRK&& trkFunctor, edm::Event&) const;
  const bool useGsfToTrack_;
  const bool usePAT_;
  edm::EDGetTokenT<reco::GsfElectronCollection> electrons_;
  edm::EDGetTokenT<pat::ElectronCollection> patElectrons_;
  const edm::EDGetTokenT<double> rho_;
  edm::EDGetTokenT<edm::Association<reco::TrackCollection>> gsf2trk_;
  edm::EDGetTokenT<edm::ValueMap<float>> unbiased_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;

  const std::vector<std::string> names_;
  std::vector<edm::EDPutTokenT<edm::ValueMap<float>>> putTokens_;
  const bool passThrough_;
  const double minPtThreshold_;
  const double maxPtThreshold_;
  std::vector<std::unique_ptr<const GBRForest>> models_;
  const std::vector<double> thresholds_;
  const std::string versionName_;
  enum class Version { V0, V1 };
  Version version_;
};

////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronIDProducer::LowPtGsfElectronIDProducer(const edm::ParameterSet& conf)
    : useGsfToTrack_(conf.getParameter<bool>("useGsfToTrack")),
      usePAT_(conf.getParameter<bool>("usePAT")),
      electrons_(),
      patElectrons_(),
      rho_(consumes<double>(conf.getParameter<edm::InputTag>("rho"))),
      gsf2trk_(),
      unbiased_(),
      fieldToken_(esConsumes()),
      names_(conf.getParameter<std::vector<std::string>>("ModelNames")),
      passThrough_(conf.getParameter<bool>("PassThrough")),
      minPtThreshold_(conf.getParameter<double>("MinPtThreshold")),
      maxPtThreshold_(conf.getParameter<double>("MaxPtThreshold")),
      thresholds_(conf.getParameter<std::vector<double>>("ModelThresholds")),
      versionName_(conf.getParameter<std::string>("Version")) {
  if (useGsfToTrack_) {
    gsf2trk_ = consumes<edm::Association<reco::TrackCollection>>(conf.getParameter<edm::InputTag>("gsfToTrack"));
  }
  if (usePAT_) {
    patElectrons_ = consumes<pat::ElectronCollection>(conf.getParameter<edm::InputTag>("electrons"));
  } else {
    electrons_ = consumes<reco::GsfElectronCollection>(conf.getParameter<edm::InputTag>("electrons"));
    unbiased_ = consumes<edm::ValueMap<float>>(conf.getParameter<edm::InputTag>("unbiased"));
  }
  for (auto& weights : conf.getParameter<std::vector<std::string>>("ModelWeights")) {
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
  if (versionName_ == "V0") {
    version_ = Version::V0;
  } else if (versionName_ == "V1") {
    version_ = Version::V1;
  } else {
    throw cms::Exception("Incorrect configuration") << "Unknown Version: " << versionName_ << "\n";
  }
  putTokens_.reserve(names_.size());
  for (const auto& name : names_) {
    putTokens_.emplace_back(produces<edm::ValueMap<float>>(name));
  }
}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronIDProducer::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  // Get z-component of B field
  math::XYZVector zfield(setup.getData(fieldToken_).inTesla(GlobalPoint(0, 0, 0)));

  // Pileup
  edm::Handle<double> hRho;
  event.getByToken(rho_, hRho);
  if (!hRho.isValid()) {
    std::ostringstream os;
    os << "Problem accessing rho collection for low-pT electrons" << std::endl;
    throw cms::Exception("InvalidHandle", os.str());
  }

  // Retrieve GsfToTrack Association from Event
  edm::Handle<edm::Association<reco::TrackCollection>> gsf2trk;
  if (useGsfToTrack_) {
    event.getByToken(gsf2trk_, gsf2trk);
  }

  double rho = *hRho;
  // Retrieve pat::Electrons or reco::GsfElectrons from Event
  edm::Handle<pat::ElectronCollection> patElectrons;
  edm::Handle<reco::GsfElectronCollection> electrons;
  if (usePAT_) {
    auto const& electrons = event.getHandle(patElectrons_);

    const std::string kUnbiased("unbiased");
    doWork(
        rho,
        zfield.z(),
        electrons,
        [&](auto const& ele) {
          if (!ele.isElectronIDAvailable(kUnbiased)) {
            return std::numeric_limits<float>::max();
          }
          return ele.electronID(kUnbiased);
        },
        [&](auto const& ele) {  // trkFunctor ...
          if (useGsfToTrack_) {
            using PackedPtr = edm::Ptr<pat::PackedCandidate>;
            const PackedPtr* ptr1 = ele.template userData<PackedPtr>("ele2packed");
            const PackedPtr* ptr2 = ele.template userData<PackedPtr>("ele2lost");
            auto hasBestTrack = [](const PackedPtr* ptr) {
              return ptr != nullptr && ptr->isNonnull() && ptr->isAvailable() && ptr->get() != nullptr &&
                     ptr->get()->bestTrack() != nullptr;
            };
            if (hasBestTrack(ptr1)) {
              return ptr1->get()->bestTrack();
            } else if (hasBestTrack(ptr2)) {
              return ptr2->get()->bestTrack();
            }
          } else {
            reco::TrackRef ref = ele.closestCtfTrackRef();
            if (ref.isNonnull() && ref.isAvailable()) {
              return ref.get();
            }
          }
          return static_cast<const reco::Track*>(nullptr);
        },
        event);
  } else {
    auto const& electrons = event.getHandle(electrons_);
    // ElectronSeed unbiased BDT
    edm::ValueMap<float> const& unbiasedH = event.get(unbiased_);
    doWork(
        rho,
        zfield.z(),
        electrons,
        [&](auto const& ele) {
          if (ele.core().isNull()) {
            return std::numeric_limits<float>::max();
          }
          const auto& gsf = ele.core()->gsfTrack();  // reco::GsfTrackRef
          if (gsf.isNull()) {
            return std::numeric_limits<float>::max();
          }
          return unbiasedH[gsf];
        },
        [&](auto const& ele) {  // trkFunctor ...
          if (useGsfToTrack_) {
            const auto& gsf = ele.core()->gsfTrack();
            if (gsf.isNonnull() && gsf.isAvailable()) {
              auto const& ref = (*gsf2trk)[gsf];
              if (ref.isNonnull() && ref.isAvailable()) {
                return ref.get();
              }
            }
          } else {
            reco::TrackRef ref = ele.closestCtfTrackRef();
            if (ref.isNonnull() && ref.isAvailable()) {
              return ref.get();
            }
          }
          return static_cast<const reco::Track*>(nullptr);
        },
        event);
  }
}

template <typename EL, typename FID, typename FTRK>
void LowPtGsfElectronIDProducer::doWork(
    double rho, float bz, EL const& electrons, FID&& idFunctor, FTRK&& trkFunctor, edm::Event& event) const {
  auto nElectrons = electrons->size();
  std::vector<float> ids;
  ids.reserve(nElectrons);
  std::transform(electrons->begin(), electrons->end(), std::back_inserter(ids), idFunctor);
  std::vector<const reco::Track*> trks;
  trks.reserve(nElectrons);
  std::transform(electrons->begin(), electrons->end(), std::back_inserter(trks), trkFunctor);

  std::vector<float> output(nElectrons);  //resused for each model
  for (unsigned int index = 0; index < names_.size(); ++index) {
    // Iterate through Electrons, evaluate BDT, and store result
    for (unsigned int iele = 0; iele < nElectrons; iele++) {
      auto const& ele = (*electrons)[iele];
      if (ids[iele] != std::numeric_limits<float>::max()) {
        output[iele] = eval(*models_[index], ele, rho, ids[iele], bz, trks[iele]);
      } else {
        output[iele] = -999.;
      }
    }
    edm::ValueMap<float> valueMap;
    edm::ValueMap<float>::Filler filler(valueMap);
    filler.insert(electrons, output.begin(), output.end());
    filler.fill();
    event.emplace(putTokens_[index], std::move(valueMap));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
//
double LowPtGsfElectronIDProducer::eval(const GBRForest& model,
                                        const reco::GsfElectron& ele,
                                        double rho,
                                        float unbiased,
                                        float field_z,
                                        const reco::Track* trk) const {
  std::vector<float> inputs;
  if (version_ == Version::V0) {
    inputs = lowptgsfeleid::features_V0(ele, rho, unbiased);
  } else if (version_ == Version::V1) {
    inputs = lowptgsfeleid::features_V1(ele, rho, unbiased, field_z, trk);
  }
  return model.GetResponse(inputs.data());
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronIDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("useGsfToTrack", false);
  desc.add<bool>("usePAT", false);
  desc.add<edm::InputTag>("electrons", edm::InputTag("lowPtGsfElectrons"));
  desc.addOptional<edm::InputTag>("gsfToTrack", edm::InputTag("lowPtGsfToTrackLinks"));
  desc.addOptional<edm::InputTag>("unbiased", edm::InputTag("lowPtGsfElectronSeedValueMaps:unbiased"));
  desc.add<edm::InputTag>("rho", edm::InputTag("fixedGridRhoFastjetAll"));
  desc.add<std::vector<std::string>>("ModelNames", {""});
  desc.add<std::vector<std::string>>(
      "ModelWeights", {"RecoEgamma/ElectronIdentification/data/LowPtElectrons/LowPtElectrons_ID_2020Nov28.root"});
  desc.add<std::vector<double>>("ModelThresholds", {-99.});
  desc.add<bool>("PassThrough", false);
  desc.add<double>("MinPtThreshold", 0.5);
  desc.add<double>("MaxPtThreshold", 15.);
  desc.add<std::string>("Version", "V1");
  descriptions.add("defaultLowPtGsfElectronID", desc);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LowPtGsfElectronIDProducer);
