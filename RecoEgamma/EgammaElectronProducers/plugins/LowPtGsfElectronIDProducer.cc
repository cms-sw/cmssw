#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
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
#include "FWCore/Framework/interface/Event.h"

#include <string>
#include <vector>

namespace {

  std::vector<float> getFeatures(const reco::GsfElectronRef& ele, double rho) {
    // KF track
    float trk_p_ = -1.;
    float trk_nhits_ = -1.;
    float trk_chi2red_ = -1.;
    // GSF track
    float gsf_nhits_ = -1.;
    float gsf_chi2red_ = -1.;
    // SC
    float sc_E_ = -1.;
    float sc_eta_ = -1.;
    float sc_etaWidth_ = -1.;
    float sc_phiWidth_ = -1.;
    // Track-cluster matching
    float match_seed_dEta_ = -1.;
    float match_eclu_EoverP_ = -1.;
    float match_SC_EoverP_ = -1.;
    float match_SC_dEta_ = -1.;
    float match_SC_dPhi_ = -1.;
    // Shower shape vars
    float shape_full5x5_sigmaIetaIeta_ = -1.;
    float shape_full5x5_sigmaIphiIphi_ = -1.;
    float shape_full5x5_HoverE_ = -1.;
    float shape_full5x5_r9_ = -1.;
    float shape_full5x5_circularity_ = -1.;
    // Misc
    float rho_ = -1.;
    float brem_frac_ = -1.;
    float ele_pt_ = -1.;

    // KF tracks
    if (ele->core().isNonnull()) {
      reco::TrackRef trk = ele->core()->ctfTrack();  //@@ is this what we want?!
      if (trk.isNonnull()) {
        trk_p_ = float(trk->p());
        trk_nhits_ = float(trk->found());
        trk_chi2red_ = float(trk->normalizedChi2());
      }
    }

    // GSF tracks
    if (ele->core().isNonnull()) {
      reco::GsfTrackRef gsf = ele->core()->gsfTrack();
      if (gsf.isNonnull()) {
        gsf_nhits_ = gsf->found();
        gsf_chi2red_ = gsf->normalizedChi2();
      }
    }

    // Super clusters
    if (ele->core().isNonnull()) {
      reco::SuperClusterRef sc = ele->core()->superCluster();
      if (sc.isNonnull()) {
        sc_E_ = sc->energy();
        sc_eta_ = sc->eta();
        sc_etaWidth_ = sc->etaWidth();
        sc_phiWidth_ = sc->phiWidth();
      }
    }

    // Track-cluster matching
    if (ele.isNonnull()) {
      match_seed_dEta_ = ele->deltaEtaSeedClusterTrackAtCalo();
      match_eclu_EoverP_ = (1. / ele->ecalEnergy()) - (1. / ele->p());
      match_SC_EoverP_ = ele->eSuperClusterOverP();
      match_SC_dEta_ = ele->deltaEtaSuperClusterTrackAtVtx();
      match_SC_dPhi_ = ele->deltaPhiSuperClusterTrackAtVtx();
    }

    // Shower shape vars
    if (ele.isNonnull()) {
      shape_full5x5_sigmaIetaIeta_ = ele->full5x5_sigmaIetaIeta();
      shape_full5x5_sigmaIphiIphi_ = ele->full5x5_sigmaIphiIphi();
      shape_full5x5_HoverE_ = ele->full5x5_hcalOverEcal();
      shape_full5x5_r9_ = ele->full5x5_r9();
      shape_full5x5_circularity_ = 1. - ele->full5x5_e1x5() / ele->full5x5_e5x5();
    }

    // Misc
    rho_ = rho;
    if (ele.isNonnull()) {
      brem_frac_ = ele->fbrem();
      ele_pt_ = ele->pt();
    }

    std::vector<float> output = {
        rho_,
        ele_pt_,
        sc_eta_,
        shape_full5x5_sigmaIetaIeta_,
        shape_full5x5_sigmaIphiIphi_,
        shape_full5x5_circularity_,
        shape_full5x5_r9_,
        sc_etaWidth_,
        sc_phiWidth_,
        shape_full5x5_HoverE_,
        trk_nhits_,
        trk_chi2red_,
        gsf_chi2red_,
        brem_frac_,
        gsf_nhits_,
        match_SC_EoverP_,
        match_eclu_EoverP_,
        match_SC_dEta_,
        match_SC_dPhi_,
        match_seed_dEta_,
        sc_E_,
        trk_p_,
    };
    return output;
  }

}  // namespace

class LowPtGsfElectronIDProducer final
    : public edm::global::EDProducer<> {
public:
  explicit LowPtGsfElectronIDProducer(const edm::ParameterSet&);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  double eval(const std::string& name, const reco::GsfElectronRef&, double rho) const;

  const edm::EDGetTokenT<reco::GsfElectronCollection> gsfElectrons_;
  const edm::EDGetTokenT<double> rho_;
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
    : gsfElectrons_(consumes<reco::GsfElectronCollection>(conf.getParameter<edm::InputTag>("electrons"))),
      rho_(consumes<double>(conf.getParameter<edm::InputTag>("rho"))),
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
          << "'ModelWeights' size (" << models_.size() << ") != 'ModelThresholds' size (" << thresholds_.size()
          << ").\n";
    }
  for (const auto& name : names_) {
    produces<edm::ValueMap<float> >(name);
  }
}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronIDProducer::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  // Pileup
  edm::Handle<double> rho;
  event.getByToken(rho_, rho);
  if (!rho.isValid()) {
    edm::LogError("Problem with rho handle");
  }

  // Retrieve GsfElectrons from Event
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  event.getByToken(gsfElectrons_, gsfElectrons);
  if (!gsfElectrons.isValid()) {
    edm::LogError("Problem with gsfElectrons handle");
  }

  // Iterate through Electrons, evaluate BDT, and store result
  std::vector<std::vector<float> > output;
  for (unsigned int iname = 0; iname < names_.size(); ++iname) {
    output.emplace_back(gsfElectrons->size(), -999.);
  }
  for (unsigned int iele = 0; iele < gsfElectrons->size(); iele++) {
    reco::GsfElectronRef ele(gsfElectrons, iele);
    //if ( !passThrough_ && ( ele->pt() < minPtThreshold_ ) ) { continue; }
    for (unsigned int iname = 0; iname < names_.size(); ++iname) {
      output[iname][iele] = eval(names_[iname], ele, *rho);
    }
  }

  // Create and put ValueMap in Event
  for (unsigned int iname = 0; iname < names_.size(); ++iname) {
    auto ptr = std::make_unique<edm::ValueMap<float> >(edm::ValueMap<float>());
    edm::ValueMap<float>::Filler filler(*ptr);
    filler.insert(gsfElectrons, output[iname].begin(), output[iname].end());
    filler.fill();
    reco::GsfElectronRef ele(gsfElectrons, 0);
    event.put(std::move(ptr), names_[iname]);
  }
}

double LowPtGsfElectronIDProducer::eval(const std::string& name, const reco::GsfElectronRef& ele, double rho) const {
  auto iter = std::find(names_.begin(), names_.end(), name);
  if (iter != names_.end()) {
    int index = std::distance(names_.begin(), iter);
    std::vector<float> inputs = getFeatures(ele, rho);
    return models_.at(index)->GetResponse(inputs.data());
  } else {
    throw cms::Exception("Unknown model name")
        << "'Name given: '" << name << "'. Check against configuration file.\n";
  }
  return 0.;
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronIDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("electrons",edm::InputTag("lowPtGsfElectrons"));
  desc.add<edm::InputTag>("rho",edm::InputTag("fixedGridRhoFastjetAllTmp"));
  desc.add< std::vector<std::string> >("ModelNames",{""});
  desc.add< std::vector<std::string> >("ModelWeights",{"RecoEgamma/ElectronIdentification/data/LowPtElectrons/RunII_Autumn18_LowPtElectrons_mva_id.xml.gz"});
  desc.add< std::vector<double> >("ModelThresholds",{-10.});
  desc.add<bool>("PassThrough",false);
  desc.add<double>("MinPtThreshold",0.5);
  desc.add<double>("MaxPtThreshold",15.);
  descriptions.add("lowPtGsfElectronID",desc);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LowPtGsfElectronIDProducer);
