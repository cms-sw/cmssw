#include "LowPtGsfElectronSeedHeavyObjectCache.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "RecoEgamma/EgammaElectronProducers/interface/LowPtGsfElectronFeatures.h"

#include <string>

namespace lowptgsfeleseed {

  ////////////////////////////////////////////////////////////////////////////////
  //
  HeavyObjectCache::HeavyObjectCache(const edm::ParameterSet& conf) {
    for (auto& name : conf.getParameter<std::vector<std::string> >("ModelNames")) {
      names_.push_back(name);
    }
    for (auto& weights : conf.getParameter<std::vector<std::string> >("ModelWeights")) {
      models_.push_back(createGBRForest(edm::FileInPath(weights)));
    }
    for (auto& thresh : conf.getParameter<std::vector<double> >("ModelThresholds")) {
      thresholds_.push_back(thresh);
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
  }

  ////////////////////////////////////////////////////////////////////////////////
  //
  bool HeavyObjectCache::eval(const std::string& name,
                              reco::PreId& ecal,
                              reco::PreId& hcal,
                              double rho,
                              const reco::BeamSpot& spot,
                              noZS::EcalClusterLazyTools& ecalTools) const {
    std::vector<std::string>::const_iterator iter = std::find(names_.begin(), names_.end(), name);
    if (iter != names_.end()) {
      int index = std::distance(names_.begin(), iter);
      std::vector<float> inputs = features(ecal, hcal, rho, spot, ecalTools);
      float output = models_.at(index)->GetResponse(inputs.data());
      bool pass = output > thresholds_.at(index);
      ecal.setMVA(pass, output, index);
      return pass;
    } else {
      throw cms::Exception("Unknown model name")
          << "'Name given: '" << name << "'. Check against configuration file.\n";
    }
  }

}  // namespace lowptgsfeleseed
