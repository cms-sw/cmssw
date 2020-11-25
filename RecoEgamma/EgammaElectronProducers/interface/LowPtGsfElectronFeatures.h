#ifndef RecoEgamma_EgammaElectronProducers_LowPtGsfElectronFeatures_h
#define RecoEgamma_EgammaElectronProducers_LowPtGsfElectronFeatures_h

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include <vector>

namespace lowptgsfeleseed {

  std::vector<float> features(const reco::PreId& ecal,
                              const reco::PreId& hcal,
                              double rho,
                              const reco::BeamSpot& spot,
                              noZS::EcalClusterLazyTools& ecalTools);

}

namespace lowptgsfeleid {

  // feature list for new model (2019Sept15)
  std::vector<float> features_V1(reco::GsfElectron const& ele, float rho, float unbiased, float field_z);

  // feature list for original models (2019Aug07 and earlier)
  std::vector<float> features_V0(reco::GsfElectron const& ele, float rho, float unbiased);

  // Find most energetic clusters
  void findEnergeticClusters(reco::SuperCluster const&, int&, float&, float&, int&, int&);

  // Track-cluster matching for most energetic clusters
  void trackClusterMatching(reco::SuperCluster const&,
                            reco::GsfTrack const&,
                            bool const&,
                            GlobalPoint const&,
                            float&,
                            float&,
                            float&,
                            float&,
                            float&,
                            float&,
                            float&,
                            float&,
                            float&);

}  // namespace lowptgsfeleid

#endif  // RecoEgamma_EgammaElectronProducers_LowPtGsfElectronFeatures_h
