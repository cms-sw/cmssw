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

  std::vector<float> features(edm::Ptr<reco::GsfElectron> const& ele, float rho, float unbiased);

  std::vector<float> features(edm::Ref<std::vector<reco::GsfElectron> > const& ele, float rho, float unbiased);

  std::vector<float> features(edm::Ref<edm::View<reco::GsfElectron> > const& ele, float rho, float unbiased);

}  // namespace lowptgsfeleid

#endif  // RecoEgamma_EgammaElectronProducers_LowPtGsfElectronFeatures_h
