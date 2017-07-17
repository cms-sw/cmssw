#ifndef EgammaIsolationProducers_HcalPFClusterIsolation_h
#define EgammaIsolationProducers_HcalPFClusterIsolation_h

//*****************************************************************************
// File:      HcalPFClusterIsolation.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matteo Sani
// Institute: UCSD
//*****************************************************************************

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include <vector>

template<typename T1>
class HcalPFClusterIsolation {
 public:

  typedef std::vector<T1> T1Collection;
  typedef edm::Ref<T1Collection> T1Ref;

  HcalPFClusterIsolation(double drMax,
			 double drVetoBarrel,
			 double drVetoEndcap,
			 double etaStripBarrel,
			 double etaStripEndcap,
			 double energyBarrel,
			 double energyEndcap,
			 bool useEt);
  
  ~HcalPFClusterIsolation();
  double getSum(const T1Ref candRef, const std::vector<edm::Handle<reco::PFClusterCollection>>& clusterHandles);  

 private:
  const double drMax_;
  const double drVetoBarrel_;
  const double drVetoEndcap_;
  const double etaStripBarrel_;
  const double etaStripEndcap_;
  const double energyBarrel_;
  const double energyEndcap_;
  const bool useEt_;

};

#endif
