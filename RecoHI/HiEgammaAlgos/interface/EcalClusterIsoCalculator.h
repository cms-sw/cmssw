#ifndef HiEgammaAlgos_EcalClusterIsoCalculator_h
#define HiEgammaAlgos_EcalClusterIsoCalculator_h

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

class EcalClusterIsoCalculator {
public:
  EcalClusterIsoCalculator(const edm::Handle<reco::BasicClusterCollection> barrel,
                           const edm::Handle<reco::BasicClusterCollection> endcap);

  /// Return the ecal cluster energy in a cone around the SC
  double getEcalClusterIso(const reco::SuperClusterRef clus, const double radius, const double threshold);
  /// Return the background-subtracted ecal cluster energy in a cone around the SC
  double getBkgSubEcalClusterIso(const reco::SuperClusterRef clus, const double radius, const double threshold);

private:
  const reco::BasicClusterCollection *fEBclusters_;
  const reco::BasicClusterCollection *fEEclusters_;
};

#endif
