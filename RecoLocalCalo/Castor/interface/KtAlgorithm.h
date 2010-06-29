#ifndef CASTOR_KTALGORITHM_H
#define CASTOR_KTALGORITHM_H

#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorCluster.h"

// typedefs
typedef math::XYZPointD Point;

class KtAlgorithm {
    double phiangle (double testphi);
    reco::CastorCluster calcRecom (reco::CastorCluster a, reco::CastorCluster b, int recom);
    double calcDistanceDeltaR (reco::CastorCluster a, reco::CastorCluster b);
    std::vector<std::vector<double> > calcdPairs (reco::CastorClusterCollection protoclusters, std::vector<std::vector<double> > dPairs);
    std::vector<double> calcddi (reco::CastorClusterCollection protoclusters, std::vector<double> ddi);
  public:
    reco::CastorClusterCollection runKtAlgo (const reco::CastorTowerRefVector& InputTowers, const int recom, const double rParameter);
};

#endif /* KTALGORITHM_H */
