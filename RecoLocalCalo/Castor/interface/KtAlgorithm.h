#ifndef CASTOR_KTALGORITHM_H
#define CASTOR_KTALGORITHM_H

#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorCluster.h"

// typedefs
typedef math::XYZPointD Point;

using namespace reco;

class KtAlgorithm {
    double phiangle (double testphi);
    CastorCluster calcRecom (CastorCluster a, CastorCluster b, int recom);
    double calcDistanceDeltaR (CastorCluster a, CastorCluster b);
    std::vector<std::vector<double> > calcdPairs (CastorClusterCollection protoclusters, std::vector<std::vector<double> > dPairs);
    std::vector<double> calcddi (CastorClusterCollection protoclusters, std::vector<double> ddi);
  public:
    CastorClusterCollection runKtAlgo (const CastorTowerRefVector& InputTowers, const int recom, const double rParameter);
};

#endif /* KTALGORITHM_H */
