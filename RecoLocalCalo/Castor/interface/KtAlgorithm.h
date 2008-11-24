#ifndef CASTOR_KTALGORITHM_H
#define CASTOR_KTALGORITHM_H

#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"

using namespace reco;

class KtAlgorithm {
    double phiangle (double testphi);
    CastorJet calcRecom (CastorJet a, CastorJet b, int recom);
    double calcDistanceDeltaR (CastorJet a, CastorJet b);
    std::vector<std::vector<double> > calcdPairs (CastorJetCollection protojets, std::vector<std::vector<double> > dPairs);
    std::vector<double> calcddi (CastorJetCollection protojets, std::vector<double> ddi);
  public:
    CastorJetCollection runKtAlgo (const CastorTowerCollection inputtowers, const int recom, const double rParameter);
};

#endif /* KTALGORITHM_H */
