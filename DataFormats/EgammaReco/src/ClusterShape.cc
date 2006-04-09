#include "DataFormats/EgammaReco/interface/ClusterShape.h"

reco::ClusterShape::ClusterShape( double cEE, double cEP, double cPP, 
                                  double eMax, double e2x2, double e3x3, double e5x5,
                                  double hadOverEcal ) :
  covEtaEta_( cEE ), covEtaPhi_( cEP ), covPhiPhi_( cPP ),
  eMax_(eMax), e2x2_(e2x2), e3x3_(e3x3), e5x5_(e5x5), hadOverEcal_( hadOverEcal ) {
}
