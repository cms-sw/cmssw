#include "DataFormats/EgammaReco/interface/ClusterShape.h"

reco::ClusterShape::ClusterShape( double cEE, double cEP, double cPP, 
                  double eMax, DetId eMaxId, double e2nd, DetId e2ndId,
		  double e2x2, double e3x2, double e3x3, double e4x4,
                  double e5x5, double e2x5Right, double e2x5Left, 
                  double e2x5Top, double e2x5Bottom, double e3x2Ratio,
		  double LAT, double etaLAT, double phiLAT, double A20, double A42,
                  const std::vector<double>& energyBasketFractionEta,
                  const std::vector<double>& energyBasketFractionPhi) :
  covEtaEta_( cEE ), covEtaPhi_( cEP ), covPhiPhi_( cPP ), 
  eMax_(eMax), e2nd_(e2nd), e2x2_(e2x2), e3x2_(e3x2), e3x3_(e3x3), e4x4_(e4x4),
  e5x5_(e5x5), e2x5Right_(e2x5Right), e2x5Left_(e2x5Left), e2x5Top_(e2x5Top), 
  e2x5Bottom_(e2x5Bottom), e3x2Ratio_(e3x2Ratio), 
  LAT_(LAT), etaLAT_(etaLAT), phiLAT_(phiLAT), A20_(A20), A42_(A42),
  eMaxId_(eMaxId), e2ndId_(e2ndId)
{
  energyBasketFractionEta_ = energyBasketFractionEta;
  energyBasketFractionPhi_ = energyBasketFractionPhi;
}
