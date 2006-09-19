#include "DataFormats/EgammaReco/interface/ClusterShape.h"

reco::ClusterShape::ClusterShape( double cEE, double cEP, double cPP, 
                  double eMax, DetId eMaxId, double e2nd, DetId e2ndId,
								  double e2x2, double e3x2, double e3x3, double e4x4, double e5x5,
        	  			double e3x2Ratio, math::XYZPoint location,
                  std::vector<double> energyBasketFractionEta,
                  std::vector<double> energyBasketFractionPhi) :
  covEtaEta_( cEE ), covEtaPhi_( cEP ), covPhiPhi_( cPP ), 
  eMax_(eMax), e2nd_(e2nd), 
  e2x2_(e2x2), e3x2_(e3x2), e3x3_(e3x3), e4x4_(e4x4), e5x5_(e5x5),
  e3x2Ratio_(e3x2Ratio), location_(location),
  eMaxId_(eMaxId), e2ndId_(e2ndId)
{
  energyBasketFractionEta_ = energyBasketFractionEta;
  energyBasketFractionPhi_ = energyBasketFractionPhi;
}
