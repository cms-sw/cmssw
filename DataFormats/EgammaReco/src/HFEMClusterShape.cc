#include "DataFormats/EgammaReco/interface/HFEMClusterShape.h"

reco::HFEMClusterShape::HFEMClusterShape(double eLong1x1,
					 double eShort1x1,
					 double eLong3x3,double eShort3x3,
					 double eLong5x5,
					 double eShort5x5,double eLongCore,
					 double CellEta,double CellPhi,
					 DetId seed):
  eLong1x1_(eLong1x1),
  eShort1x1_(eShort1x1),
  eLong3x3_(eLong3x3), 
  eShort3x3_(eShort3x3),  
  eLong5x5_(eLong5x5), 
  eShort5x5_(eShort5x5), 
  eLongCore_(eLongCore),
  CellEta_(CellEta), 
  CellPhi_(CellPhi),
  seed_(seed)
{
}


double reco::HFEMClusterShape::e1x1() const {
  return eLong1x1_+eShort1x1_;
}
double reco::HFEMClusterShape::e3x3() const {
  return eLong3x3_+eShort3x3_;
}
double reco::HFEMClusterShape::e5x5() const {
  return eLong5x5_+eShort5x5_;
}

double reco::HFEMClusterShape::eSeL() const{
  return eShort3x3()/eLong3x3();
}
double reco::HFEMClusterShape::eCOREe9() const{
  return eCore()/eLong3x3();
}
double reco::HFEMClusterShape::e9e25() const{
  return (eLong3x3()+eShort3x3())/(eLong5x5()+eShort5x5());
} 
