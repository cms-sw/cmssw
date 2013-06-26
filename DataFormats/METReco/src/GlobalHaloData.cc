#include "DataFormats/METReco/interface/GlobalHaloData.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"


/*
  [class]:  GlobalHaloData
  [authors]: R. Remington, The University of Florida
  [description]: See GlobalHaloData.h
  [date]: October 15, 2009
*/

using namespace reco;

GlobalHaloData::GlobalHaloData()
{
  METOverSumEt_ = 0.;
  dMEx_ = 0.;
  dMEy_ = 0.;
  dSumEt_ = 0.;
}


reco::CaloMET GlobalHaloData::GetCorrectedCaloMET(const reco::CaloMET& RawMET) const
{
  double mex = RawMET.px() + dMEx_;
  double mey = RawMET.py() + dMEy_;
  double mez = RawMET.pz() ;
  double sumet  = RawMET.sumEt() + dSumEt_ ; 
  const math::XYZTLorentzVector p4( mex, mey, mez, std::sqrt(mex*mex + mey*mey + mez*mez));
  const math::XYZPoint vtx (0., 0., 0.);
  
  reco::CaloMET CorrectedMET( RawMET.getSpecific(), sumet, p4, vtx );
  return CorrectedMET;
}

