#ifndef Math_LorentzVector_h
#define Math_LorentzVector_h
// $Id: LorentzVector.h,v 1.2 2005/12/15 03:39:02 llista Exp $
#include <Rtypes.h>
#include <Math/PtEtaPhiE4D.h>
#include <Math/PxPyPzE4D.h>
#include <Math/LorentzVector.h>

namespace math {
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<Double32_t> > PtEtaPhiELorentzVector;
  typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> > XYZTLorentzVector;
}

#endif
