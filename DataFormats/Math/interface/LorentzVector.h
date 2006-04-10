#ifndef Math_LorentzVector_h
#define Math_LorentzVector_h
// $Id: LorentzVector.h,v 1.4 2006/03/06 12:45:29 llista Exp $
#include <Rtypes.h>
#include <Math/PtEtaPhiE4D.h>
#include <Math/PxPyPzE4D.h>
#include <Math/LorentzVector.h>

namespace math {
  /// Lorentz vector with cartesian internal representation
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<Double32_t> > PtEtaPhiELorentzVector;
  /// Lorentz vector with cylindrical internal representation using pseudorapidity
  typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> > XYZTLorentzVector;

  /// Lorentz vector with cartesian internal representation
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
  /// Lorentz vector with cylindrical internal representation using pseudorapidity
  typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

  /// Lorentz vector with cartesian internal representation
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float> > PtEtaPhiELorentzVectorF;
  /// Lorentz vector with cylindrical internal representation using pseudorapidity
  typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> > XYZTLorentzVectorF;
}

#endif
