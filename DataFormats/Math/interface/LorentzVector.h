#ifndef Math_LorentzVector_h
#define Math_LorentzVector_h
// $Id: LorentzVector.h,v 1.6 2006/06/26 08:56:09 llista Exp $
#include "Rtypes.h"
#include "Math/PtEtaPhiE4D.h"
#include "Math/PxPyPzE4D.h"
#include "Math/LorentzVector.h"

namespace math {

  /// Lorentz vector with cartesian internal representation
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
  /// Lorentz vector with cylindrical internal representation using pseudorapidity
  typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

  /// Lorentz vector with cartesian internal representation
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float> > PtEtaPhiELorentzVectorF;
  /// Lorentz vector with cylindrical internal representation using pseudorapidity
  typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> > XYZTLorentzVectorF;

  /// Lorentz vector with cartesian internal representation
  typedef PtEtaPhiELorentzVectorD PtEtaPhiELorentzVector;
  /// Lorentz vector with cylindrical internal representation using pseudorapidity
  typedef XYZTLorentzVectorD XYZTLorentzVector;
}

#endif
