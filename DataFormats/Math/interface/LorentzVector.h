#ifndef Math_LorentzVector_h
#define Math_LorentzVector_h
// $Id: LorentzVector.h,v 1.10 2007/07/31 15:20:15 ratnik Exp $
#include "Math/PtEtaPhiE4D.h"
#include "Math/PtEtaPhiM4D.h"
#include "Math/LorentzVector.h"

namespace math {

  /// Lorentz vector with cartesian internal representation
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> > PtEtaPhiMLorentzVectorD;
  /// Lorentz vector with cartesian internal representation
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
  /// Lorentz vector with cylindrical internal representation using pseudorapidity
  typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;
  /// Lorentz vector with cylindrical internal representation using pseudorapidity

  /// Lorentz vector with cartesian internal representation
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float> > PtEtaPhiMLorentzVectorF;
  /// Lorentz vector with cartesian internal representation
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float> > PtEtaPhiELorentzVectorF;
  /// Lorentz vector with cylindrical internal representation using pseudorapidity
  typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> > XYZTLorentzVectorF;

  /// Lorentz vector with cartesian internal representation
  typedef PtEtaPhiMLorentzVectorD PtEtaPhiMLorentzVector;
  /// Lorentz vector with cartesian internal representation
  typedef PtEtaPhiELorentzVectorD PtEtaPhiELorentzVector;
  /// Lorentz vector with cylindrical internal representation using pseudorapidity
  typedef XYZTLorentzVectorD XYZTLorentzVector;
}

#endif
