#ifndef FML3PtSmearer_H
#define FML3PtSmearer_H

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"

/** \class FML3PtSmearer
 * Class to deal with the 'smearing' of the L3 muon transverse momentum.
 * The output momentum is generated according to the probablility that  
 * a MC generated muon with the same pt leads to that momentum value in
 * the GMT. 
 *
 * \author  Andrea Perrotta Date: 25/05/2007
 */

// class declarations

class RandomEngine;

class FML3PtSmearer {

public:

 /// Constructor
  FML3PtSmearer(const RandomEngine * engine);
 
  /// Destructor
  ~FML3PtSmearer();
 
  /// smear the transverse momentum of a reco::Muon
  math::XYZTLorentzVector smear(math::XYZTLorentzVector simP4 , math::XYZVector recP3) const;

private:

  static double MuonMassSquared_;

  const RandomEngine * random;
  /// smear the transverse momentum of a SimplL3MuGMTCand
  double error(double thePt, double theEta) const;
  double shift(double thePt, double theEta) const;
  double funShift(double x) const;
  double funSigma(double eta , double pt) const;
  double funSigmaPt(double x) const;
  double funSigmaEta(double x) const;


};

#endif
