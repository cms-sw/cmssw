#ifndef KTJET_KTUTIL_H
#define KTJET_KTUTIL_H

#ifndef CLHEP_CLHEP_H
#include "CLHEP/config/CLHEP.h"
#define CLHEP_CLHEP_H
#endif
#ifndef CLHEP_THREEVECTOR_H
#include "CLHEP/Vector/ThreeVector.h"
#define CLHEP_THREEVECTOR_H
#endif
#ifndef CLHEP_LORENTZVECTOR_H
#include "CLHEP/Vector/LorentzVector.h"
#define CLHEP_LORENTZVECTOR_H
#endif

namespace KtJet{

  // always double precision is used (A.Heister 10.07.2003)
  //
  //#ifdef KTDOUBLEPRECISION
  //  typedef double KtFloat;
  //#else
  //  typedef float KtFloat;
  //#endif
typedef double KtFloat;
  //
  // always double precision is used (A.Heister 10.07.2003)

class KtLorentzVector;

/** Phi angle forced into range -pi to +pi */
KtFloat phiAngle(KtFloat testphi);

} //end of namespace
#endif
