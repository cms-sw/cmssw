#include "RecoJets/JetAlgorithms/interface/KtUtil.h"
#include "RecoJets/JetAlgorithms/interface/KtLorentzVector.h"
#include "CLHEP/config/CLHEP.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/LorentzVector.h"
#include <cmath>

namespace KtJet {
  /** Put phi in range [-pi,+pi]. No such function in CLHEP 1.7. (But is in 1.8.)
   */
KtFloat phiAngle(KtFloat testphi) {
  KtFloat phi = testphi;
  while (phi>M_PI) phi -= (2*M_PI);
  while (phi<-M_PI) phi += (2*M_PI);
  return phi;
}

}//end of namespace
