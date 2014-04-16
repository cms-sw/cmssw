#ifndef PFJETMVATOOLS_H
#define PFJETMVATOOLS_H

#include <vector>
#include "Math/VectorUtil.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

namespace HWWFunctions {

  typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> > LorentzVector;

  bool sortByPFJetPt (const std::pair <LorentzVector, Int_t> &pfjet1, const std::pair<LorentzVector, Int_t> &pfjet2);
  bool getGoodMVAs(HWW&, std::vector <float> &goodmvas, std::string variable);

}
#endif
