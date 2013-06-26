#ifndef MuonIsolation_CutsConeSizeFunction_H
#define MuonIsolation_CutsConeSizeFunction_H

#include "RecoMuon/MuonIsolation/interface/Cuts.h"
#include "RecoMuon/MuonIsolation/interface/IsolatorByDeposit.h"

namespace muonisolation {
class CutsConeSizeFunction : public IsolatorByDeposit::ConeSizeFunction {
public: 
  CutsConeSizeFunction(const Cuts & cuts) : theLastCut(0), theCuts(cuts) {} 
  float threshold() const { return theLastCut->threshold; }
  float coneSize( float eta, float pt) const {
    theLastCut = & theCuts(eta); 
    return theLastCut->conesize;
  } 
private:
  mutable const Cuts::CutSpec * theLastCut;
  const Cuts & theCuts;
};
}
#endif
