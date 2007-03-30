#ifndef MuonIsolation_MuIsoBaseIsolator_H
#define MuonIsolation_MuIsoBaseIsolator_H

#include <vector>
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"

namespace muonisolation {
class MuIsoBaseIsolator {

public:
  typedef std::vector<const reco::MuIsoDeposit*> DepositContainer;

  virtual ~MuIsoBaseIsolator(){}

  /// Compute and return the isolation variable
  virtual float result(DepositContainer deposits) const = 0;
};
}
#endif

