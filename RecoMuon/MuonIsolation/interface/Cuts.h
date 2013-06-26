#ifndef MuonIsolation_Cuts_H
#define MuonIsolation_Cuts_H

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include <vector>
#include <string>

namespace edm { class ParameterSet; }

namespace muonisolation {

class Cuts {
public:

  struct CutSpec { muonisolation::Range<double> etaRange; double conesize; double threshold; }; 

  /// dummy constructor
  Cuts(){}

  /// ctor by PSet
  Cuts(const edm::ParameterSet & pset);

  /// constructor from valid parameters 
  Cuts( const std::vector<double> & etaBounds, 
        const std::vector<double> & coneSizes, 
        const std::vector<double> & thresholds);

  const CutSpec & operator()(double eta) const;

  const CutSpec & operator[](unsigned int i) const {return theCuts[i];};

  unsigned int size() {return theCuts.size();};

  std::string print() const;

private:
  void init(
      const std::vector<double> & etaBounds,
      const std::vector<double> & coneSizes,
      const std::vector<double> & thresholds);

  std::vector<CutSpec> theCuts;
};

}
#endif
