#ifndef L3NominalEfficiencyConfigurator_H
#define L3NominalEfficiencyConfigurator_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoMuon/MuonIsolation/interface/Cuts.h"
#include <string>
#include <vector>

class L3NominalEfficiencyConfigurator {
public:
  L3NominalEfficiencyConfigurator(const edm::ParameterSet & pset); 
  muonisolation::Cuts cuts() const;
private:
  edm::ParameterSet theConfig;
  std::vector<std::string> theBestCones;
  std::vector<double> theWeights;
  std::string theFileName; 
   
};
#endif
