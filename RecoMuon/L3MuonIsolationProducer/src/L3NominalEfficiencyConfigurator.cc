#include "L3NominalEfficiencyConfigurator.h"
#include "RecoMuon/MuonIsolation/interface/IsolatorByNominalEfficiency.h"

using namespace muonisolation;

L3NominalEfficiencyConfigurator::L3NominalEfficiencyConfigurator(const edm::ParameterSet& pset)
    : theConfig(pset), theWeights(std::vector<double>(1, 1.)) {
  std::string name = theConfig.getParameter<std::string>("ComponentName");
  std::string lumi = theConfig.getParameter<std::string>("LumiOption");

  std::string dir = "RecoMuon/L3MuonIsolationProducer/data/";
  if (name == "L3NominalEfficiencyCuts_PXLS") {
    if (lumi == "2E33") {
      theFileName = dir + "L3Pixel_PTDR_2x1033.dat";
      theBestCones = std::vector<std::string>(1, "8:0.97");
    }
  } else if (name == "L3NominalEfficiencyCuts_TRKS") {
  } else {
  }
}

Cuts L3NominalEfficiencyConfigurator::cuts() const

{
  IsolatorByNominalEfficiency nomEff(theFileName, theBestCones, theWeights);
  double threshold = theConfig.getParameter<double>("NominalEfficiency");
  return nomEff.cuts(threshold);
}
