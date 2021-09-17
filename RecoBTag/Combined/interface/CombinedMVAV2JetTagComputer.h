#ifndef RecoBTau_JetTagComputer_CombinedMVAV2JetTagComputer_h
#define RecoBTau_JetTagComputer_CombinedMVAV2JetTagComputer_h

#include <string>
#include <memory>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/MVAUtils/interface/TMVAEvaluator.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

class CombinedMVAV2JetTagComputer : public JetTagComputer {
public:
  struct Tokens {
    Tokens(const edm::ParameterSet &parameters, edm::ESConsumesCollector &&cc);
    edm::ESGetToken<GBRForest, GBRWrapperRcd> gbrForest_;
    std::vector<edm::ESGetToken<JetTagComputer, JetTagComputerRecord> > computers_;
  };

  CombinedMVAV2JetTagComputer(const edm::ParameterSet &parameters, Tokens tokens);
  ~CombinedMVAV2JetTagComputer() override;

  void initialize(const JetTagComputerRecord &record) override;

  float discriminator(const TagInfoHelper &info) const override;

private:
  std::vector<const JetTagComputer *> computers;

  const std::string mvaName;
  const std::vector<std::string> variables;
  const std::vector<std::string> spectators;
  const edm::FileInPath weightFile;
  const bool useGBRForest;
  const bool useAdaBoost;
  const Tokens tokens;

  std::unique_ptr<TMVAEvaluator> mvaID;
};

#endif  // RecoBTau_JetTagComputer_CombinedMVAV2JetTagComputer_h
