#ifndef RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h
#define RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h

#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/MVAUtils/interface/TMVAEvaluator.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"

class CandidateBoostedDoubleSecondaryVertexComputer : public JetTagComputer {
public:
  struct Tokens {
    Tokens(const edm::ParameterSet &parameters, edm::ESConsumesCollector &&cc);
    edm::ESGetToken<GBRForest, GBRWrapperRcd> gbrForest_;
  };

  CandidateBoostedDoubleSecondaryVertexComputer(const edm::ParameterSet &parameters, Tokens tokens);

  void initialize(const JetTagComputerRecord &) override;
  float discriminator(const TagInfoHelper &tagInfos) const override;

private:
  const edm::FileInPath weightFile_;
  const bool useGBRForest_;
  const bool useAdaBoost_;
  const Tokens tokens_;

  std::unique_ptr<TMVAEvaluator> mvaID;
};

#endif  // RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h
