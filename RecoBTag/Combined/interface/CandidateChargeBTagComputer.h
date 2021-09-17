#ifndef RecoBTag_Combined_CandidateChargeBTagComputer_h
#define RecoBTag_Combined_CandidateChargeBTagComputer_h

#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/MVAUtils/interface/TMVAEvaluator.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class CandidateChargeBTagComputer : public JetTagComputer {
public:
  struct Tokens {
    Tokens(const edm::ParameterSet &parameters, edm::ESConsumesCollector &&cc);
    edm::ESGetToken<GBRForest, GBRWrapperRcd> gbrForest_;
  };

  CandidateChargeBTagComputer(const edm::ParameterSet &parameters, Tokens tokens);
  ~CandidateChargeBTagComputer() override;
  void initialize(const JetTagComputerRecord &record) override;
  float discriminator(const TagInfoHelper &tagInfo) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  const edm::FileInPath weightFile_;
  const bool useAdaBoost_;
  std::unique_ptr<TMVAEvaluator> mvaID;
  const double jetChargeExp_;
  const double svChargeExp_;
  const Tokens tokens_;
};

#endif  // RecoBTag_Combined_CandidateChargeBTagComputer_h
