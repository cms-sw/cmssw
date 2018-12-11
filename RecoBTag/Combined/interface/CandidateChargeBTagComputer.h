#ifndef RecoBTag_Combined_CandidateChargeBTagComputer_h
#define RecoBTag_Combined_CandidateChargeBTagComputer_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
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
  CandidateChargeBTagComputer(const edm::ParameterSet &parameters);
  ~CandidateChargeBTagComputer() override;
  void initialize(const JetTagComputerRecord & record) override;
  float discriminator(const TagInfoHelper & tagInfo) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
 private:
  const bool useCondDB_;
  const std::string gbrForestLabel_;
  const edm::FileInPath weightFile_;
  const bool useAdaBoost_;
  std::unique_ptr<TMVAEvaluator> mvaID;
  const double jetChargeExp_;
  const double svChargeExp_;
};

#endif // RecoBTag_Combined_CandidateChargeBTagComputer_h
