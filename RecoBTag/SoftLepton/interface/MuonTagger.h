// * Author: Alberto Zucchetta
// * Mail: a.zucchetta@cern.ch
// * January 16, 2015

#ifndef RecoBTag_SoftLepton_MuonTagger_h
#define RecoBTag_SoftLepton_MuonTagger_h

#include <memory>

#include "CommonTools/MVAUtils/interface/TMVAEvaluator.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

class MuonTagger : public JetTagComputer {
public:
  struct Tokens {
    Tokens(const edm::ParameterSet& cfg, edm::ESConsumesCollector&& cc);
    edm::ESGetToken<GBRForest, GBRWrapperRcd> gbrForest_;
  };

  MuonTagger(const edm::ParameterSet&, Tokens);
  void initialize(const JetTagComputerRecord&) override;
  float discriminator(const TagInfoHelper& tagInfo) const override;

  static void fillPSetDescription(edm::ParameterSetDescription& desc);

private:
  btag::LeptonSelector m_selector;
  const edm::FileInPath m_weightFile;
  const bool m_useGBRForest;
  const bool m_useAdaBoost;
  const Tokens m_tokens;

  std::unique_ptr<TMVAEvaluator> mvaID;
};

#endif
