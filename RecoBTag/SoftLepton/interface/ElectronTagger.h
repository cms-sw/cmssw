#ifndef RecoBTag_SoftLepton_ElectronTagger_h
#define RecoBTag_SoftLepton_ElectronTagger_h

#include "CommonTools/MVAUtils/interface/TMVAEvaluator.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

/** \class ElectronTagger
 *
 *
 *  \author P. Demin - UCL, Louvain-la-Neuve - Belgium
 *
 */

class ElectronTagger : public JetTagComputer {
public:
  struct Tokens {
    Tokens(const edm::ParameterSet &cfg, edm::ESConsumesCollector &&cc);
    edm::ESGetToken<GBRForest, GBRWrapperRcd> gbrForest_;
  };

  /// explicit ctor
  ElectronTagger(const edm::ParameterSet &, Tokens);
  void initialize(const JetTagComputerRecord &) override;
  float discriminator(const TagInfoHelper &tagInfo) const override;

  static void fillPSetDescription(edm::ParameterSetDescription &desc);

private:
  const btag::LeptonSelector m_selector;
  const edm::FileInPath m_weightFile;
  const bool m_useGBRForest;
  const bool m_useAdaBoost;
  const Tokens m_tokens;

  std::unique_ptr<TMVAEvaluator> mvaID;
};

#endif
