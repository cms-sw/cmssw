#ifndef RecoBTag_SoftLepton_ElectronTagger_h
#define RecoBTag_SoftLepton_ElectronTagger_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/TMVAEvaluator.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"

/** \class ElectronTagger
 *
 *
 *  \author P. Demin - UCL, Louvain-la-Neuve - Belgium
 *
 */

class ElectronTagger : public JetTagComputer {
public:

  /// explicit ctor 
  ElectronTagger(const edm::ParameterSet & );
  void initialize(const JetTagComputerRecord &) override;
  virtual float discriminator(const TagInfoHelper & tagInfo) const override;

private:
  const btag::LeptonSelector m_selector;
  const bool m_useCondDB;
  const std::string m_gbrForestLabel;
  const edm::FileInPath m_weightFile;
  const bool m_useGBRForest;
  const bool m_useAdaBoost;

  std::unique_ptr<TMVAEvaluator> mvaID;
};

#endif
