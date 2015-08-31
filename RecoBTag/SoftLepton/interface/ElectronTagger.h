#ifndef RecoBTag_SoftLepton_ElectronTagger_h
#define RecoBTag_SoftLepton_ElectronTagger_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/TMVAEvaluator.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include <mutex>

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
  virtual float discriminator(const TagInfoHelper & tagInfo) const override;

private:
  btag::LeptonSelector m_selector;
  mutable std::mutex m_mutex;
  [[cms::thread_guard("m_mutex")]] std::unique_ptr<TMVAEvaluator> mvaID;
};

#endif
