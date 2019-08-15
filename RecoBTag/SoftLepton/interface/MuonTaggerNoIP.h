#ifndef RecoBTag_SoftLepton_MuonTaggerNoIP_h
#define RecoBTag_SoftLepton_MuonTaggerNoIP_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/MuonTaggerNoIPMLP.h"

/**  \class MuonTagger
 *
 *   Implementation of muon b-tagging using a softmax multilayer perceptron neural network
 *
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class MuonTaggerNoIP : public JetTagComputer {
public:
  using Tokens = void;

  /// explicit ctor
  explicit MuonTaggerNoIP(const edm::ParameterSet& configuration) : m_selector(configuration) { uses("slTagInfos"); }

  /// dtor
  ~MuonTaggerNoIP() override {}

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  float discriminator(const TagInfoHelper& tagInfo) const override;

private:
  btag::LeptonSelector m_selector;
};

#endif  // RecoBTag_SoftLepton_MuonTaggerNoIP_h
