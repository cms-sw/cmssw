#ifndef RecoBTag_SoftLepton_LeptonTaggerByPt_h
#define RecoBTag_SoftLepton_LeptonTaggerByPt_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"

/**  \class LeptonTaggerByPt
 *
 *   Implementation of muon b-tagging cutting on the lepton's transverse momentum relative to the jet axis
 *
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class LeptonTaggerByPt : public JetTagComputer {
public:
  using Tokens = void;

  /// explicit ctor
  explicit LeptonTaggerByPt(const edm::ParameterSet& configuration) : m_selector(configuration) { uses("slTagInfos"); }

  /// dtor
  ~LeptonTaggerByPt() override {}

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  float discriminator(const TagInfoHelper& tagInfo) const override;

private:
  btag::LeptonSelector m_selector;
};

#endif  // RecoBTag_SoftLepton_LeptonTaggerByPt_h
