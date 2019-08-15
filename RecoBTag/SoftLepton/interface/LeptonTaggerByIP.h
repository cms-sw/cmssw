#ifndef RecoBTag_SoftLepton_LeptonTaggerByIP_h
#define RecoBTag_SoftLepton_LeptonTaggerByIP_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"

/**  \class LeptonTaggerByIP
 *
 *   Implementation of muon b-tagging cutting on the lepton's transverse momentum relative to the jet axis
 *
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class LeptonTaggerByIP : public JetTagComputer {
public:
  using Tokens = void;

  /// explicit ctor
  explicit LeptonTaggerByIP(const edm::ParameterSet& configuration)
      : m_use3d(configuration.getParameter<bool>("use3d")), m_selector(configuration) {
    uses("slTagInfos");
  }

  /// dtor
  ~LeptonTaggerByIP() override {}

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  float discriminator(const TagInfoHelper& tagInfo) const override;

private:
  bool m_use3d;

  btag::LeptonSelector m_selector;
};

#endif  // RecoBTag_SoftLepton_LeptonTaggerByIP_h
