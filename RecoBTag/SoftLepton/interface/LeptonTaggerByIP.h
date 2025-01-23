#ifndef RecoBTag_SoftLepton_LeptonTaggerByIP_h
#define RecoBTag_SoftLepton_LeptonTaggerByIP_h

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

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

  /// ctor
  LeptonTaggerByIP(const edm::ParameterSet&);

  /// dtor
  ~LeptonTaggerByIP() override {}

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  float discriminator(const TagInfoHelper& tagInfo) const override;

  static void fillPSetDescription(edm::ParameterSetDescription& desc);

private:
  bool m_use3d;
  btag::LeptonSelector m_selector;
};

#endif  // RecoBTag_SoftLepton_LeptonTaggerByIP_h
