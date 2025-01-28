#ifndef RecoBTag_SoftLepton_LeptonSelector_h
#define RecoBTag_SoftLepton_LeptonSelector_h

#include <string>

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace btag {

  class LeptonSelector {
  public:
    LeptonSelector(const edm::ParameterSet &params);
    ~LeptonSelector();

    bool operator()(const reco::SoftLeptonProperties &properties, bool use3d = true) const;

    inline bool isAny() const { return m_sign == any; }
    inline bool isPositive() const { return m_sign == positive; }
    inline bool isNegative() const { return m_sign == negative; }

    static void fillPSetDescription(edm::ParameterSetDescription &desc);

  private:
    /// optionally select leptons based on their impact parameter sign

    enum sign { negative = -1, any = 0, positive = 1 };

    static sign option(const std::string &election);

    sign m_sign;
    reco::SoftLeptonProperties::Quality::Generic m_leptonId;
    float m_qualityCut;
  };

}  // namespace btag

#endif  // RecoBTag_SoftLepton_LeptonSelector_h
