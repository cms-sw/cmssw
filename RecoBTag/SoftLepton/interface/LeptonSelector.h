#ifndef RecoBTag_SoftLepton_LeptonSelector_h
#define RecoBTag_SoftLepton_LeptonSelector_h

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

namespace btag {

class LeptonSelector {
  public:
    LeptonSelector(const edm::ParameterSet &params);
    ~LeptonSelector();

    bool operator() (const reco::SoftLeptonProperties &properties, bool use3d = true) const;

    inline bool isAny()      const { return m_sign == any; }
    inline bool isPositive() const { return m_sign == positive; }
    inline bool isNegative() const { return m_sign == negative; }

  private:
    /// optionally select leptons based on their impact parameter sign

    enum sign {
      negative = -1,
      any      =  0,
      positive =  1
    };

    static sign option(const std::string & election);

    sign                                         m_sign;
    reco::SoftLeptonProperties::quality::Generic m_leptonId;
    float                                        m_qualityCut;
};

} // namespace btag

#endif // RecoBTag_SoftLepton_LeptonSelector_h
