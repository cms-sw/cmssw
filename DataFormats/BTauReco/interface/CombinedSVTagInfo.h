#ifndef DataFormats_BTauReco_CombinedSVTagInfo_h
#define DataFormats_BTauReco_CombinedSVTagInfo_h

#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/BTauReco/interface/VertexTypes.h"
// #include "RecoBTag/CombinedSVTagInfo/interface/CombinedData.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/JetTagFwd.h"

namespace reco {
  class CombinedSVTagInfo : public JTATagInfo {
  public:
    /**
     *  The tag object of the combined btagger. Holds
     *  the tagging variables, and the discriminator.
     */
    CombinedSVTagInfo( const reco::TaggingVariableList &,
                 double discriminator );

    CombinedSVTagInfo();

    virtual ~CombinedSVTagInfo();

    float discriminator() const;
    const TaggingVariableList & variables() const;
    virtual CombinedSVTagInfo * clone() const;

  private:
    reco::TaggingVariableList vars_;
    double discriminator_;
  };
}

#endif
