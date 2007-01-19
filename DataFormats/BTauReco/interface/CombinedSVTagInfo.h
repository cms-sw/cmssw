#ifndef DataFormats_BTauReco_CombinedSVTagInfo_h
#define DataFormats_BTauReco_CombinedSVTagInfo_h

#include "DataFormats/BTauReco/interface/VertexTypes.h"
// #include "RecoBTag/CombinedSVTagInfo/interface/CombinedData.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/JetTagFwd.h"

namespace reco {
  class CombinedSVTagInfo {
  public:
    /**
     *  The tag object of the combined btagger. Holds
     *  the tagging variables, and the discriminator.
     */
    CombinedSVTagInfo( const reco::TaggingVariableList &,
                 double discriminator );

    CombinedSVTagInfo();

    virtual ~CombinedSVTagInfo();

    double discriminator() const;
    const reco::TaggingVariableList & variables() const;
    virtual CombinedSVTagInfo * clone() const;
    void setJetTag ( const JetTagRef ref );

  private:
    reco::TaggingVariableList vars_;
    double discriminator_;
    reco::JetTagRef basetag_;
  };
}

#endif
