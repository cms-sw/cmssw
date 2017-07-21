#ifndef DataFormats_DeepFormats_DeepFlavourTagInfo_h
#define DataFormats_DeepFormats_DeepFlavourTagInfo_h

#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/DeepFormats/interface/DeepFlavourFeatures.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

namespace reco {

template<class Features> class FeaturesTagInfo : public BaseTagInfo {

  public:

    FeaturesTagInfo() {} 

    FeaturesTagInfo(const Features & features,
                    const  edm::RefToBase<Jet> & jet_ref) :
      features_(features),
      jet_ref_(jet_ref) {}

    virtual edm::RefToBase<Jet> jet() const { return jet_ref_; }

    const Features & features() const { return features_; } ; 

  private:
    Features features_;
    edm::RefToBase<Jet> jet_ref_;
};

typedef  FeaturesTagInfo<deep::DeepFlavourFeatures> DeepFlavourTagInfo;

DECLARE_EDM_REFS( DeepFlavourTagInfo )

}

#endif // DataFormats_DeepFormats_DeepFlavourTagInfo_h
