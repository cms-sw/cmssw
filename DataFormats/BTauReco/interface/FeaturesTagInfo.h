#ifndef DataFormats_BTauReco_FeaturesTagInfo_h
#define DataFormats_BTauReco_FeaturesTagInfo_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

namespace reco {

template<class Features> class FeaturesTagInfo : public BaseTagInfo {

  public:

    FeaturesTagInfo() {}

    FeaturesTagInfo(const Features & features,
                    const  edm::RefToBase<Jet> & jet_ref) :
      features_(features),
      jet_ref_(jet_ref) {}

    edm::RefToBase<Jet> jet() const override { return jet_ref_; }

    const Features & features() const { return features_; }

    ~FeaturesTagInfo() override {}
    // without overidding clone from base class will be store/retrieved
    FeaturesTagInfo* clone(void) const override { return new FeaturesTagInfo(*this); }


    CMS_CLASS_VERSION(3)

  private:
    Features features_;
    edm::RefToBase<Jet> jet_ref_;
};

}

#endif // DataFormats_BTauReco_FeaturesTagInfo_h
