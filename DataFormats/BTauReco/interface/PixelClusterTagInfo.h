#ifndef DataFormats_BTauReco_PixelClusterTagInfo_h
#define DataFormats_BTauReco_PixelClusterTagInfo_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

#include "DataFormats/BTauReco/interface/BaseTagInfo.h"

namespace reco {

  struct PixelClusterProperties {
    float x = 0;
    float y = 0;
    float z = 0;
    int charge = 0;
    unsigned int layer = 0;
  };

  struct PixelClusterData {
    std::vector<int8_t> r004;
    std::vector<int8_t> r006;
    std::vector<int8_t> r008;
    std::vector<int8_t> r010;
    std::vector<int8_t> r016;
    std::vector<int8_t> rvar;
    std::vector<unsigned int> rvwt;
    PixelClusterData(unsigned int l = 4) {
      r004 = std::vector<int8_t>(l, 0);
      r006 = std::vector<int8_t>(l, 0);
      r008 = std::vector<int8_t>(l, 0);
      r010 = std::vector<int8_t>(l, 0);
      r016 = std::vector<int8_t>(l, 0);
      rvar = std::vector<int8_t>(l, 0);
      rvwt = std::vector<unsigned int>(l, 0);
    }
    CMS_CLASS_VERSION(3)
  };

  class PixelClusterTagInfo : public BaseTagInfo {
  public:
    PixelClusterTagInfo() {}

    PixelClusterTagInfo(const PixelClusterData& data, const edm::RefToBase<Jet>& ref)
        : pixelClusters_(data), jetRef_(ref) {}

    ~PixelClusterTagInfo() override {}

    // without overriding clone from base class will be store/retrieved
    PixelClusterTagInfo* clone(void) const override { return new PixelClusterTagInfo(*this); }

    // method to set the jet RefToBase
    void setJetRef(const edm::RefToBase<Jet>& ref) { jetRef_ = ref; }

    // method to jet the jet RefToBase
    edm::RefToBase<Jet> jet() const override { return jetRef_; }

    // method to set the PixelClusterData
    void setData(const PixelClusterData& data) { pixelClusters_ = data; }

    // method to get the PixelClusterData struct
    const PixelClusterData& data() const { return pixelClusters_; }

    CMS_CLASS_VERSION(3)

  private:
    PixelClusterData pixelClusters_;

    edm::RefToBase<Jet> jetRef_;
  };

  typedef std::vector<reco::PixelClusterTagInfo> PixelClusterTagInfoCollection;

}  // namespace reco

#endif  // DataFormats_BTauReco_PixelClusterTagInfo_h
