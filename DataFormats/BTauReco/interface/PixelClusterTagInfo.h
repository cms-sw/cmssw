#ifndef DataFormats_BTauReco_PixelClusterTagInfo_h
#define DataFormats_BTauReco_PixelClusterTagInfo_h

#include "DataFormats/BTauReco/interface/BaseTagInfo.h"

namespace reco {

  struct PixelClusterProperties {
    float x = 0;
    float y = 0;
    float z = 0;
    int charge = 0;
    int layer = 0;
  };

  struct PixelClusterData {
    std::vector<int8_t> r004;
    std::vector<int8_t> r006;
    std::vector<int8_t> r008;
    std::vector<int8_t> r010;
    std::vector<int8_t> r016;
    std::vector<int8_t> rvar;
    std::vector<int8_t> rvwt;
    PixelClusterData(unsigned int l = 4) {
      r004.resize(l, 0);
      r006.resize(l, 0);
      r008.resize(l, 0);
      r010.resize(l, 0);
      r016.resize(l, 0);
      rvar.resize(l, 0);
      rvwt.resize(l, 0);
    }
  };

  class PixelClusterTagInfo : public BaseTagInfo {
  public:
    PixelClusterTagInfo() {}

    PixelClusterTagInfo(const PixelClusterData& data_, const edm::RefToBase<Jet>& ref_)
        : pixelClusters(data_), jetRef(ref_) {}

    ~PixelClusterTagInfo() override {}

    // without overriding clone from base class will be store/retrieved
    PixelClusterTagInfo* clone(void) const override { return new PixelClusterTagInfo(*this); }

    // method to set the jet RefToBase
    void setJetRef(const edm::RefToBase<Jet>& ref_) { jetRef = ref_; }

    // method to jet the jet RefToBase
    edm::RefToBase<Jet> jet() const override { return jetRef; }

    // method to set the PixelClusterData
    void setData(const PixelClusterData& data_) { pixelClusters = data_; }

    // method to get the PixelClusterData struct
    const PixelClusterData& data() const { return pixelClusters; }

  private:
    PixelClusterData pixelClusters;

    edm::RefToBase<Jet> jetRef;
  };

  typedef std::vector<reco::PixelClusterTagInfo> PixelClusterTagInfoCollection;

}  // namespace reco

#endif  // DataFormats_BTauReco_PixelClusterTagInfo_h
