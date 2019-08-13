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
    int8_t R004[4] = {};
    int8_t R006[4] = {};
    int8_t R008[4] = {};
    int8_t R010[4] = {};
    int8_t R016[4] = {};
    int8_t RVAR[4] = {};
    unsigned int RVWT[4] = {};
  };

  class PixelClusterTagInfo : public BaseTagInfo {
  public:
    PixelClusterTagInfo() {}

    PixelClusterTagInfo(const PixelClusterData& data, const edm::RefToBase<Jet>& jet_ref)
        : pixelClusters(data), jetRef(jet_ref) {}

    ~PixelClusterTagInfo() override {}

    // without overriding clone from base class will be store/retrieved
    PixelClusterTagInfo* clone(void) const override { return new PixelClusterTagInfo(*this); }

    // method to set the jet RefToBase
    void setJetRef(const edm::RefToBase<Jet>& ref) { jetRef = ref; }

    // method to jet the jet RefToBase
    edm::RefToBase<Jet> jet() const override { return jetRef; }

    // method to set the PixelClusterData
    void setData(const PixelClusterData& data) { pixelClusters = data; }

    // method to get the PixelClusterData struct
    const PixelClusterData& data() const { return pixelClusters; }

  private:
    PixelClusterData pixelClusters;

    edm::RefToBase<Jet> jetRef;
  };

  typedef std::vector<reco::PixelClusterTagInfo> PixelClusterTagInfoCollection;

}  // namespace reco

#endif  // DataFormats_BTauReco_PixelClusterTagInfo_h
