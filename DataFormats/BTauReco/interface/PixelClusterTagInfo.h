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
    char L1_R004 = 0;
    char L2_R004 = 0;
    char L3_R004 = 0;
    char L4_R004 = 0;
    
    char L1_R006 = 0;
    char L2_R006 = 0;
    char L3_R006 = 0;
    char L4_R006 = 0;
    
    char L1_R008 = 0;
    char L2_R008 = 0;
    char L3_R008 = 0;
    char L4_R008 = 0;
    
    char L1_R010 = 0;
    char L2_R010 = 0;
    char L3_R010 = 0;
    char L4_R010 = 0;
    
    char L1_R016 = 0;
    char L2_R016 = 0;
    char L3_R016 = 0;
    char L4_R016 = 0;
    
    char L1_RVAR = 0;
    char L2_RVAR = 0;
    char L3_RVAR = 0;
    char L4_RVAR = 0;

    unsigned int L1_RVWT = 0;
    unsigned int L2_RVWT = 0;
    unsigned int L3_RVWT = 0;
    unsigned int L4_RVWT = 0;
};


class PixelClusterTagInfo : public BaseTagInfo {

    public:
      
        PixelClusterTagInfo() {}

        PixelClusterTagInfo(const PixelClusterData& data, const edm::RefToBase<Jet>& jet_ref):
          pixelClusters(data),
          jetRef(jet_ref) {}

        ~PixelClusterTagInfo() override {}
        
        // without overriding clone from base class will be store/retrieved
        PixelClusterTagInfo* clone(void) const override { return new PixelClusterTagInfo(*this); }


        // method to set the jet RefToBase
        void setJetRef( const edm::RefToBase<Jet>& ref) { jetRef = ref; }
        
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

}

#endif // DataFormats_BTauReco_PixelClusterTagInfo_h
