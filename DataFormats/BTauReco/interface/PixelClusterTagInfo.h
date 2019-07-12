#ifndef DataFormats_BTauReco_PixelClusterTagInfo_h
#define DataFormats_BTauReco_PixelClusterTagInfo_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace reco {

struct PixelClusterProperties {
    float x;
    float y;
    float z;
    int charge;
    int layer;
};

struct PixelClusterData {
    char L1_R004;
    char L2_R004;
    char L3_R004;
    char L4_R004;
    
    char L1_R006;
    char L2_R006;
    char L3_R006;
    char L4_R006;
    
    char L1_R008;
    char L2_R008;
    char L3_R008;
    char L4_R008;
    
    char L1_R010;
    char L2_R010;
    char L3_R010;
    char L4_R010;
    
    char L1_R016;
    char L2_R016;
    char L3_R016;
    char L4_R016;
    
    char L1_RVAR;
    char L2_RVAR;
    char L3_RVAR;
    char L4_RVAR;

    unsigned int L1_RVWT;
    unsigned int L2_RVWT;
    unsigned int L3_RVWT;
    unsigned int L4_RVWT;
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

        CMS_CLASS_VERSION(3)

    private:
    
        PixelClusterData pixelClusters;
        
        edm::RefToBase<Jet> jetRef;
    
};

typedef std::vector<reco::PixelClusterTagInfo> PixelClusterTagInfoCollection;

}

#endif // DataFormats_BTauReco_PixelClusterTagInfo_h
