// Authors: todo

#ifndef RecoLocalCalo_HGCalRecProducers_HGCalCLUEStrategy_h
#define RecoLocalCalo_HGCalRecProducers_HGCalCLUEStrategy_h

class HGCalSiliconStrategy{
public:
    float distance2(float dim1Cell1, float dim2Cell1, float dim1Cell2, float dim2Cell2) const { 
        const float d1 = dim1Cell1 - dim1Cell2;
        const float d2 = dim2Cell1 - dim2Cell2;
        return (d1 * d1 + d2 * d2);
    }
    //todo remove
    bool isSi(){
        return true;
    }
    std::string type(){
        return "Silicon";
    }
};

class HGCalScintillatorStrategy{
public:
    float distance2(float dim1Cell1, float dim2Cell1, float dim1Cell2, float dim2Cell2) const { 
        const float dphi = reco::deltaPhi(dim2Cell1, dim2Cell2);
        const float deta = dim1Cell1 - dim1Cell2;
        return (deta * deta + dphi * dphi);
    }
    //todo remove
    bool isSi(){
        return false;
    }
    std::string type(){
        return "Scintilator";
    }
};


#endif