#ifndef RecoPixelVertexing_PixelTriplets_src_CACut_h
#define RecoPixelVertexing_PixelTriplets_src_CACut_h

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CAGraph.h"


class CACut {
public:
    explicit CACut(const double defaultCut, const edm::ParameterSet& tripletCuts) {
        setCutValuesByTripletNames(defaultCut, tripletCuts);
    }

    void setCutValuesByTripletNames(double defaultCut, const edm::ParameterSet& tripletCuts) {
        std::vector<std::string> tripletNames = tripletCuts.getParameterNames();

        for(const std::string &thisTripletName : tripletNames) {

            CAValueByTripletName thisTriplet;
            thisTriplet.tripletName = thisTripletName;

            float thisCutValue = tripletCuts.getParameter<double>(thisTripletName);
            if (thisCutValue > 0) {
                thisTriplet.cutValue = thisCutValue;
            }
            else {
                //TODO: Uncomment the following line once the tuned values are added to the PSet
                //edm::LogWarning("Configuration") << "Layer triplet '" << tripletName <<"' not in the CACuts parameter set. Using default cut value: " << defaultCut;                
                thisTriplet.cutValue = defaultCut;
            }

            valuesByTripletNames_.emplace_back(thisTriplet);

        }

    }

    void setCutValuesByLayerIds(CAGraph &caLayers) {

        for(const auto &thisTriplet : valuesByTripletNames_) {

            CAValueByLayerIds thisCACut;

            // Triplet name, e.g. "layerA__layerB__layerC"
            std::string layersToSet = thisTriplet.tripletName; 

            // Layer names and id's            
            std::string layerName;
            std::size_t layerPos = 0;

            for(int thisLayer=0; thisLayer < 3; thisLayer++) {
                // Get layer name
                layerPos = layersToSet.find("__");
                layerName = layersToSet.substr(0, layerPos);
                layersToSet = layersToSet.substr(layerPos+2);

                // Get layer ID
                thisCACut.layerIds.emplace_back(caLayers.getLayerId(layerName));
                if (thisCACut.layerIds.at(thisLayer)==-1) {
                    edm::LogWarning("Configuration") << "Layer name '" << layerName <<"' not found in the CAGraph. Please enter a valid layer name in the CACuts parameter set";
                }
            }

            // Cut
            thisCACut.cutValue = thisTriplet.cutValue;

            // Add to vector
            valuesByLayerIds_.emplace_back(thisCACut);
        }

    }


    class CAValuesByInnerLayerIds {
    public:

        float at(int layerId) {
            for(size_t thisLayer = 0; thisLayer < layerIds.size(); thisLayer++ ) {
                if ( layerIds.at(thisLayer) == layerId) return cutValues.at(thisLayer);
             }

             return -1.;
        }

        std::vector<int> layerIds;
        std::vector<float> cutValues;
    };

    // Check all triplets with outer cell (layerId1, layerId2) and return a map of (layerId0, cut)
    CAValuesByInnerLayerIds getCutsByInnerLayer(int layerIds1, int layerIds2) const {

        CAValuesByInnerLayerIds cutsByInnerLayer;

        for(const auto &thisCut : valuesByLayerIds_) {
            if(thisCut.layerIds[1] == layerIds1 && thisCut.layerIds[2] == layerIds2) {
                cutsByInnerLayer.layerIds.emplace_back(thisCut.layerIds[0]);
                cutsByInnerLayer.cutValues.emplace_back(thisCut.cutValue);
            }
        }

        return cutsByInnerLayer;
    }


private:
    class CAValueByTripletName {
    public:
       std::string tripletName;
       float cutValue;
    };
 
    class CAValueByLayerIds {
    public:
        std::vector<int> layerIds;
        float cutValue;
    }; 

private: 
    std::vector<CAValueByTripletName>  valuesByTripletNames_;
    std::vector<CAValueByLayerIds>     valuesByLayerIds_;    
};

#endif
