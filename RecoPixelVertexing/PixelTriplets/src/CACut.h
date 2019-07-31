#ifndef RecoPixelVertexing_PixelTriplets_src_CACut_h
#define RecoPixelVertexing_PixelTriplets_src_CACut_h

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CAGraph.h"

//TODO: Make member variables private and member functions public // Have also private member funcions!
// Added a type to the getter and setter functions. // Done!
// Added a constructor to the class // Done!
// Movedthe LayerIds outside the class? // Done!
// Add a default constructor with the single value for caCut // Done!
//TODO: Make sure all member variables are initialized //TODO!!!
// Add an initializer list to CACut, where the two psets and the default are passed // Done!


class CACut {
public:
    //using CAValuesByTripletNames = std::map<std::string,float>; //Replaced by class

    // Constructor
    explicit CACut(const double defaultCut, const edm::ParameterSet& tripletCuts) {
        setCutValuesByTripletNames(tripletCuts, defaultCut);
    }

    // Setters
    // TODO: Remove the first and last functions once everything is working!

    // Not needed... replaced by CALayer vector
/*    void setLayerNamesAndIds(const edm::ParameterSet& caLayers) //:
    //layerNames_(caLayers.getParameterNames()) {
    {
        std::vector<std::string> layerNames = caLayers.getParameterNames();

        // Debugging:
        //std::cout << "Setting layer Name-ID association map" << std::endl;

        for(const std::string &layerName : layerNames) {
            layerNamesAndIds_.emplace(layerName, caLayers.getParameter<int>(layerName));
        }

        // Debugging:
        //for(const auto &thisLayerNameAndId : layerNamesAndIds_) {
        //    std::cout << "Layer '" << thisLayerNameAndId.first << "' has id " << thisLayerNameAndId.second << std::endl;
        //}

    }
*/
/*
    void setCutValuesByTripletNames(const edm::ParameterSet& tripletCuts, double defaultCut) {
        std::vector<std::string> tripletNames = tripletCuts.getParameterNames();

        // Debugging:
        std::cout << "Setting CA cut values by triplet name" << std::endl;

        for(const std::string &tripletName : tripletNames) {
            float thisValue = tripletCuts.getParameter<double>(tripletName);
            if (thisValue > 0) {
                valuesByTripletNames_.emplace(tripletName, thisValue);
            }
            else {
                //TODO: Remove the following line?
                //edm::LogWarning("Configuration") << "Layer triplet '" << tripletName <<"' not in the CACuts parameter set. Using default cut value: " << defaultCut;                
                valuesByTripletNames_.emplace(tripletName, defaultCut);
            }
        }

        // Debugging:
        for(const auto &thisTripletNameAndCut : valuesByTripletNames_) {
            std::cout << "Triplet '" << thisTripletNameAndCut.first << "' has a cut value of " << thisTripletNameAndCut.second << std::endl;
        }
    }
*/

    void setCutValuesByTripletNames(const edm::ParameterSet& tripletCuts, double defaultCut) {
        std::vector<std::string> tripletNames = tripletCuts.getParameterNames();

        // Debugging:
        //std::cout << "Setting CA cut values by triplet name" << std::endl;

        for(const std::string &thisTripletName : tripletNames) {

            CAValueByTripletName thisTriplet;
            thisTriplet.tripletName = thisTripletName;

            float thisCutValue = tripletCuts.getParameter<double>(thisTripletName);
            if (thisCutValue > 0) {
                thisTriplet.cutValue = thisCutValue;
            }
            else {
                //TODO: Remove the following line?
                //edm::LogWarning("Configuration") << "Layer triplet '" << tripletName <<"' not in the CACuts parameter set. Using default cut value: " << defaultCut;                
                thisTriplet.cutValue = defaultCut;
            }

            valuesByTripletNames_.emplace_back(thisTriplet);

        }

        // Debugging:
        //for(const auto &thisTripletNameAndCut : valuesByTripletNames_) {
        //    std::cout << "Triplet '" << thisTripletNameAndCut.tripletName << "' has a cut value of " << thisTripletNameAndCut.cutValue << std::endl;
        //}
    }

/*
    void setCutValuesByLayerIds(CAGraph &caLayers) {

        //std::cout << "Setting CA cut values by layer ID's" << std::endl;

        for(const auto &thisTriplet : valuesByTripletNames_) {

            CAValueByLayerIds thisCACut;

            // Triplet name, e.g. "layerA__layerB__layerC"
            std::string layersToSet = thisTriplet.first; //TODO Replace by vector element
            // Debugging:
            //std::cout << "Debugging: " << layersToSet << std::endl;

            // Layer names and id's            
            std::string layerName;
            std::size_t layerPos = 0;

            for(int thisLayer=0; thisLayer < 3; thisLayer++) {
                // Get layer name
                layerPos = layersToSet.find("__");
                layerName = layersToSet.substr(0, layerPos);
                layersToSet = layersToSet.substr(layerPos+2);

                // Debugging:
                // std::cout << layerName << " = ";

                // Get layer ID
                //std::cout << layerName << std::endl;
                thisCACut.layerNames.at(thisLayer) = layerName;//TODO Replace map by vector element
                thisCACut.layerIds.at(thisLayer) = caLayers.getLayerId(layerName);//TODO Replace map by vector element
                if (thisCACut.layerIds.at(thisLayer)==-1) {//TODO Replace map by vector element
                    edm::LogWarning("Configuration") << "Layer name '" << layerName <<"' not found in the CAGraph. Please enter a valid layer name in the CACuts parameter set";
                    //throw cms::Exception("Configuration") << "Layer name '" << layerName <<"' not found in the CAGraph. Please enter a valid layer name in the CACuts parameter set";
                }
                // Debugging:
                // std::cout << caLayers.getLayerId(layerName);
                // std::cout << std::endl;
            }

            // Cut
            thisCACut.cutValue = thisTriplet.second;//TODO Replace by vector element

            // Add to map
            valuesByLayerIds_.emplace_back(thisCACut);
        }

        // Debugging:
        //for(const auto &thisCut : valuesByLayerIds_) {
        //   std::cout << "Layer ids (" << thisCut.layerIds[0] << ", " << thisCut.layerIds[1] << ", " << thisCut.layerIds[2];
        //    std::cout << ") have a cut value of " << thisCut.cutValue << std::endl;
        //}
    }
*/

    void setCutValuesByLayerIds(CAGraph &caLayers) {

        //std::cout << "Setting CA cut values by layer ID's" << std::endl;

        for(const auto &thisTriplet : valuesByTripletNames_) {

            CAValueByLayerIds thisCACut;

            // Triplet name, e.g. "layerA__layerB__layerC"
            std::string layersToSet = thisTriplet.tripletName; //Replaced map by class
            // Debugging:
            //std::cout << "Debugging: " << layersToSet << std::endl;

            // Layer names and id's            
            std::string layerName;
            std::size_t layerPos = 0;

            for(int thisLayer=0; thisLayer < 3; thisLayer++) {
                // Get layer name
                layerPos = layersToSet.find("__");
                layerName = layersToSet.substr(0, layerPos);
                layersToSet = layersToSet.substr(layerPos+2);

                // Debugging:
                //std::cout << layerName << " = ";

                // Get layer ID
                thisCACut.layerNames.emplace_back(layerName);//Replaced array access by vector emplace_back
                thisCACut.layerIds.emplace_back(caLayers.getLayerId(layerName));//Replaced array access by vector emplace_back
                if (thisCACut.layerIds.at(thisLayer)==-1) {
                    edm::LogWarning("Configuration") << "Layer name '" << layerName <<"' not found in the CAGraph. Please enter a valid layer name in the CACuts parameter set";
                    //throw cms::Exception("Configuration") << "Layer name '" << layerName <<"' not found in the CAGraph. Please enter a valid layer name in the CACuts parameter set";
                }
                // Debugging:
                //std::cout << caLayers.getLayerId(layerName);
                //std::cout << std::endl;
            }

            // Cut
            thisCACut.cutValue = thisTriplet.cutValue;//Replaced map by class

            // Debugging: 
            //std::cout << "Cut value: " << thisTriplet.cutValue << std::endl;

            // Add to map
            valuesByLayerIds_.emplace_back(thisCACut);
        }

        // Debugging:
        //for(const auto &thisCut : valuesByLayerIds_) {
            //std::cout << "Layer ids (" << thisCut.layerIds[0] << ", " << thisCut.layerIds[1] << ", " << thisCut.layerIds[2];
            //std::cout << ") have a cut value of " << thisCut.cutValue << std::endl;
        //}
    }

    // Getters
    //CANameIdAssociationMap getAssociationMap() const { return layerNamesAndIds_; }
    //std::map<std::string,float> getValuesByTripletNames() const { return valuesByTripletNames_; }
    //std::vector<CAValueByLayerIds> getValuesByLayerIds() const { return valuesByLayerIds_; }

/*
    int getLayerId(std::string layerName) const { 
        if(layerNamesAndIds_.count(layerName) > 0) {
            return layerNamesAndIds_.at(layerName); 
        }
        else {
            throw cms::Exception("Configuration") << "Layer name '" << layerName <<"' not in the CALayerIds parameter set";
            //std::cout << "Layer name '" << layerName << "' not in Parameter Set" << std::endl;
            return -1;
        }
    }
*/
    // Check all triplets with outer cell (layerName1, layerName2) and return a map of (layerId0, cut)
    //TODO: Replace this with a class! Should be straightforward (replace type, define dummy and change assignments)
/*
    std::map<int,float> getCutsByInnerLayer(std::string layer1, std::string layer2) const { 
        
        std::map<int,float> cutsByInnerLayer;

        for(const auto &thisCut : valuesByLayerIds_) {
            if(thisCut.layerNames[1] == layer1 && thisCut.layerNames[2] == layer2) cutsByInnerLayer.emplace(thisCut.layerIds[0], thisCut.cutValue);
        }

        return cutsByInnerLayer;
    }

    std::map<int,float> getCutsByInnerLayer(int layer1, int layer2) const {

        std::map<int,float> cutsByInnerLayer;

        for(const auto &thisCut : valuesByLayerIds_) {
            if(thisCut.layerIds[1] == layer1 && thisCut.layerIds[2] == layer2) cutsByInnerLayer.emplace(thisCut.layerIds[0], thisCut.cutValue);
        }

        return cutsByInnerLayer;
    }
*/

    class CAValueByInnerLayerId {
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

    CAValueByInnerLayerId getCutsByInnerLayer(int layer1, int layer2) const {

        CAValueByInnerLayerId cutsByInnerLayer;

        for(const auto &thisCut : valuesByLayerIds_) {
            if(thisCut.layerIds[1] == layer1 && thisCut.layerIds[2] == layer2) {
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
 
    class CAValueByLayerIds { // Stores one cut parameter and the three layer names & id's
    public:

        std::vector<int> layerIds;
        std::vector<std::string> layerNames;
        //std::array<int,3> layerIds;
        //std::array<std::string,3> layerNames;
        float cutValue;

    }; 
/*
    class CAValueByInnerLayerId {
    public:

       std::vector<int> layerIds;
       std::vector<float> cutValues;
    };
*/

private: // Private member variables
    //const std::vector<std::string>     tripletNames_;//TODO Bring this back?
    //std::vector<float>                 cutValuesByTripletNames_;//TODO Add this?

    //const std::vector<std::string>     layerNames_;//TODO Remove this?


    std::vector<CAValueByTripletName>  valuesByTripletNames_;
    std::vector<CAValueByLayerIds>     valuesByLayerIds_;    
    //std::vector<CAValueByInnerLayerId> valuesByInnerLayerIds_;//TODO Add this?

    //TODO Copy getCutsByInnerLayer function structure to access the information from former maps using these classes!

};

#endif
