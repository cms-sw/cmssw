/**
 *  See header file for a description of this class.
 *
 */


#include "RecoMuon/MuonSeedGenerator/src/RPCSeedLayerFinder.h"

using namespace std;
using namespace edm;


RPCSeedLayerFinder::RPCSeedLayerFinder() {

    // Initiate the member
    LayersinRPC.clear();  
    isConfigured = false;
    isInputset = false;
    isOutputset = false;
}

RPCSeedLayerFinder::~RPCSeedLayerFinder() {

}

void RPCSeedLayerFinder::configure(const edm::ParameterSet& iConfig) {

    // Set the configuration
    isCosmic = iConfig.getParameter<bool>("isCosmic");
    isMixBarrelwithEndcap = iConfig.getParameter<bool>("isMixBarrelwithEndcap");
    RangeofLayersinBarrel = iConfig.getParameter< std::vector<unsigned int> >("RangeofLayersinBarrel");
    RangeofLayersinEndcap = iConfig.getParameter< std::vector<unsigned int> >("RangeofLayersinEndcap");
    isSpecialLayers = iConfig.getParameter<bool>("isSpecialLayers");
    LayersinBarrel = iConfig.getParameter< std::vector<unsigned int> >("LayersinBarrel");
    LayersinEndcap = iConfig.getParameter< std::vector<unsigned int> >("LayersinEndcap");
    constrainedLayersinBarrel = iConfig.getParameter< std::vector<unsigned int> >("constrainedLayersinBarrel");

    // Set the signal open
    isConfigured = true;
}

void RPCSeedLayerFinder::setInput(MuonRecHitContainer (&recHitsRPC)[RPCLayerNumber]) {

    for(unsigned int i = 0; i < RPCLayerNumber; i++)
        recHitsinLayers[i] = recHitsRPC[i].size();

    // Set the signal open
    isInputset = true;
}

void RPCSeedLayerFinder::unsetInput() {

    isInputset = false;
}

void RPCSeedLayerFinder::setOutput(RPCSeedrecHitFinder* Ref = nullptr, RPCCosmicSeedrecHitFinder* CosmicRef = nullptr) {

    RPCrecHitFinderRef = Ref;
    RPCCosmicrecHitFinderRef = CosmicRef;
    isOutputset = true;
}

void RPCSeedLayerFinder::fill() {

    // Check if already configured
    if(isConfigured == false || isInputset == false || isOutputset == false) {
        cout << "RPCSeedLayerFinder needs to be configured and set IO before running RPCSeedLayerFinder::fillLayers()" << endl;
        return;
    }

    // Clear the vector LayersinRPC
    LayersinRPC.clear();

    // Now fill the Layers
    if(isCosmic == true) {
        if(RPCCosmicrecHitFinderRef != nullptr)
            fillCosmicLayers();
        else
            cout << "RPCCosmicrecHitFinderRef not set" << endl;
    }
    else {
        if(RPCrecHitFinderRef != nullptr)
            fillLayers();
        else
            cout << "RPCrecHitFinderRef not set" << endl;
    }
}

void RPCSeedLayerFinder::fillLayers() {

    if(isSpecialLayers == false && isMixBarrelwithEndcap == false) {
        for(std::vector<unsigned int>::iterator NumberofLayersinBarrel = RangeofLayersinBarrel.begin(); NumberofLayersinBarrel != RangeofLayersinBarrel.end(); NumberofLayersinBarrel++) {
            // find N layers out of 6 Barrel Layers to fill to SeedinRPC
            unsigned int NumberofLayers = *NumberofLayersinBarrel;
            if(NumberofLayers < 1 || NumberofLayers > BarrelLayerNumber)
                continue;
            int type = 0;  // type=0 for barrel
            LayersinRPC.clear();
            SpecialLayers(-1, NumberofLayers, type);
            LayersinRPC.clear();
        }

        for(std::vector<unsigned int>::iterator NumberofLayersinEndcap = RangeofLayersinEndcap.begin(); NumberofLayersinEndcap != RangeofLayersinEndcap.end(); NumberofLayersinEndcap++) {
            unsigned int NumberofLayers = *NumberofLayersinEndcap;
            if(NumberofLayers < 1 || NumberofLayers > EachEndcapLayerNumber)
                continue;
            int type = 1; // type=1 for endcap
            // for -Z layers
            LayersinRPC.clear();
            SpecialLayers(BarrelLayerNumber-1, NumberofLayers, type);
            LayersinRPC.clear();
            //for +Z layers
            LayersinRPC.clear();
            SpecialLayers(BarrelLayerNumber+EachEndcapLayerNumber-1, NumberofLayers, type);
            LayersinRPC.clear();
        }
    }

    if(isSpecialLayers == true && isMixBarrelwithEndcap == false) {
        // Fill barrel layer for seed
        bool EnoughforBarrel = true;
        unsigned int i = 0;
        LayersinRPC.clear();
        for(std::vector<unsigned int>::iterator it = LayersinBarrel.begin(); it != LayersinBarrel.end(); it++, i++) {   
            if((*it) != 0 && i < BarrelLayerNumber) {
                if(recHitsinLayers[i] != 0)
                    LayersinRPC.push_back(i);
                else {
                    cout << "Not recHits in special Barrel layer " << i << endl;
                    EnoughforBarrel = false;
                }
            }
        }
        if(EnoughforBarrel && (!LayersinRPC.empty())) {
            // Initiate and call recHit Finder
            RPCrecHitFinderRef->setLayers(LayersinRPC);
            RPCrecHitFinderRef->fillrecHits();
        }
        LayersinRPC.clear();

        // Fill -Z and +Z endcap layer
        bool EnoughforEndcap = true;

        // Fill endcap- layer for seed
        i = BarrelLayerNumber;
        EnoughforEndcap = true;
        LayersinRPC.clear();
        for(std::vector<unsigned int>::iterator it = LayersinEndcap.begin(); it != LayersinEndcap.end(); it++, i++) {
            if((*it) != 0 && i < (BarrelLayerNumber+EachEndcapLayerNumber)) {
                if(recHitsinLayers[i] != 0)
                    LayersinRPC.push_back(i);
                else {
                    cout << "Not recHits in special Endcap " << (i - BarrelLayerNumber) << endl;
                    EnoughforEndcap = false;
                }
            }
        }
        if(EnoughforEndcap && (!LayersinRPC.empty())) {
            // Initiate and call recHit Finder
            RPCrecHitFinderRef->setLayers(LayersinRPC);
            RPCrecHitFinderRef->fillrecHits();
        }
        LayersinRPC.clear();

        //Fill endcap+ layer for seed
        i = BarrelLayerNumber;
        EnoughforEndcap = true;
        LayersinRPC.clear();
        for(std::vector<unsigned int>::iterator it = LayersinEndcap.begin(); it != LayersinEndcap.end(); it++, i++) {
            if((*it) != 0 && i >= (BarrelLayerNumber+EachEndcapLayerNumber) && i < (BarrelLayerNumber+EachEndcapLayerNumber*2)) {
                if(recHitsinLayers[i] != 0)
                    LayersinRPC.push_back(i);
                else {
                    cout << "Not recHits in special Endcap " << i << endl;
                    EnoughforEndcap = false;
                }
            }
        }
        if(EnoughforEndcap && (!LayersinRPC.empty())) {
            // Initiate and call recHit Finder
            RPCrecHitFinderRef->setLayers(LayersinRPC);
            RPCrecHitFinderRef->fillrecHits();
        }
        LayersinRPC.clear();
    }

    if(isMixBarrelwithEndcap == true) {
        cout <<" Mix is not ready for non-cosmic case" << endl;
        LayersinRPC.clear();
    }
}

void RPCSeedLayerFinder::fillCosmicLayers() {
    
    // For cosmic only handle the SpecialLayers case
    if(isSpecialLayers == true && isMixBarrelwithEndcap == false) {

        // Fill barrel layer for seed
        unsigned int i = 0;
        LayersinRPC.clear();
        for(std::vector<unsigned int>::iterator it = LayersinBarrel.begin(); it != LayersinBarrel.end(); it++, i++) {   
            if((*it) != 0 && i < BarrelLayerNumber)
                if(recHitsinLayers[i] != 0)
                    LayersinRPC.push_back(i);
        }
        if(!LayersinRPC.empty()) {
            // Initiate and call recHit Finder
            RPCCosmicrecHitFinderRef->setLayers(LayersinRPC);
            RPCCosmicrecHitFinderRef->fillrecHits();
        }
        LayersinRPC.clear();

        // Fill -Z and +Z endcap layer

        // Fill endcap- layer for seed
        i = BarrelLayerNumber;
        LayersinRPC.clear();
        for(std::vector<unsigned int>::iterator it = LayersinEndcap.begin(); it != LayersinEndcap.end(); it++, i++) {
            if((*it) != 0 && i < (BarrelLayerNumber+EachEndcapLayerNumber))
                if(recHitsinLayers[i] != 0)
                    LayersinRPC.push_back(i);
        }
        if(!LayersinRPC.empty()) {
            // Initiate and call recHit Finder
            RPCCosmicrecHitFinderRef->setLayers(LayersinRPC);
            RPCCosmicrecHitFinderRef->fillrecHits();
        }
        LayersinRPC.clear();

        //Fill endcap+ layer for seed
        i = BarrelLayerNumber;
        LayersinRPC.clear();
        for(std::vector<unsigned int>::iterator it = LayersinEndcap.begin(); it != LayersinEndcap.end(); it++, i++) {
            if((*it) != 0 && i >= (BarrelLayerNumber+EachEndcapLayerNumber) && i < (BarrelLayerNumber+EachEndcapLayerNumber*2))
                if(recHitsinLayers[i] != 0)
                    LayersinRPC.push_back(i);
        }
        if(!LayersinRPC.empty()) {
            // Initiate and call recHit Finder
            RPCCosmicrecHitFinderRef->setLayers(LayersinRPC);
            RPCCosmicrecHitFinderRef->fillrecHits();
        }
        LayersinRPC.clear();
    }

    if(isSpecialLayers == true && isMixBarrelwithEndcap == true) {

        // Fill all
        unsigned int i = 0;
        LayersinRPC.clear();
        for(std::vector<unsigned int>::iterator it = LayersinBarrel.begin(); it != LayersinBarrel.end(); it++, i++) {   
            if((*it) != 0 && i < BarrelLayerNumber)
                if(recHitsinLayers[i] != 0)
                    LayersinRPC.push_back(i);
        }
        i = BarrelLayerNumber;
        for(std::vector<unsigned int>::iterator it = LayersinEndcap.begin(); it != LayersinEndcap.end(); it++, i++) {
            if((*it) != 0 && i < (BarrelLayerNumber+EachEndcapLayerNumber*2))
                if(recHitsinLayers[i] != 0)
                    LayersinRPC.push_back(i);
        }

        if(!LayersinRPC.empty()) {
            // Initiate and call recHit Finder
            RPCCosmicrecHitFinderRef->setLayers(LayersinRPC);
            RPCCosmicrecHitFinderRef->fillrecHits();
        }
        LayersinRPC.clear();
    }

    if(isSpecialLayers == false) {
        cout << "Not ready for not SpecialLayers for Cosmic case" << endl;
        LayersinRPC.clear();
    }
}

void RPCSeedLayerFinder::SpecialLayers(int last, unsigned int NumberofLayers, int type) {

    // check type, 0=barrel, 1=endcap, 2=mix

    // barrel has 6 layers
    if(type == 0) {
        if(NumberofLayers > BarrelLayerNumber) {
            cout << "NumberofLayers larger than max layers in barrel" << endl;
            return;
        }
        for(unsigned int i = (last+1); i <= (BarrelLayerNumber-NumberofLayers+LayersinRPC.size()); i++) {
            if(recHitsinLayers[i] != 0) {
                LayersinRPC.push_back(i);
                last = i;
                if(LayersinRPC.size() < NumberofLayers)
                    SpecialLayers(last, NumberofLayers, type);
                else {
                    if(checkConstrain()) {
                        cout << "Find special barrel layers: ";
                        for(unsigned int k = 0; k < NumberofLayers; k++)
                            cout << LayersinRPC[k] <<" ";
                        cout << endl;
                        // Initiate and call recHit Finder
                        RPCrecHitFinderRef->setLayers(LayersinRPC);
                        RPCrecHitFinderRef->fillrecHits();
                    }
                    else
                        cout << "The layers don't contain all layers in constrain" << endl;
                }
                LayersinRPC.pop_back();
            }
        }
    }

    // endcap has 3 layers for each -Z and +Z
    if(type == 1) {
        if(NumberofLayers > EachEndcapLayerNumber) {
            cout << "NumberofLayers larger than max layers in endcap" << endl;
            return;
        }
        if(last < (BarrelLayerNumber+EachEndcapLayerNumber-1) || (last == (BarrelLayerNumber+EachEndcapLayerNumber-1) && !LayersinRPC.empty())) {
            // For -Z case
            for(unsigned int i =  (last+1); i <= (BarrelLayerNumber+EachEndcapLayerNumber-NumberofLayers+LayersinRPC.size()); i++) {
                if(recHitsinLayers[i] != 0) {
                    LayersinRPC.push_back(i);
                    last = i;
                    if(LayersinRPC.size() < NumberofLayers)
                        SpecialLayers(last, NumberofLayers, type);
                    else {
                        cout << "Find special -Z endcap layers: ";
                        for(unsigned int k = 0; k < NumberofLayers; k++)
                            cout << LayersinRPC[k] <<" ";
                        cout << endl;
                        // Initiate and call recHit Finder
                        RPCrecHitFinderRef->setLayers(LayersinRPC);
                        RPCrecHitFinderRef->fillrecHits();
                    }
                    LayersinRPC.pop_back();
                }
            }
        }
        else
        {
            // For +Z case
            for(unsigned int i = (last+1); i <= (BarrelLayerNumber+EachEndcapLayerNumber*2-NumberofLayers+LayersinRPC.size()); i++) {
                if(recHitsinLayers[i] != 0) {
                    LayersinRPC.push_back(i);
                    last = i;
                    if(LayersinRPC.size() < NumberofLayers)
                        SpecialLayers(last, NumberofLayers, type);
                    else {
                        cout << "Find special +Z endcap layers: ";
                        for(unsigned int k = 0; k < NumberofLayers; k++)
                            cout << LayersinRPC[k] <<" ";
                        cout << endl;
                        // Initiate and call recHit Finder
                        RPCrecHitFinderRef->setLayers(LayersinRPC);
                        RPCrecHitFinderRef->fillrecHits();
                    }
                    LayersinRPC.pop_back();
                }
            }
        }
    }
}

bool RPCSeedLayerFinder::checkConstrain() {

    bool pass = true;
    std::vector<unsigned int> fitConstrain = constrainedLayersinBarrel;
    for(unsigned int i = 0; i < LayersinRPC.size(); i++)
        fitConstrain[LayersinRPC[i]] = 0;
    for(unsigned int i = 0; i < BarrelLayerNumber; i++)
        if(fitConstrain[i] != 0)
            pass = false;
    return pass;
}
