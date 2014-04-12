/**
 *  See header file for a description of this class.
 *
 */


#include "RecoMuon/MuonSeedGenerator/src/RPCCosmicSeedrecHitFinder.h"
#include <DataFormats/TrackingRecHit/interface/TrackingRecHit.h>
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

using namespace std;
using namespace edm;

MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator find(MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator firstIter, MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator lastIter, const MuonTransientTrackingRecHit::MuonRecHitPointer& recHitRef) {

    MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator index = lastIter;
    for(MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator iter = firstIter; iter != lastIter; iter++)
        if((*iter) == recHitRef)
            index = iter;
    return index;
}

RPCCosmicSeedrecHitFinder::RPCCosmicSeedrecHitFinder() {

    // Initiate the member
    isLayerset = false;
    isConfigured = false;
    isInputset = false;
    isOutputset = false;
    isEdgeset = false;
    BxRange = 0;
    MaxDeltaPhi = 0;
    ClusterSet.clear();
    innerBounds.clear();
    isLayersmixed = false;
    LayersinRPC.clear();
    therecHits.clear();
    isOuterLayerfilled = false;
}

RPCCosmicSeedrecHitFinder::~RPCCosmicSeedrecHitFinder() {

}

void RPCCosmicSeedrecHitFinder::configure(const edm::ParameterSet& iConfig) {

    // Set the configuration
    BxRange = iConfig.getParameter<unsigned int>("BxRange");
    MaxDeltaPhi = iConfig.getParameter<double>("MaxDeltaPhi");
    ClusterSet = iConfig.getParameter< std::vector<int> >("ClusterSet");

    // Set the signal open
    isConfigured = true;
}

void RPCCosmicSeedrecHitFinder::setInput(MuonRecHitContainer (&recHits)[RPCLayerNumber]) {

    for(unsigned int i = 0; i < RPCLayerNumber; i++) {
        AllrecHits[i].clear();
        AllrecHits[i] = recHits[i];
    }

    // Set the signal open
    isInputset = true;
}

void RPCCosmicSeedrecHitFinder::setEdge(const edm::EventSetup& iSetup) {
  
    // Get RPCGeometry
    edm::ESHandle<RPCGeometry> rpcGeometry;
    iSetup.get<MuonGeometryRecord>().get(rpcGeometry);

    // Find all chamber in RB1in and collect their surface
    const std::vector<DetId> AllRPCId = rpcGeometry->detIds();
    for(std::vector<DetId>::const_iterator it = AllRPCId.begin(); it != AllRPCId.end(); it++) {
        RPCDetId RPCId(it->rawId());
        int Region = RPCId.region();
        int Station = RPCId.station();
        int Layer = RPCId.layer();
        if(Region == 0 && Station == 1 && Layer == 1) {
            const BoundPlane RPCChamberSurface = rpcGeometry->chamber(RPCId)->surface();
            innerBounds.push_back(RPCChamberSurface);
        }
    }

    // Set the signal open
    isEdgeset = true;
}

void RPCCosmicSeedrecHitFinder::unsetEdge() {

    // Clear all surfaces of chambers in RB1in
    innerBounds.clear();
    isEdgeset = false;
}

void RPCCosmicSeedrecHitFinder::unsetInput() {

    for(unsigned int i = 0; i < RPCLayerNumber; i++)
        AllrecHits[i].clear();
    isInputset = false;
}

void RPCCosmicSeedrecHitFinder::setOutput(RPCSeedFinder *Seed) {

    theSeed = Seed;
    // Set the signal open
    isOutputset = true;
}

void RPCCosmicSeedrecHitFinder::setLayers(const std::vector<unsigned int>& Layers) {

    LayersinRPC = Layers;
    // Set the signal open
    isLayerset = true;
}

void RPCCosmicSeedrecHitFinder::fillrecHits() {

    if(isLayerset == false || isConfigured == false || isOutputset == false || isInputset == false || isEdgeset == false) {
        cout << "Not set the IO or not configured yet" << endl;
        return;
    } 

    therecHits.clear();

    if(LayersinRPC.size() == 0) {
        cout << "Not set with any layers" << endl;
        LayersinRPC.clear();
        therecHits.clear();
        isLayerset = false;
    }

    // check the layers, 1=all barrel, 2=all endcap, 3=mix
    unsigned int Component = LayerComponent();
    if(Component == 3)
        isLayersmixed = true;
    else
        isLayersmixed = false;

    GlobalVector initVector(0, 0, 0);
    const MuonRecHitPointer recHitRef;
    isOuterLayerfilled = false;
    complete(initVector, recHitRef);

    // Unset the signal
    LayersinRPC.clear();
    therecHits.clear();
    isLayerset = false;
}

int RPCCosmicSeedrecHitFinder::LayerComponent() {

    bool isBarrel = false;
    bool isEndcap = false;
    for(std::vector<unsigned int>::const_iterator it = LayersinRPC.begin(); it != LayersinRPC.end(); it++) {
        if((*it) < BarrelLayerNumber)
            isBarrel = true;
        if((*it) >= BarrelLayerNumber && (*it) < (BarrelLayerNumber+EachEndcapLayerNumber*2))
            isEndcap = true;
    }
    if(isBarrel == true && isEndcap == true)
        return 3;
    if(isEndcap == true)
        return 2;
    if(isBarrel == true)
        return 1;
    return 0;
}

bool RPCCosmicSeedrecHitFinder::complete(const GlobalVector& lastSegment, const MuonRecHitPointer& lastrecHitRef) {

    bool isrecHitsfound = false;

    for(unsigned int i = 0; i < RPCLayerNumber; i++)
        for(MuonRecHitContainer::const_iterator it = AllrecHits[i].begin(); it != AllrecHits[i].end(); it++) {

            cout << "Finding recHits from " << i << " th layer" << endl;
            // information for recHits
            GlobalPoint currentPosition = (*it)->globalPosition();
            int currentBX;

            // Check validation
            if(!(*it)->isValid())
                continue;

            // Check BX range, be sure there is only RPCRecHit in the MuonRecHitContainer when use the dynamic_cast
            TrackingRecHit* thisTrackingRecHit = (*it)->hit()->clone();
            // Should also delete the RPCRecHit object cast by dynamic_cast<> ?
            RPCRecHit* thisRPCRecHit = dynamic_cast<RPCRecHit*>(thisTrackingRecHit);
            currentBX = thisRPCRecHit->BunchX();
            int ClusterSize = thisRPCRecHit->clusterSize();
            delete thisTrackingRecHit;
            // Check BX
            if((unsigned int)abs(currentBX) > BxRange)
                continue;

            // Check cluster size
            bool Clustercheck = false;
            if(ClusterSet.size() == 0)
                Clustercheck = true;
            for(std::vector<int>::const_iterator CluIter = ClusterSet.begin(); CluIter != ClusterSet.end(); CluIter++)
                if(ClusterSize == (*CluIter))
                    Clustercheck = true;
            if(Clustercheck != true)
                continue;
 
            cout << "Candidate recHit's position: " << currentPosition.x() << ", " << currentPosition.y() << ", " << currentPosition.z() << ". BX : " << currentBX << endl;
            // Fill 1st recHit from outer layers and rest recHits from all layers
            if(isOuterLayerfilled == false) {
                // Pick out the recHit from outer layers and fill it
                if(!isouterLayer(*it))
                    continue;

                // If pass all, add to the seed
                GlobalVector currentSegment = GlobalVector(0, 0, 0); 
                cout << "1st recHit's global position: " << currentPosition.x() << ", " << currentPosition.y() << ", " << currentPosition.z() << ". BX: " << currentBX << endl;
                isrecHitsfound = true;
                therecHits.push_back(*it);
                isOuterLayerfilled = true;
                complete(currentSegment, *it);
                // Remember to pop the recHit before add another one from the same layer!
                therecHits.pop_back();
                isOuterLayerfilled = false;
            }
            else {
                GlobalPoint lastPosition = lastrecHitRef->globalPosition();
                TrackingRecHit* lastTrackingRecHit = lastrecHitRef->hit()->clone();
                // Should also delete the RPCRecHit object cast by dynamic_cast<> ?
                RPCRecHit* lastRPCRecHit = dynamic_cast<RPCRecHit*>(lastTrackingRecHit);
                int lastBX = lastRPCRecHit->BunchX();
                delete lastTrackingRecHit;

                // Check the Y coordinate, shoule be lower than current one
                if(currentPosition.y() >= lastPosition.y())
                    continue;

                // Check the BX, should be larger than current one
                if(currentBX < lastBX)
                    continue;

                // If be the 2nd recHit, just fill it
                bool isinsideRegion = isinsideAngleRange(lastSegment, lastPosition, currentPosition);
                cout << "Check isinsideRegion: " << isinsideRegion << endl;
                if(!isinsideRegion)
                    continue;

                // If cross the edge the recHit should belong to another seed
                bool iscrossanyEdge = iscorssEdge(lastrecHitRef, *it);
                cout << "Check iscrossanyEdge: " << iscrossanyEdge << endl;
                if(iscrossanyEdge)
                    continue;

                // If pass all, add to the seed
                unsigned int NumberinSeed = therecHits.size();
                GlobalVector currentSegment = (GlobalVector)(currentPosition - lastPosition);
                cout << (NumberinSeed + 1) << "th recHit's global position: " << currentPosition.x() << ", " << currentPosition.y() << ", " << currentPosition.z() << ". BX: " << currentBX << endl;
                isrecHitsfound = true;
                therecHits.push_back(*it);

                // if could not find next recHit in the search path, and have enough recHits already, that is the candidate
                bool findNext = complete(currentSegment, *it);
                if(findNext == false && therecHits.size() > 3) {
                    for(ConstMuonRecHitContainer::const_iterator iter = therecHits.begin(); iter != therecHits.end(); iter++)
                        cout << "Find recHit in seed candidate : " << (*iter)->globalPosition().x() << ", " << (*iter)->globalPosition().y() << ", " << (*iter)->globalPosition().z() << endl;
                    checkandfill();
                }

                // Remember to pop the recHit before add another one from the same layer!
                therecHits.pop_back();
            }
        }
    return isrecHitsfound;
}

bool RPCCosmicSeedrecHitFinder::isouterLayer(const MuonRecHitPointer& recHitRef) {

    bool isinsideLayers = false;
    for(std::vector<unsigned int>::const_iterator it = LayersinRPC.begin(); it != LayersinRPC.end(); it++) {
        MuonRecHitContainer::const_iterator index = find(AllrecHits[*it].begin(), AllrecHits[*it].end(), recHitRef);
        if(index != AllrecHits[*it].end())
            isinsideLayers = true;
    }
    return isinsideLayers;
}

bool RPCCosmicSeedrecHitFinder::isinsideAngleRange(const GlobalVector& lastSegment, const GlobalPoint& lastPosition, const GlobalPoint& currentPosition) {

    bool isinsideAngle = true;
    GlobalVector SegVec = currentPosition - lastPosition;
    if(lastSegment.mag() != 0)
        if(fabs((lastSegment.phi()-SegVec.phi()).value()) > MaxDeltaPhi)
            isinsideAngle = false;
        
    return isinsideAngle;
}

bool RPCCosmicSeedrecHitFinder::iscorssEdge(const MuonRecHitPointer& lastrecHitRef, const MuonRecHitPointer& currentrecHitRef) {

    bool iscorss = false;

    
    // Check if 2 recHits corss the inner bounds
    GlobalPoint lastPosition = lastrecHitRef->globalPosition();
    GlobalPoint currentPosition = currentrecHitRef->globalPosition();
    GlobalPoint testPosition((lastPosition.x()+currentPosition.x())/2, (lastPosition.y()+currentPosition.y())/2, (lastPosition.z()+currentPosition.z())/2);
    /*
    for(std::vector<BoundPlane>::const_iterator it = innerBounds.begin(); it != innerBounds.end(); it++) {
        //SurfaceOrientation::Side TestSide0 = it->side(currentPosition, 0);
        //SurfaceOrientation::Side TestSide1 = it->side(lastPosition, 0);
        SurfaceOrientation::Side TestSide = it->side(testPosition, 0);
        //cout << "Side of currentPosition: " << TestSide0 << ", Side of lastPosition: " << TestSide1 << ", Side of middlePosition: " << TestSide << endl;
        //if(TestSide != SurfaceOrientation::positiveSide)
            //iscorss = true;
    }
    */

    // Check when mixLayer is not set
    if(isLayersmixed == false) {
        DetId lastId = lastrecHitRef->geographicalId();
        RPCDetId lastRPCId(lastId.rawId());
        int lastRegion = lastRPCId.region();
        DetId currentId = currentrecHitRef->geographicalId();
        RPCDetId currentRPCId(currentId.rawId());
        int currentRegion = currentRPCId.region();
        // Check if 2 recHits from different regions
        if(lastRegion != currentRegion)
            iscorss = true;
    }

    return iscorss;
}

void RPCCosmicSeedrecHitFinder::checkandfill() {

    if(therecHits.size() >= 3) {
        theSeed->setrecHits(therecHits); 
        theSeed->seed();
    }
    else
        cout << "Layer less than 3, could not fill a RPCSeedFinder" << endl;
}
