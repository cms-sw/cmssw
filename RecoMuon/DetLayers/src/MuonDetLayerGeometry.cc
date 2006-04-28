/** \file
 *
 *  $Date: 2006/04/12 16:49:57 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include <RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h>

using namespace std;

MuonDetLayerGeometry::MuonDetLayerGeometry(pair<vector<DetLayer*>, vector<DetLayer*> > csc):cscLayers_fw(csc.first), 
    cscLayers_bg(csc.second) {}


MuonDetLayerGeometry::~MuonDetLayerGeometry(){}


const vector<DetLayer*>& 
MuonDetLayerGeometry::allDTLayers() const {    
    return cscLayers_fw; // FIXME !!!
}

const vector<DetLayer*>& 
MuonDetLayerGeometry::allCSCLayers() const {
    
    vector<DetLayer*> temp;    
    vector<DetLayer*>::const_iterator it;
    for(it = cscLayers_fw.begin(); it!=cscLayers_fw.end(); it++)
        temp.push_back(*it);

    for(it = cscLayers_bg.begin(); it!=cscLayers_bg.end(); it++)
        temp.push_back(*it);
    
    return temp;    
}


const vector<DetLayer*>&
MuonDetLayerGeometry::forwardCSCLayers() const {
    return cscLayers_fw;
}


const vector<DetLayer*>& 
MuonDetLayerGeometry::backwardCSCLayers() const {
    return cscLayers_bg;
}


const vector<DetLayer*>& 
MuonDetLayerGeometry::allRPCLayers() const {
    return cscLayers_fw; // FIXME !!!
}


const vector<DetLayer*>& 
MuonDetLayerGeometry::barrelRPCLayers() const {
    return cscLayers_fw; // FIXME !!!
}


const vector<DetLayer*>& 
MuonDetLayerGeometry::endcapRPCLayers() const {
     return cscLayers_fw; // FIXME !!!
}


const vector<DetLayer*>& 
MuonDetLayerGeometry::forwardRPCLayers() const {
     return cscLayers_fw; // FIXME !!!
}


const vector<DetLayer*>& 
MuonDetLayerGeometry::backwardRPCLayers() const {
    return cscLayers_fw; // FIXME !!!
}


const vector<DetLayer*> 
MuonDetLayerGeometry::allLayers() const {
    return cscLayers_fw; // FIXME !!!
}    


const vector<DetLayer*> 
MuonDetLayerGeometry::allBarrelLayers() const {
    return cscLayers_fw; // FIXME !!!
}    

const vector<DetLayer*> 
MuonDetLayerGeometry::allEndcapLayers() const {
    return cscLayers_fw; // FIXME !!!
}    


const vector<DetLayer*> 
MuonDetLayerGeometry::allForwardLayers() const {
    return cscLayers_fw; // FIXME !!!
}    


const vector<DetLayer*> 
MuonDetLayerGeometry::allBackwardLayers() const {
    return cscLayers_fw; // FIXME !!!
}    
