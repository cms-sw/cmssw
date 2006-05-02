/** \file
 *
 *  $Date: 2006/04/28 11:53:42 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

#include <RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h>

using namespace std;

MuonDetLayerGeometry::MuonDetLayerGeometry() {}

MuonDetLayerGeometry::~MuonDetLayerGeometry(){}

void MuonDetLayerGeometry::addCSCLayers(pair<vector<DetLayer*>, vector<DetLayer*> > csclayers) {
    
    vector<DetLayer*>::const_iterator it;
    for(it = csclayers.first.begin(); it!=csclayers.first.end(); it++) {
        cscLayers_fw.push_back(*it);
        cscLayers_all.push_back(*it);
        allForward.push_back(*it);
        allEndcap.push_back(*it);
        allDetLayers.push_back(*it);
    }
    
    for(it = csclayers.second.begin(); it!=csclayers.second.end(); it++) {
        cscLayers_bk.push_back(*it);
        cscLayers_all.push_back(*it);
        allBackward.push_back(*it);
        allEndcap.push_back(*it);
        allDetLayers.push_back(*it);
    }    
}    

/*
void MuonDetLayerGeometry::addRPCLayers(pair<vector<DetLayer*>, vector<DetLayer*> > csclayers) {
    
    cscLayers_fw = csclayers.first;
    cscLayers_bg = csclayers.second;
}    

void MuonDetLayerGeometry::addDTLayers(pair<vector<DetLayer*>, vector<DetLayer*> > csclayers) {
    
    cscLayers_fw = csclayers.first;
    cscLayers_bg = csclayers.second;
}    
*/

const vector<DetLayer*>& 
MuonDetLayerGeometry::allDTLayers() const {    
    return dtLayers; 
}

const vector<DetLayer*>&
MuonDetLayerGeometry::allCSCLayers() const {
    
    return cscLayers_all;
}


const vector<DetLayer*>&
MuonDetLayerGeometry::forwardCSCLayers() const {

    return cscLayers_fw;
}


const vector<DetLayer*>& 
MuonDetLayerGeometry::backwardCSCLayers() const {

    return cscLayers_bk;
}


const vector<DetLayer*>& 
MuonDetLayerGeometry::allRPCLayers() const {

    return rpcLayers_all;    
}


const vector<DetLayer*>& 
MuonDetLayerGeometry::barrelRPCLayers() const {

    return rpcLayers_barrel; 
}


const vector<DetLayer*>& 
MuonDetLayerGeometry::endcapRPCLayers() const {

    return rpcLayers_endcap;    
}


const vector<DetLayer*>& 
MuonDetLayerGeometry::forwardRPCLayers() const {

     return rpcLayers_fw; 
}


const vector<DetLayer*>& 
MuonDetLayerGeometry::backwardRPCLayers() const {

    return rpcLayers_bk; 
}


const vector<DetLayer*> 
MuonDetLayerGeometry::allLayers() const {

    return allDetLayers;    
}    


const vector<DetLayer*> 
MuonDetLayerGeometry::allBarrelLayers() const {

    return allBarrel;    
}    

const vector<DetLayer*> 
MuonDetLayerGeometry::allEndcapLayers() const {

    return allEndcap;    
}    


const vector<DetLayer*> 
MuonDetLayerGeometry::allForwardLayers() const {

    return allForward;    
}    


const vector<DetLayer*> 
MuonDetLayerGeometry::allBackwardLayers() const {

    return allBackward;    
}    
