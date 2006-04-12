/** \file
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"

using namespace std;

MuonDetLayerGeometry::MuonDetLayerGeometry(){}


MuonDetLayerGeometry::~MuonDetLayerGeometry(){}


const vector<DetLayer*>& 
MuonDetLayerGeometry::allDTLayers() const{return vector<DetLayer*>();}

const vector<DetLayer*>& 
MuonDetLayerGeometry::allCSCLayers() const{return vector<DetLayer*>();}


const vector<DetLayer*>&
MuonDetLayerGeometry::forwardCSCLayers() const{return vector<DetLayer*>();}


const vector<DetLayer*>& 
MuonDetLayerGeometry::backwardCSCLayers() const{return vector<DetLayer*>();}


const vector<DetLayer*>& 
MuonDetLayerGeometry::allRPCLayers() const{return vector<DetLayer*>();}


const vector<DetLayer*>& 
MuonDetLayerGeometry::barrelRPCLayers() const{return vector<DetLayer*>();}


const vector<DetLayer*>& 
MuonDetLayerGeometry::endcapRPCLayers() const{return vector<DetLayer*>();}


const vector<DetLayer*>& 
MuonDetLayerGeometry::forwardRPCLayers() const{return vector<DetLayer*>();}


const vector<DetLayer*>& 
MuonDetLayerGeometry::backwardRPCLayers() const{return vector<DetLayer*>();}


const vector<DetLayer*> 
MuonDetLayerGeometry::allLayers() const{return vector<DetLayer*>();}


const vector<DetLayer*> 
MuonDetLayerGeometry::allBarrelLayers() const{return vector<DetLayer*>();}


const vector<DetLayer*> 
MuonDetLayerGeometry::allEndcapLayers() const{return vector<DetLayer*>();}


const vector<DetLayer*> 
MuonDetLayerGeometry::allForwardLayers() const{return vector<DetLayer*>();}


const vector<DetLayer*> 
MuonDetLayerGeometry::allBackwardLayers() const{return vector<DetLayer*>();}  
