/** \file
 *
 *  \author N. Amapane - CERN
 *
 *  \modified by R. Radogna & C. Calabria & A. Sharma
 *  \modified by R. Radogna & C. Calabria
 *  \modified by D. Nash
 */

#include <RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h>

#include <FWCore/Utilities/interface/Exception.h>
#include <TrackingTools/DetLayers/interface/DetLayer.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/MuonDetId/interface/ME0DetId.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <DataFormats/GeometrySurface/interface/GeometricSorting.h>

#include <algorithm>

using namespace std;
using namespace geomsort;

MuonDetLayerGeometry::MuonDetLayerGeometry() {}

MuonDetLayerGeometry::~MuonDetLayerGeometry(){
  for(vector<const DetLayer*>::const_iterator it = allDetLayers.begin(); it != allDetLayers.end(); ++it)
  {
    delete *it;
  }
}

void MuonDetLayerGeometry::addCSCLayers(const pair<vector<DetLayer*>, vector<DetLayer*> >& csclayers) {
    
  for(auto const it : csclayers.first) {
    cscLayers_fw.push_back(it);
    allForward.push_back(it);
    
    detLayersMap[ makeDetLayerId(it) ] = it;
  }

  for(auto const it: csclayers.second) {
    cscLayers_bk.push_back(it);
    allBackward.push_back(it);
    
    detLayersMap[ makeDetLayerId(it) ] = it;
  }    
}    

void MuonDetLayerGeometry::addGEMLayers(const pair<vector<DetLayer*>, vector<DetLayer*> >& gemlayers) {
  LogDebug("Muon|RecoMuon|MuonDetLayerGeometry") << "Adding GEMlayers "<<std::endl;
  for(auto const it : gemlayers.first) {
    gemLayers_fw.push_back(it);
    allForward.push_back(it);
    detLayersMap[ makeDetLayerId(it) ] = it;
    LogDebug("Muon|RecoMuon|MuonDetLayerGeometry") << "Adding GEMforward "<<std::endl;
  }
  for(auto const it: gemlayers.second) {
    gemLayers_bk.push_back(it);
    allBackward.push_back(it);
    detLayersMap[ makeDetLayerId(it) ] = it;
    LogDebug("Muon|RecoMuon|MuonDetLayerGeometry") << "Adding GEMbackward "<<std::endl;
    }
}   


void MuonDetLayerGeometry::addME0Layers(const pair<vector<DetLayer*>, vector<DetLayer*> >& me0layers) {

  LogDebug("Muon|RecoMuon|MuonDetLayerGeometry") << "Adding ME0layers "<<std::endl;

  for(auto const it : me0layers.first) {
    me0Layers_fw.push_back(it);
    allForward.push_back(it);

    detLayersMap[ makeDetLayerId(it) ] = it;
    LogDebug("Muon|RecoMuon|MuonDetLayerGeometry") << "Adding ME0forward "<<std::endl;
  }
  for(auto const it : me0layers.second) {
    me0Layers_bk.push_back(it);
    allBackward.push_back(it);

    detLayersMap[ makeDetLayerId(it) ] = it;
    LogDebug("Muon|RecoMuon|MuonDetLayerGeometry") << "Adding ME0backward "<<std::endl;
  }
}

void MuonDetLayerGeometry::addRPCLayers(const vector<DetLayer*>& barrelLayers, const pair<vector<DetLayer*>, vector<DetLayer*> >& endcapLayers) {
  
  for(auto const it: barrelLayers) {
    rpcLayers_barrel.push_back(it);
    allBarrel.push_back(it);

    detLayersMap[ makeDetLayerId(it) ] = it;
  }
  for(auto const it: endcapLayers.first) {
    rpcLayers_fw.push_back(it);
    allForward.push_back(it);

    detLayersMap[ makeDetLayerId(it) ] = it;
  }

  for(auto const it: endcapLayers.second) {
    rpcLayers_bk.push_back(it);
    allBackward.push_back(it);

    detLayersMap[ makeDetLayerId(it) ] = it;
  }
  
}    

void MuonDetLayerGeometry::addDTLayers(const vector<DetLayer*>& dtlayers) {

    for(auto const it : dtlayers) {
        dtLayers.push_back(it);
        allBarrel.push_back(it);

	detLayersMap[ makeDetLayerId(it) ] = it;
    }
}    

DetId MuonDetLayerGeometry::makeDetLayerId(const DetLayer* detLayer) const{

  if(detLayer->subDetector() ==  GeomDetEnumerators::CSC){
    CSCDetId id( detLayer->basicComponents().front()->geographicalId().rawId() ) ;

    if(id.station() == 1 )
      {
	if(id.ring() == 1 || id.ring() == 4)
	  return CSCDetId(id.endcap(),1,1,0,0);
	else if(id.ring() == 2 || id.ring() == 3)
	  return CSCDetId(id.endcap(),1,2,0,0);
	else
	  throw cms::Exception("InvalidCSCRing")<<" Invalid CSC Ring: "<<id.ring()<<endl;
      }
    else
      return CSCDetId(id.endcap(),id.station(),0,0,0);
    
  }
  else if(detLayer->subDetector() == GeomDetEnumerators::DT){
    DTChamberId id( detLayer->basicComponents().front()->geographicalId().rawId() ) ;
    return  DTChamberId(0,id.station(),0);
  }
  else if(detLayer->subDetector()== GeomDetEnumerators::RPCBarrel ||
	  detLayer->subDetector()== GeomDetEnumerators::RPCEndcap){
    RPCDetId id( detLayer->basicComponents().front()->geographicalId().rawId());
    return RPCDetId(id.region(),0,id.station(),0,id.layer(),0,0);
  }
  else if( detLayer->subDetector()== GeomDetEnumerators::GEM){
    GEMDetId id( detLayer->basicComponents().front()->geographicalId().rawId());
    return GEMDetId(id.region(),1,id.station(),id.layer(),0,0);
  }
  else if( detLayer->subDetector()== GeomDetEnumerators::ME0){
    ME0DetId id( detLayer->basicComponents().front()->geographicalId().rawId());
    return ME0DetId(id.region(),id.layer(),0,0);
  }
  else throw cms::Exception("InvalidModuleIdentification"); // << detLayer->module();
}


const vector<const DetLayer*>& 
MuonDetLayerGeometry::allDTLayers() const {    
    return dtLayers; 
}

const vector<const DetLayer*>&
MuonDetLayerGeometry::allCSCLayers() const {
    return cscLayers_all;
}


const vector<const DetLayer*>&
MuonDetLayerGeometry::forwardCSCLayers() const {
    return cscLayers_fw;
}


const vector<const DetLayer*>& 
MuonDetLayerGeometry::backwardCSCLayers() const {
    return cscLayers_bk;
}

////////////////////////////GEMs
const vector<const DetLayer*>&
MuonDetLayerGeometry::allGEMLayers() const {
    return gemLayers_all;
}


const vector<const DetLayer*>&
MuonDetLayerGeometry::forwardGEMLayers() const {
    return gemLayers_fw;
}


const vector<const DetLayer*>&
MuonDetLayerGeometry::backwardGEMLayers() const {
    return gemLayers_bk;
}

//////////////////////////////////////////


//////////////////// ME0s

const vector<const DetLayer*>&
MuonDetLayerGeometry::allME0Layers() const {
    return me0Layers_all;
}


const vector<const DetLayer*>&
MuonDetLayerGeometry::forwardME0Layers() const {
    return me0Layers_fw;
}


const vector<const DetLayer*>&
MuonDetLayerGeometry::backwardME0Layers() const {
    return me0Layers_bk;
}

////////////////////

const vector<const DetLayer*>& 
MuonDetLayerGeometry::allRPCLayers() const {
    return rpcLayers_all;    
}


const vector<const DetLayer*>& 
MuonDetLayerGeometry::barrelRPCLayers() const {
    return rpcLayers_barrel; 
}


const vector<const DetLayer*>& 
MuonDetLayerGeometry::endcapRPCLayers() const {
    return rpcLayers_endcap;    
}


const vector<const DetLayer*>& 
MuonDetLayerGeometry::forwardRPCLayers() const {
     return rpcLayers_fw; 
}


const vector<const DetLayer*>& 
MuonDetLayerGeometry::backwardRPCLayers() const {
    return rpcLayers_bk; 
}


const vector<const DetLayer*>&
MuonDetLayerGeometry::allLayers() const {
    return allDetLayers;    
}    


const vector<const DetLayer*>&
MuonDetLayerGeometry::allBarrelLayers() const {
    return allBarrel;    
}    

const vector<const DetLayer*>&
MuonDetLayerGeometry::allEndcapLayers() const {
    return allEndcap;    
}    


const vector<const DetLayer*>&
MuonDetLayerGeometry::allForwardLayers() const {
    return allForward;    
}    


const vector<const DetLayer*>&
MuonDetLayerGeometry::allBackwardLayers() const {
    return allBackward;    
}    

//////////////////////////////GEMs
//
const vector<const DetLayer*>&
MuonDetLayerGeometry::allEndcapCscGemLayers() const {
    return allEndcapCscGem;
}


const vector<const DetLayer*>&
MuonDetLayerGeometry::allCscGemForwardLayers() const {
    return allCscGemForward;
}


const vector<const DetLayer*>&
MuonDetLayerGeometry::allCscGemBackwardLayers() const {
    return allCscGemBackward;
}

//////////////////// ME0s

const vector<const DetLayer*>&
MuonDetLayerGeometry::allEndcapCscME0Layers() const {
    return allEndcapCscME0;    
}    


const vector<const DetLayer*>&
MuonDetLayerGeometry::allCscME0ForwardLayers() const {
    return allCscME0Forward;    
}    


const vector<const DetLayer*>&
MuonDetLayerGeometry::allCscME0BackwardLayers() const {
    return allCscME0Backward;    
}    


////////////////////////////////////////////////////

const DetLayer* MuonDetLayerGeometry::idToLayer(const DetId &detId) const{

  DetId id;
  
  if(detId.subdetId() == MuonSubdetId::CSC){
    CSCDetId cscId( detId.rawId() );

    if(cscId.station() == 1)
      {
	if(cscId.ring() == 1 || cscId.ring() == 4)
	  id = CSCDetId(cscId.endcap(),1,1,0,0);
	else if(cscId.ring() == 2 || cscId.ring() == 3)
	  id = CSCDetId(cscId.endcap(),1,2,0,0);
	else
	  throw cms::Exception("InvalidCSCRing")<<" Invalid CSC Ring: "<<cscId.ring()<<endl;
      }
    else id = CSCDetId(cscId.endcap(),cscId.station(),0,0,0);
  }
  
  else if (detId.subdetId() == MuonSubdetId::DT){
    DTChamberId dtId( detId.rawId() );
    id = DTChamberId(0,dtId.station(),0);
  }
  else if (detId.subdetId() == MuonSubdetId::RPC){
    RPCDetId rpcId(detId.rawId() );
    id = RPCDetId(rpcId.region(),0,rpcId.station(),0,rpcId.layer(),0,0);
  }
  else if (detId.subdetId() == MuonSubdetId::GEM){
    GEMDetId gemId(detId.rawId() );
    id = GEMDetId(gemId.region(),1,gemId.station(),gemId.layer(),0,0);
  }
  else if (detId.subdetId() == MuonSubdetId::ME0){
    ME0DetId me0Id(detId.rawId() );
    id = ME0DetId(me0Id.region(),me0Id.layer(),0,0);
    LogDebug("Muon|RecoMuon|MuonDetLayerGeometry") << " Found an ME0DetId:  " << me0Id.rawId()
						   <<",id: "<< id.rawId()<<std::endl;

  }
  else throw cms::Exception("InvalidSubdetId")<< detId.subdetId();

  std::map<DetId,const DetLayer*>::const_iterator layer = detLayersMap.find(id);
  if (layer == detLayersMap.end()) return nullptr;
  return layer->second; 
}


// Quick way to sort barrel det layers by increasing R,
// do not abuse!
#include <TrackingTools/DetLayers/interface/BarrelDetLayer.h>
struct ExtractBarrelDetLayerR {
  typedef Surface::Scalar result_type;
  result_type operator()(const DetLayer* p) const {
    const BarrelDetLayer * bdl = dynamic_cast<const BarrelDetLayer*>(p);
    if (bdl) return bdl->specificSurface().radius();
    else return -1.;
  }
};

void MuonDetLayerGeometry::sortLayers() {

  // The following are filled inside-out, no need to re-sort
  // precomputed_value_sort(dtLayers.begin(), dtLayers.end(),ExtractR<DetLayer,float>());
  // precomputed_value_sort(cscLayers_fw.begin(), cscLayers_fw.end(),ExtractAbsZ<DetLayer,float>());
  // precomputed_value_sort(cscLayers_bk.begin(), cscLayers_bk.end(),ExtractAbsZ<DetLayer,float>());
  // precomputed_value_sort(rpcLayers_fw.begin(), rpcLayers_fw.end(),ExtractAbsZ<DetLayer,float>());
  // precomputed_value_sort(rpcLayers_bk.begin(), rpcLayers_bk.end(),ExtractAbsZ<DetLayer,float>());
  // precomputed_value_sort(rpcLayers_barrel.begin(), rpcLayers_barrel.end(), ExtractR<DetLayer,float>());

  // Sort these inside-out
  precomputed_value_sort(allBarrel.begin(), allBarrel.end(), ExtractBarrelDetLayerR());
  precomputed_value_sort(allBackward.begin(), allBackward.end(), ExtractAbsZ<DetLayer,float>());
  precomputed_value_sort(allForward.begin(), allForward.end(), ExtractAbsZ<DetLayer,float>());  

  // Build more complicated vectors with correct sorting

  //cscLayers_all: from -Z to +Z
  cscLayers_all.reserve(cscLayers_bk.size()+cscLayers_fw.size());
  std::copy(cscLayers_bk.begin(),cscLayers_bk.end(),back_inserter(cscLayers_all));
  std::reverse(cscLayers_all.begin(),cscLayers_all.end());
  std::copy(cscLayers_fw.begin(),cscLayers_fw.end(),back_inserter(cscLayers_all));

  //gemLayers_all: from -Z to +Z
  gemLayers_all.reserve(gemLayers_bk.size()+gemLayers_fw.size());
  std::copy(gemLayers_bk.begin(),gemLayers_bk.end(),back_inserter(gemLayers_all));
  std::reverse(gemLayers_all.begin(),gemLayers_all.end());
  std::copy(gemLayers_fw.begin(),gemLayers_fw.end(),back_inserter(gemLayers_all));

  //me0Layers_all: from -Z to +Z
  me0Layers_all.reserve(me0Layers_bk.size()+me0Layers_fw.size());
  std::copy(me0Layers_bk.begin(),me0Layers_bk.end(),back_inserter(me0Layers_all));
  std::reverse(me0Layers_all.begin(),me0Layers_all.end());
  std::copy(me0Layers_fw.begin(),me0Layers_fw.end(),back_inserter(me0Layers_all));

  //rpcLayers_endcap: from -Z to +Z
  rpcLayers_endcap.reserve(rpcLayers_bk.size()+rpcLayers_fw.size());
  std::copy(rpcLayers_bk.begin(),rpcLayers_bk.end(),back_inserter(rpcLayers_endcap));
  std::reverse(rpcLayers_endcap.begin(),rpcLayers_endcap.end());
  std::copy(rpcLayers_fw.begin(),rpcLayers_fw.end(),back_inserter(rpcLayers_endcap));

  //rpcLayers_all: order is bw, barrel, fw
  rpcLayers_all.reserve(rpcLayers_bk.size()+rpcLayers_barrel.size()+rpcLayers_fw.size());
  std::copy(rpcLayers_bk.begin(),rpcLayers_bk.end(),back_inserter(rpcLayers_all));
  std::reverse(rpcLayers_all.begin(),rpcLayers_all.end());
  std::copy(rpcLayers_barrel.begin(),rpcLayers_barrel.end(),back_inserter(rpcLayers_all));
  std::copy(rpcLayers_fw.begin(),rpcLayers_fw.end(),back_inserter(rpcLayers_all));

  // allEndcap: order is  all bw, all fw
  allEndcap.reserve(allBackward.size()+allForward.size());
  std::copy(allBackward.begin(),allBackward.end(),back_inserter(allEndcap));
  std::reverse(allEndcap.begin(),allEndcap.end());
  std::copy(allForward.begin(),allForward.end(),back_inserter(allEndcap));

  // allEndcapCSCGEM: order is  all bw, all fw
  allEndcapCscGem.reserve(cscLayers_bk.size()+cscLayers_fw.size()+gemLayers_bk.size()+gemLayers_fw.size());
  std::copy(cscLayers_bk.begin(),cscLayers_bk.end(),back_inserter(allEndcapCscGem));
  std::copy(gemLayers_bk.begin(),gemLayers_bk.end(),back_inserter(allEndcapCscGem));
  std::reverse(allEndcapCscGem.begin(),allEndcapCscGem.end());
  std::copy(cscLayers_fw.begin(),cscLayers_fw.end(),back_inserter(allEndcapCscGem));
  std::copy(gemLayers_fw.begin(),gemLayers_fw.end(),back_inserter(allEndcapCscGem));

  // allCscGemForward
  allCscGemForward.reserve(cscLayers_fw.size()+gemLayers_fw.size());
  std::copy(cscLayers_fw.begin(),cscLayers_fw.end(),back_inserter(allCscGemForward));
  std::copy(gemLayers_fw.begin(),gemLayers_fw.end(),back_inserter(allCscGemForward));

  // allCscGemBackward
  allCscGemBackward.reserve(cscLayers_bk.size()+gemLayers_bk.size());
  std::copy(cscLayers_bk.begin(),cscLayers_bk.end(),back_inserter(allCscGemBackward));
  std::copy(gemLayers_bk.begin(),gemLayers_bk.end(),back_inserter(allCscGemBackward));

  // allCscME0Forward
  allCscME0Forward.reserve(cscLayers_fw.size()+me0Layers_fw.size());
  std::copy(cscLayers_fw.begin(),cscLayers_fw.end(),back_inserter(allCscME0Forward));
  std::copy(me0Layers_fw.begin(),me0Layers_fw.end(),back_inserter(allCscME0Forward));

  // allCscME0Backward
  allCscME0Backward.reserve(cscLayers_bk.size()+me0Layers_bk.size());
  std::copy(cscLayers_bk.begin(),cscLayers_bk.end(),back_inserter(allCscME0Backward));
  std::copy(me0Layers_bk.begin(),me0Layers_bk.end(),back_inserter(allCscME0Backward));

  // allEndcapCSCME0: order is  all bw, all fw
  allEndcapCscME0.reserve(cscLayers_bk.size()+cscLayers_fw.size()+me0Layers_bk.size()+me0Layers_fw.size());
  std::copy(cscLayers_bk.begin(),cscLayers_bk.end(),back_inserter(allEndcapCscME0));
  std::copy(me0Layers_bk.begin(),me0Layers_bk.end(),back_inserter(allEndcapCscME0));
  std::reverse(allEndcapCscME0.begin(),allEndcapCscME0.end());
  std::copy(cscLayers_fw.begin(),cscLayers_fw.end(),back_inserter(allEndcapCscME0));
  std::copy(me0Layers_fw.begin(),me0Layers_fw.end(),back_inserter(allEndcapCscME0));

  // allDetLayers: order is  all bw, all barrel, all fw
  allDetLayers.reserve(allBackward.size()+allBarrel.size()+allForward.size());
  std::copy(allBackward.begin(),allBackward.end(),back_inserter(allDetLayers));
  std::reverse(allDetLayers.begin(),allDetLayers.end());
  std::copy(allBarrel.begin(),allBarrel.end(),back_inserter(allDetLayers));
  std::copy(allForward.begin(),allForward.end(),back_inserter(allDetLayers));

  // number layers
  int sq=0;
  for (auto l : allDetLayers) 
    (*const_cast<DetLayer*>(l)).setSeqNum(sq++);


}
