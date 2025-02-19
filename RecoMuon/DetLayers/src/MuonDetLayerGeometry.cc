/** \file
 *
 *  $Date: 2009/07/03 09:12:48 $
 *  $Revision: 1.20 $
 *  \author N. Amapane - CERN
 */

#include <RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h>

#include <FWCore/Utilities/interface/Exception.h>
#include <TrackingTools/DetLayers/interface/DetLayer.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <DataFormats/GeometrySurface/interface/GeometricSorting.h>

#include <algorithm>

using namespace std;
using namespace geomsort;

MuonDetLayerGeometry::MuonDetLayerGeometry() {}

MuonDetLayerGeometry::~MuonDetLayerGeometry(){
  for(vector<DetLayer*>::const_iterator it = allDetLayers.begin(); it != allDetLayers.end(); ++it)
  {
    delete *it;
  }
}

void MuonDetLayerGeometry::addCSCLayers(pair<vector<DetLayer*>, vector<DetLayer*> > csclayers) {
    
  vector<DetLayer*>::const_iterator it;
  for(it=csclayers.first.begin(); it!=csclayers.first.end(); it++) {
    cscLayers_fw.push_back(*it);
    //    cscLayers_all.push_back(*it);
    allForward.push_back(*it);
    //    allEndcap.push_back(*it);
    //    allDetLayers.push_back(*it);
    
    detLayersMap[ makeDetLayerId(*it) ] = *it;
  }
  
  for(it=csclayers.second.begin(); it!=csclayers.second.end(); it++) {
    cscLayers_bk.push_back(*it);
    //    cscLayers_all.push_back(*it);
    allBackward.push_back(*it);
    //    allEndcap.push_back(*it);
    //    allDetLayers.push_back(*it);
    
    detLayersMap[ makeDetLayerId(*it) ] = *it;
  }    
}    

void MuonDetLayerGeometry::addRPCLayers(vector<DetLayer*> barrelLayers, pair<vector<DetLayer*>, vector<DetLayer*> > endcapLayers) {
  
  vector<DetLayer*>::const_iterator it;
  
  for (it=barrelLayers.begin();it!=barrelLayers.end();it++){
    rpcLayers_barrel.push_back(*it);
    //    rpcLayers_all.push_back(*it);
    allBarrel.push_back(*it);
    //    allDetLayers.push_back(*it);

    detLayersMap[ makeDetLayerId(*it) ] = *it;
  }
  for (it=endcapLayers.first.begin(); it!=endcapLayers.first.end(); it++){
    rpcLayers_fw.push_back(*it);
    //    rpcLayers_all.push_back(*it);
    //    rpcLayers_endcap.push_back(*it);
    allForward.push_back(*it);
    //    allEndcap.push_back(*it);
    //    allDetLayers.push_back(*it);

    detLayersMap[ makeDetLayerId(*it) ] = *it;
  }
  
  for (it=endcapLayers.second.begin(); it!=endcapLayers.second.end(); it++){
    rpcLayers_bk.push_back(*it);
    //    rpcLayers_all.push_back(*it);
    //    rpcLayers_endcap.push_back(*it);
    allBackward.push_back(*it);
    //    allEndcap.push_back(*it);
    //    allDetLayers.push_back(*it);

    detLayersMap[ makeDetLayerId(*it) ] = *it;
  }
  
}    

void MuonDetLayerGeometry::addDTLayers(vector<DetLayer*> dtlayers) {

    vector<DetLayer*>::const_iterator it;
    for(it=dtlayers.begin(); it!=dtlayers.end(); it++) {
        dtLayers.push_back(*it);
        allBarrel.push_back(*it);
	//        allDetLayers.push_back(*it);

	detLayersMap[ makeDetLayerId(*it) ] = *it;
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
  else throw cms::Exception("InvalidModuleIdentification"); // << detLayer->module();
}


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


const vector<DetLayer*>&
MuonDetLayerGeometry::allLayers() const {
    return allDetLayers;    
}    


const vector<DetLayer*>&
MuonDetLayerGeometry::allBarrelLayers() const {
    return allBarrel;    
}    

const vector<DetLayer*>&
MuonDetLayerGeometry::allEndcapLayers() const {
    return allEndcap;    
}    


const vector<DetLayer*>&
MuonDetLayerGeometry::allForwardLayers() const {
    return allForward;    
}    


const vector<DetLayer*>&
MuonDetLayerGeometry::allBackwardLayers() const {
    return allBackward;    
}    

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

  else throw cms::Exception("InvalidSubdetId")<< detId.subdetId();

  std::map<DetId,DetLayer*>::const_iterator layer = detLayersMap.find(id);
  if (layer == detLayersMap.end()) return 0;
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

  // allDetLayers: order is  all bw, all barrel, all fw
  allDetLayers.reserve(allBackward.size()+allBarrel.size()+allForward.size());
  std::copy(allBackward.begin(),allBackward.end(),back_inserter(allDetLayers));
  std::reverse(allDetLayers.begin(),allDetLayers.end());
  std::copy(allBarrel.begin(),allBarrel.end(),back_inserter(allDetLayers));
  std::copy(allForward.begin(),allForward.end(),back_inserter(allDetLayers));


}
