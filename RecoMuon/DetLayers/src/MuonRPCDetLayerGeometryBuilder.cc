#include <RecoMuon/DetLayers/src/MuonRPCDetLayerGeometryBuilder.h>

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <RecoMuon/DetLayers/interface/MuRingForwardLayer.h>
#include <RecoMuon/DetLayers/interface/MuRodBarrelLayer.h>
#include <RecoMuon/DetLayers/interface/MuDetRing.h>
#include <RecoMuon/DetLayers/interface/MuDetRod.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/CommonDetUnit/interface/DetSorting.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

using namespace std;

MuonRPCDetLayerGeometryBuilder::~MuonRPCDetLayerGeometryBuilder() {
}


// Builds the forward (first) and backward (second) layers
pair<vector<DetLayer*>, vector<DetLayer*> > 
MuonRPCDetLayerGeometryBuilder::buildEndcapLayers(const RPCGeometry& geo) {
  
  vector<DetLayer*> result[2];
  //std::cout<<"I am in the endcap!"<<std::endl;

  for (int endcap = -1; endcap<=1; endcap+=2) {
    int iendcap = (endcap==1) ? 0 : 1; // +1: forward, -1: backward
    std::cout <<"Which endcap am I ? "<<endcap<<" index " <<iendcap<<std::endl;
    
    // ME 1
    int firstStation=1;
        
    // ME 1/1
    for (int layer = RPCDetId::minLayerId; layer <= RPCDetId::maxLayerId; ++layer) { 
      vector<int> rolls;      
      std::vector<int> rings;
      int FirstStationRing = 1; 
      rings.push_back(FirstStationRing);
      for(int roll = RPCDetId::minRollId; 
	  roll <= RPCDetId::maxRollId; ++roll) {
	rolls.push_back(roll);
      }
      

      
      MuRingForwardLayer* ringLayer = buildLayer(endcap, rings,
						 firstStation , layer, 
						 rolls, geo);          
      if (ringLayer) result[iendcap].push_back(ringLayer);
      
    }
        
    // ME 1/2 and ME1/3       
    for(int layer = RPCDetId::minLayerId; layer <= RPCDetId::maxLayerId; ++layer) { 
      vector<int> rolls;      
      std::vector<int> rings;
      for(int ring = 2; ring <= 3; ++ring) {
	rings.push_back(ring);
      }
      for(int roll = RPCDetId::minRollId; roll <= RPCDetId::maxRollId; 
	  ++roll) {
	rolls.push_back(roll);
      }
                
      MuRingForwardLayer* ringLayer = buildLayer(endcap, rings, firstStation , layer, rolls, geo);          
      if (ringLayer) result[iendcap].push_back(ringLayer);
    }
  

    // ME 2 and ME 3 
    for(int station = 2; station <= RPCDetId::maxStationId; ++station) {
      for(int layer = RPCDetId::minLayerId; layer <= RPCDetId::maxLayerId; ++layer) { 
	vector<int> rolls;      
	std::vector<int> rings;
	for(int ring = RPCDetId::minRingForwardId; ring <= RPCDetId::maxRingForwardId; ++ring) {
	  rings.push_back(ring);
	}
	for(int roll = RPCDetId::minRollId; roll <= RPCDetId::maxRollId; ++roll) {
	  rolls.push_back(roll);
	}
                
	MuRingForwardLayer* ringLayer = buildLayer(endcap, rings, station, layer, rolls, geo);          
	if (ringLayer) result[iendcap].push_back(ringLayer);
      }
    }
    
  }
  std::cout<<" Results size0 and size1 "<<result[0].size()<<" "<<result[1].size()<<std::endl;
  pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(result[0], result[1]); 
  return res_pair;

}



MuRingForwardLayer* 
MuonRPCDetLayerGeometryBuilder::buildLayer(int endcap,std::vector<int> rings, int station,
					   int layer,
					   vector<int>& rolls,
					   const RPCGeometry& geo) {

  std::cout <<"Building an endcap Layer"<<std::endl;
  MuRingForwardLayer* result=0;

  //std::cout<<"Number of rolls "<<rolls.size()<<std::endl;

  vector<const ForwardDetRing*> muDetRings;

  for (std::vector<int>::iterator ring=rings.begin(); ring<rings.end();++ring){ 
    for (vector<int>::iterator roll = rolls.begin(); roll!=rolls.end(); ++roll) {    
      vector<const GeomDet*> geomDets;
      for(int sector = RPCDetId::minSectorForwardId; sector <= RPCDetId::maxSectorForwardId; ++sector) {
	for(int subsector = RPCDetId::minSubSectorForwardId; subsector <= RPCDetId::maxSectorForwardId; ++subsector) {
	  //std::cout<<"ring station, sector, layer"<<ring<<" "<<station<<" "<<sector<<" "<<layer<<std::endl;
	  const GeomDet* geomDet = geo.idToDet(RPCDetId(endcap,*ring, station,sector,layer,subsector, (*roll)));
	  //std::cout<<geomDet<<std::endl;
	  if (geomDet) {
	    
	    geomDets.push_back(geomDet);
	    std::cout << "get RPC chamber "
		      <<  RPCDetId(endcap,*ring, station,sector,layer,subsector, (*roll))
		      << " at z" << geomDet->position().z()
		      << ", phi=" << geomDet->position().phi()<<std::endl;
	    
	    
	    LogDebug("Muon|RPC|RecoMuonDetLayers") << "get RPC chamber "
						   <<  RPCDetId(endcap,*ring, station,sector,layer,subsector, (*roll))
						   << " at R=" << geomDet->position().perp()
						   << ", phi=" << geomDet->position().phi();
	    
	  }
	}
      }
      if (geomDets.size()!=0) {
	//std::cout<<"++++++++++ Got "<<geomDets.size()<<" rolls on this layer"<<std::endl;
	precomputed_value_sort(geomDets.begin(), geomDets.end(), geomsort::DetPhi());
	muDetRings.push_back(new MuDetRing(geomDets));
	LogDebug("Muon|RPC|RecoMuonDetLayers") << "New ring with " << geomDets.size()
					       << " chambers at z="<< muDetRings.back()->position().z();
      }
    }
  }
  
  //std::cout <<"Number of Det Rings "<<muDetRings.size()<<std::endl;
  
  if (muDetRings.size()!=0) {
    result = new MuRingForwardLayer(muDetRings);  
    LogDebug("Muon|RPC|RecoMuonDetLayers") << "New layer with " << muDetRings.size() 
                                           << " rolls, at Z " << result->position().z();
  }

  return result;
}


vector<DetLayer*> 
MuonRPCDetLayerGeometryBuilder::buildBarrelLayers(const RPCGeometry& geo) {
        
  vector<DetLayer*> detlayers;
  vector<MuRodBarrelLayer*> result;
  int region =0;

  for(int station = RPCDetId::minStationId; station <= RPCDetId::maxStationId; station++) {
    for(int layer=RPCDetId::minLayerId; layer<= RPCDetId::maxLayerId;++layer){
      
      vector<const DetRod*> muDetRods;
      for(int sector = RPCDetId::minSectorId; sector <= RPCDetId::maxSectorId; sector++) {
	for(int subsector = RPCDetId::minSubSectorId; subsector <= RPCDetId::maxSubSectorId; subsector++) {

	  vector<const GeomDet*> geomDets;
	  for(int wheel = RPCDetId::minRingBarrelId; wheel <= RPCDetId::maxRingBarrelId; wheel++) {
	    for(int roll=RPCDetId::minRollId+1; roll <= RPCDetId::maxRollId; roll++){         
	      const GeomDet* geomDet = geo.idToDet(RPCDetId(region,wheel,station,sector,layer,subsector,roll));
	      if (geomDet) {
		geomDets.push_back(geomDet);
		LogDebug("Muon|RPC|RecoMuonDetLayers") << "get RPC roll " <<  RPCDetId(region,wheel,station,sector,layer,subsector,roll)
						       << " at R=" << geomDet->position().perp()
						       << ", phi=" << geomDet->position().phi() ;
	      }
	    }
	  }
                
	  if (geomDets.size()!=0) {
	    muDetRods.push_back(new MuDetRod(geomDets));
	    LogDebug("Muon|RPC|RecoMuonDetLayers") << "  New MuDetRod with " << geomDets.size()

						   << " chambers at R=" << muDetRods.back()->position().perp()
						   << ", phi=" << muDetRods.back()->position().phi();
	  }
	}
      }
      if (muDetRods.size()!=0) {
	result.push_back(new MuRodBarrelLayer(muDetRods));  
	LogDebug("Muon|RPC|RecoMuonDetLayers") << "    New MuRodBarrelLayer with " << muDetRods.size()
					       << " rods, at R " << result.back()->specificSurface().radius();
      }
    }
  }
  
  for(vector<MuRodBarrelLayer*>::const_iterator it = result.begin(); it != result.end(); it++)
    detlayers.push_back((DetLayer*)(*it));

  return detlayers;
}
