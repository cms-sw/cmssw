#include <RecoMuon/DetLayers/src/MuonGEMDetLayerGeometryBuilder.h>

#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <RecoMuon/DetLayers/interface/MuRingForwardDoubleLayer.h>
#include <RecoMuon/DetLayers/interface/MuRodBarrelLayer.h>
#include <RecoMuon/DetLayers/interface/MuDetRing.h>
#include <RecoMuon/DetLayers/interface/MuDetRod.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/CommonDetUnit/interface/DetSorting.h>
#include "Utilities/BinningTools/interface/ClusterizingHistogram.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

using namespace std;

MuonGEMDetLayerGeometryBuilder::~MuonGEMDetLayerGeometryBuilder() {
}


// Builds the forward (first) and backward (second) layers
pair<vector<DetLayer*>, vector<DetLayer*> > 
MuonGEMDetLayerGeometryBuilder::buildEndcapLayers(const GEMGeometry& geo) {
  
  vector<DetLayer*> result[2];

  for (int endcap = -1; endcap<=1; endcap+=2) {
    int iendcap = (endcap==1) ? 0 : 1; // +1: forward, -1: backward

    for(int station = GEMDetId::minStationId; station <= GEMDetId::maxStationId; ++station) {
      for(int layer = GEMDetId::minLayerId; layer <= GEMDetId::maxLayerId; ++layer) { 
	vector<int> rolls;      
	std::vector<int> rings;
	for(int ring = GEMDetId::minRingId; ring <= GEMDetId::maxRingId; ++ring) {
	  rings.push_back(ring);
	}
	for(int roll = GEMDetId::minRollId+1; roll <= GEMDetId::maxRollId; ++roll) {
	  rolls.push_back(roll);
	}
                
	MuRingForwardDoubleLayer* ringLayer = buildLayer(endcap, rings, station, layer, rolls, geo);          
	if (ringLayer) result[iendcap].push_back(ringLayer);
      }
    }
    
  }
  pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(result[0], result[1]); 
  return res_pair;

}



MuRingForwardDoubleLayer* 
MuonGEMDetLayerGeometryBuilder::buildLayer(int endcap,std::vector<int> rings, int station,
					   int layer,
					   vector<int>& rolls,
					   const GEMGeometry& geo) {

  const std::string metname = "Muon|GEM|RecoMuon|RecoMuonDetLayers|MuonGEMDetLayerGeometryBuilder";

  vector<const ForwardDetRing*> frontRings, backRings;


  for (std::vector<int>::iterator ring=rings.begin(); ring<rings.end();++ring){ 
    for (vector<int>::iterator roll = rolls.begin(); roll!=rolls.end(); ++roll) {    
      vector<const GeomDet*> frontDets, backDets;
      for(int chamber = GEMDetId::minChamberId+1; chamber <= GEMDetId::maxChamberId; chamber++) {
          GEMDetId gemId(endcap,*ring, station,layer,chamber, (*roll));
          bool isInFront = isFront(gemId);
	  const GeomDet* geomDet = geo.idToDet(gemId);
	  if (geomDet) {
	    
	    if(isInFront)
            {
              frontDets.push_back(geomDet);
            }
            else 
            {
              backDets.push_back(geomDet);
            }
	    LogTrace(metname) << "get GEM Endcap roll "
			      << gemId
                              << (isInFront ? "front" : "back ")
			      << " at R=" << geomDet->position().perp()
			      << ", phi=" << geomDet->position().phi()
                              << ", Z=" << geomDet->position().z();
   
	  }
      }
      if (frontDets.size()!=0) {
	precomputed_value_sort(frontDets.begin(), frontDets.end(), geomsort::DetPhi());
	frontRings.push_back(new MuDetRing(frontDets));
	LogTrace(metname) << "New front ring with " << frontDets.size()
			  << " chambers at z="<< frontRings.back()->position().z();
      }
      if (backDets.size()!=0) {
        precomputed_value_sort(backDets.begin(), backDets.end(), geomsort::DetPhi());
        backRings.push_back(new MuDetRing(backDets));
        LogTrace(metname) << "New back ring with " << backDets.size()
                          << " chambers at z="<< backRings.back()->position().z();
      }
    }
  }

   MuRingForwardDoubleLayer * result = 0;

  
  return result;
}


bool MuonGEMDetLayerGeometryBuilder::isFront(const GEMDetId & gemId)
{

  bool result = false;
//  int ring = gemId.ring();
//  int station = gemId.station();
  int chamber = gemId.chamber();
// 20 degree rings are a little weird! not anymore from 17x
//  if(ring == 1 && station > 1)
//  {
//    result = (gemId.subsector() != 2);
    if(chamber%2 == 0) result = !result;
    return result;
//  }
//  else
//  {
    // 10 degree rings have odd subsectors in front
//    result = (gemId.subsector()%2 == 0);
//  }
//  return result;
}

