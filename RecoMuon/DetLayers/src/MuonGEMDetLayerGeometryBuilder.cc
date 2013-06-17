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
  std::cout<<"[MuonGEMDetLayerGeometryBuilder] buildEndcapLayers :: "<<std::endl;
  for (int endcap = -1; endcap<=1; endcap+=2) {
    int iendcap = (endcap==1) ? 0 : 1; // +1: forward, -1: backward

    std::cout <<"Building DetLayer from Station "<<GEMDetId::minStationId<<" to Station "<<GEMDetId::maxStationId<<std::endl;
    std::cout <<"              and from Layer   "<<GEMDetId::minLayerId  <<" to Layer   "<<GEMDetId::maxLayerId<<std::endl;
    std::cout <<"              and from Ring    "<<GEMDetId::minRingId   <<" to Ring    "<<GEMDetId::maxRingId<<std::endl;
    std::cout <<"              and from Roll    "<<GEMDetId::minRollId+1 <<" to Roll    "<<GEMDetId::maxRollId<<std::endl;
    for(int station = GEMDetId::minStationId; station <= GEMDetId::minStationId; ++station) {
      for(int layer = GEMDetId::minLayerId; layer <= GEMDetId::maxLayerId; ++layer) { 
	vector<int> rolls;      
	std::vector<int> rings;
	std::vector<int> chambers;
	for(int ring = GEMDetId::minRingId; ring <= GEMDetId::maxRingId; ++ring) {
	  rings.push_back(ring);
	}
	for(int roll = GEMDetId::minRollId+1; roll <= GEMDetId::maxRollId; ++roll) {
	  rolls.push_back(roll);
	}
	for(int chamber = GEMDetId::minChamberId; chamber <= GEMDetId::maxChamberId; chamber++ ){	     chambers.push_back(chamber);
	}
std::cout<<"Endcap = "<<endcap<<" Station = "<<station<<" Layer = "<<layer;
	std::cout<<" rolls.size = "<<rolls.size()<<" rings.size = "<<rings.size()<<" chambers.size = "<<chambers.size()<<std::endl;         
	MuRingForwardDoubleLayer* ringLayer = buildLayer(endcap, rings, station, layer, chambers, rolls, geo);          
std::cout<<"a"<<std::endl;
	if (ringLayer) result[iendcap].push_back(ringLayer);
std::cout<<"b"<<std::endl;
      }
std::cout<<"c"<<std::endl;
    }
std::cout<<"d"<<std::endl;
    
  }
  pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(result[0], result[1]); 
  std::cout<<"respair.first.size = "<<res_pair.first.size()<<" respair.second.size = "<<res_pair.second.size()<<std::endl;
  return res_pair;

}



MuRingForwardDoubleLayer* 
MuonGEMDetLayerGeometryBuilder::buildLayer(int endcap,vector<int>& rings, int station,
					   int layer,
					   vector<int>& chambers,
					   vector<int>& rolls,
					   const GEMGeometry& geo) {

  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonGEMDetLayerGeometryBuilder";
  MuRingForwardDoubleLayer * result = 0;
  vector<const ForwardDetRing*> frontRings, backRings;


std::cout<<"Endcap = "<<endcap<<" Station = "<<station<<" Layer = "<<layer;
	std::cout<<" rolls.size = "<<rolls.size()<<" rings.size = "<<rings.size()<<" chambers.size = "<<chambers.size()<<std::endl;         
//  for (std::vector<int>::iterator ring=rings.begin(); ring!=rings.end();ring++){ 
  for (std::vector<int>::iterator ring=rings.begin(); ring!=rings.end()-2;ring++){ 
std::cout<<"a"<<std::endl;
    for (vector<int>::iterator roll = rolls.begin(); roll!=rolls.end(); roll++) {    
std::cout<<"b"<<std::endl;
      vector<const GeomDet*> frontDets, backDets;
std::cout<<"c"<<std::endl;
      for(std::vector<int>::iterator chamber=chambers.begin()+1; chamber<chambers.end(); chamber++) {
          GEMDetId gemId(endcap,(*ring), station,layer,(*chamber), (*roll));
	  std::cout <<" DET LAYER Detid " << gemId<<std::endl;
 	  const GeomDet* geomDet = geo.idToDet(gemId);
	  
	  std::cout<<"geoDet is: "<<geomDet<<" and geo is: "<<&geo<<" and &gemId: "<<&gemId<<" gemId "<<gemId<<std::endl;
	  if (geomDet !=0) {
	    bool isInFront = isFront(gemId);
	    if(isInFront)
            {
	      std::cout<<"d front"<<std::endl;
              frontDets.push_back(geomDet);
            }
            else 
            {
	      std::cout<<"d back"<<std::endl;
              backDets.push_back(geomDet);
            }
	    LogTrace(metname) << "get GEM Endcap roll "
			      << gemId
                              << (isInFront ? "front" : "back ")
			      << " at R=" << geomDet->position().perp()
			      << ", phi=" << geomDet->position().phi()
                              << ", Z=" << geomDet->position().z();
std::cout<<"d2"<<std::endl;
   
	  }
      }
std::cout<<"e"<<std::endl;
      if (frontDets.size()!=0) {
	precomputed_value_sort(frontDets.begin(), frontDets.end(), geomsort::DetPhi());
	frontRings.push_back(new MuDetRing(frontDets));
	LogTrace(metname) << "New front ring with " << frontDets.size()
			  << " chambers at z="<< frontRings.back()->position().z();
	std::cout << "New front ring with " << frontDets.size()
		  << " chambers at z="<< frontRings.back()->position().z() <<std::endl;
      }
      if (backDets.size()!=0) {
        precomputed_value_sort(backDets.begin(), backDets.end(), geomsort::DetPhi());
        backRings.push_back(new MuDetRing(backDets));
        LogTrace(metname) << "New back ring with " << backDets.size()
                          << " chambers at z="<< backRings.back()->position().z();
	std::cout << "New back ring with " << backDets.size()
		  << " chambers at z="<< backRings.back()->position().z()<<std::endl;
      }
/*
//if(!backDets.empty())
 if (backDets.size()!=0) 
    {
std::cout<<"f"<<std::endl;
      backRings.push_back(makeDetRing(backDets));
    }

//    if(!frontDets.empty())
      if (frontDets.size()!=0) 
    {
std::cout<<"f"<<sTd"::Endl;
      Frontrings".push_back(makeDetRing(frontDets));
      assert(!backDets.empty());
      float frontz = frontRings[0]->position().z();
      float backz  = backRings[0]->position().z();
      assert(fabs(frontz) < fabs(backz));
    }
//std::cout<<"g"<<std::endl;
*/
    }
std::cout<<"h"<<std::endl;
  }
  std::cout<<"i"<<std::endl;
  // How should they be sorted?
  //    precomputed_value_sort(muDetRods.begin(), muDetRods.end(), geomsort::ExtractZ<GeometricSearchDet,float>());                                   
  result = new MuRingForwardDoubleLayer(frontRings, backRings);
  //result = 0;
  std::cout<<"l"<<std::endl;
  
  LogTrace(metname) << "New MuRingForwardLayer with " << frontRings.size()
                    << " and " << backRings.size()
                    << " rings, at Z " << result->position().z()
                    << " R1: " << result->specificSurface().innerRadius()
                    << " R2: " << result->specificSurface().outerRadius();
  
  return result;
std::cout<<"m"<<std::endl;
}


bool MuonGEMDetLayerGeometryBuilder::isFront(const GEMDetId & gemId)
{

  bool result = false;
//  int ring = gemId.ring();
//  int station = gemId.station();
  int chamber = gemId.chamber();
std::cout<<"in front function and champer is: "<<chamber<<std::endl;
// 20 degree rings are a little weird! not anymore from 17x
//  if(ring == 1 && station > 1)
//  {
//    result = (gemId.subsector() != 2);
    if(chamber%2 == 0) result = !result;
std::cout<<"in front result: "<<result<<std::endl;
    return result;
//  }
//  else
//  {
    // 10 degree rings have odd subsectors in front
//    result = (gemId.subsector()%2 == 0);
//  }
//  return result;
}

MuDetRing * MuonGEMDetLayerGeometryBuilder::makeDetRing(vector<const GeomDet*> & geomDets)
{
    const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonGEMDetLayerGeometryBuilder";


    precomputed_value_sort(geomDets.begin(), geomDets.end(), geomsort::DetPhi());
    MuDetRing * result = new MuDetRing(geomDets);
    LogTrace(metname) << "New MuDetRing with " << geomDets.size()
                        << " chambers at z="<< result->position().z()
                        << " R1: " << result->specificSurface().innerRadius()
                        << " R2: " << result->specificSurface().outerRadius();
    return result;
}
