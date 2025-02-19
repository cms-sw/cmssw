#include <RecoMuon/DetLayers/src/MuonRPCDetLayerGeometryBuilder.h>

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
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

namespace rpcdetlayergeomsort {
  template <class T, class Scalar = typename T::Scalar>
  struct ExtractInnerRadius {
    typedef Scalar result_type;
    Scalar operator()(const T* p) const {return fabs(p->specificSurface().innerRadius());}
    Scalar operator()(const T& p) const {return fabs(p.specificSurface().innerRadius());}
  };
}


MuonRPCDetLayerGeometryBuilder::~MuonRPCDetLayerGeometryBuilder() {
}


// Builds the forward (first) and backward (second) layers
pair<vector<DetLayer*>, vector<DetLayer*> > 
MuonRPCDetLayerGeometryBuilder::buildEndcapLayers(const RPCGeometry& geo) {
  
  vector<DetLayer*> result[2];

  for (int endcap = -1; endcap<=1; endcap+=2) {
    int iendcap = (endcap==1) ? 0 : 1; // +1: forward, -1: backward
    
    // ME 1
    int firstStation=1;
        
    // ME 1/1
    for (int layer = RPCDetId::minLayerId; layer <= RPCDetId::maxLayerId; ++layer) { 
      vector<int> rolls;      
      std::vector<int> rings;
      int FirstStationRing = 1; 
      rings.push_back(FirstStationRing);
      for(int roll = RPCDetId::minRollId+1; 
	  roll <= RPCDetId::maxRollId; ++roll) {
	rolls.push_back(roll);
      }
      

      
      MuRingForwardDoubleLayer* ringLayer = buildLayer(endcap, rings,
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
      for(int roll = RPCDetId::minRollId+1; roll <= RPCDetId::maxRollId; 
	  ++roll) {
	rolls.push_back(roll);
      }
                
      MuRingForwardDoubleLayer* ringLayer = buildLayer(endcap, rings, firstStation , layer, rolls, geo);          
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
	for(int roll = RPCDetId::minRollId+1; roll <= RPCDetId::maxRollId; ++roll) {
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
MuonRPCDetLayerGeometryBuilder::buildLayer(int endcap,std::vector<int> rings, int station,
					   int layer,
					   vector<int>& rolls,
					   const RPCGeometry& geo) {

  const std::string metname = "Muon|RPC|RecoMuon|RecoMuonDetLayers|MuonRPCDetLayerGeometryBuilder";

  vector<const ForwardDetRing*> frontRings, backRings;


  for (std::vector<int>::iterator ring=rings.begin(); ring<rings.end();++ring){ 
    for (vector<int>::iterator roll = rolls.begin(); roll!=rolls.end(); ++roll) {    
      vector<const GeomDet*> frontDets, backDets;
      for(int sector = RPCDetId::minSectorForwardId; sector <= RPCDetId::maxSectorForwardId; ++sector) {
	for(int subsector = RPCDetId::minSubSectorForwardId; subsector <= RPCDetId::maxSectorForwardId; ++subsector) {
          RPCDetId rpcId(endcap,*ring, station,sector,layer,subsector, (*roll));
          bool isInFront = isFront(rpcId);
	  const GeomDet* geomDet = geo.idToDet(rpcId);
	  if (geomDet) {
	    
	    if(isInFront)
            {
              frontDets.push_back(geomDet);
            }
            else 
            {
              backDets.push_back(geomDet);
            }
	    LogTrace(metname) << "get RPC Endcap roll "
			      << rpcId
                              << (isInFront ? "front" : "back ")
			      << " at R=" << geomDet->position().perp()
			      << ", phi=" << geomDet->position().phi()
                              << ", Z=" << geomDet->position().z();
	    
	  }
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

  if(!backRings.empty() || !frontRings.empty())
  {
    typedef rpcdetlayergeomsort::ExtractInnerRadius<ForwardDetRing, float> SortRingByInnerR;
    precomputed_value_sort(frontRings.begin(), frontRings.end(), SortRingByInnerR());
    precomputed_value_sort(backRings.begin(), backRings.end(), SortRingByInnerR());
  
    result = new MuRingForwardDoubleLayer(frontRings, backRings);  
    LogTrace(metname) << "New layer with " << frontRings.size() 
                    << " front rings and " << backRings.size()
                    << " back rings, at Z " << result->position().z();
  }
  
  return result;
}


vector<DetLayer*> 
MuonRPCDetLayerGeometryBuilder::buildBarrelLayers(const RPCGeometry& geo) {
  const std::string metname = "Muon|RPC|RecoMuon|RecoMuonDetLayers|MuonRPCDetLayerGeometryBuilder";

  vector<DetLayer*> detlayers;
  vector<MuRodBarrelLayer*> result;
  int region =0;

  for(int station = RPCDetId::minStationId; station <= RPCDetId::maxStationId; station++) {

    vector<const GeomDet*> geomDets;

    for(int layer=RPCDetId::minLayerId; layer<= RPCDetId::maxLayerId;++layer){
      for(int sector = RPCDetId::minSectorId; sector <= RPCDetId::maxSectorId; sector++) {
	for(int subsector = RPCDetId::minSubSectorId; subsector <= RPCDetId::maxSubSectorId; subsector++) {
	  for(int wheel = RPCDetId::minRingBarrelId; wheel <= RPCDetId::maxRingBarrelId; wheel++) {
	    for(int roll=RPCDetId::minRollId+1; roll <= RPCDetId::maxRollId; roll++){         
	      const GeomDet* geomDet = geo.idToDet(RPCDetId(region,wheel,station,sector,layer,subsector,roll));
	      if (geomDet) {
		geomDets.push_back(geomDet);
		LogTrace(metname) << "get RPC Barrel roll " <<  RPCDetId(region,wheel,station,sector,layer,subsector,roll)
				  << " at R=" << geomDet->position().perp()
				  << ", phi=" << geomDet->position().phi() ;
	      }
	    }
	  }
	}
      }
    }
    makeBarrelLayers(geomDets, result);
  }
  

  for(vector<MuRodBarrelLayer*>::const_iterator it = result.begin(); it != result.end(); it++)
    detlayers.push_back((DetLayer*)(*it));

  return detlayers;
}


void MuonRPCDetLayerGeometryBuilder::makeBarrelLayers(vector<const GeomDet *> & geomDets,
                                                      vector<MuRodBarrelLayer*> & result)
{
  const std::string metname = "Muon|RPC|RecoMuon|RecoMuonDetLayers|MuonRPCDetLayerGeometryBuilder";

  //Sort in R
  precomputed_value_sort(geomDets.begin(), geomDets.end(),geomsort::DetR());

  // Clusterize in phi - phi0
  float resolution(25); // cm
  float r0 = float(geomDets.front()->position().perp());
  float rMin = - float(resolution);
  float rMax = float(geomDets.back()->position().perp()) - r0 + resolution;

  ClusterizingHistogram hisR( int((rMax-rMin)/resolution) + 1,
                                rMin, rMax);

  vector<const GeomDet*>::iterator first = geomDets.begin();
  vector<const GeomDet*>::iterator last = geomDets.end();

  for (vector<const GeomDet*>::iterator i=first; i!=last; i++){
    hisR.fill(float((*i)->position().perp())-r0);
    LogTrace(metname) << "R " << float((*i)->position().perp())-r0;
  }
  vector<float> rClust = hisR.clusterize(resolution);

  // LogTrace(metname) << "     Found " << phiClust.size() << " clusters in Phi, ";

  vector<const GeomDet*>::iterator layerStart = first;
  vector<const GeomDet*>::iterator separ = first;

  for (unsigned int i=0; i<rClust.size(); i++) {
    float rSepar;
    if (i<rClust.size()-1) {
      rSepar = (rClust[i] + rClust[i+1])/2.f;
    } else {
      rSepar = rMax;
    }

    // LogTrace(metname) << "       cluster " << i
    // << " phisepar " << phiSepar <<endl;
    while (separ < last && float((*separ)->position().perp())-r0 < rSepar ) {
      // LogTrace(metname) << "         roll at dphi:  " << float((*separ)->position().phi())-phi0;
      separ++;
    }

    if (int(separ-layerStart) > 0) {
      // we have a layer in R.  Now separate it into rods
      vector<const DetRod*> rods;
      vector<const GeomDet*> layerDets(layerStart, separ);
      makeBarrelRods(layerDets, rods);

      if (rods.size()!=0) {
        result.push_back(new MuRodBarrelLayer(rods));
        LogTrace(metname) << "    New MuRodBarrelLayer with " << rods.size()
                          << " rods, at R " << result.back()->specificSurface().radius();
      }
    }
    layerStart = separ;
  }
}


void
MuonRPCDetLayerGeometryBuilder::makeBarrelRods(vector<const GeomDet *> & geomDets,
                                               vector<const DetRod*> & result)
{
  const std::string metname = "Muon|RPC|RecoMuon|RecoMuonDetLayers|MuonRPCDetLayerGeometryBuilder";

  //Sort in phi
  precomputed_value_sort(geomDets.begin(), geomDets.end(),geomsort::DetPhi());

  // Clusterize in phi - phi0
  float resolution(0.01); // rad
  float phi0 = float(geomDets.front()->position().phi());
  float phiMin = - float(resolution);
  float phiMax = float(geomDets.back()->position().phi()) - phi0 + resolution;

  ClusterizingHistogram hisPhi( int((phiMax-phiMin)/resolution) + 1,
                                phiMin, phiMax);

  vector<const GeomDet*>::iterator first = geomDets.begin();
  vector<const GeomDet*>::iterator last = geomDets.end();

  for (vector<const GeomDet*>::iterator i=first; i!=last; i++){
    hisPhi.fill(float((*i)->position().phi())-phi0);
    LogTrace(metname) << "C " << float((*i)->position().phi())-phi0;
  }
  vector<float> phiClust = hisPhi.clusterize(resolution);

  // LogTrace(metname) << "     Found " << phiClust.size() << " clusters in Phi, ";

  vector<const GeomDet*>::iterator rodStart = first;
  vector<const GeomDet*>::iterator separ = first;

  for (unsigned int i=0; i<phiClust.size(); i++) {
    float phiSepar;
    if (i<phiClust.size()-1) {
      phiSepar = (phiClust[i] + phiClust[i+1])/2.f;
    } else {
      phiSepar = phiMax;
    }

    // LogTrace(metname) << "       cluster " << i
    // << " phisepar " << phiSepar <<endl;
    while (separ < last && float((*separ)->position().phi())-phi0 < phiSepar ) {
      // LogTrace(metname) << "         roll at dphi:  " << float((*separ)->position().phi())-phi0;
      separ++;
    }

    if (int(separ-rodStart) > 0) {
      result.push_back(new MuDetRod(rodStart, separ));
      LogTrace(metname) << "  New MuDetRod with " << int(separ-rodStart)
                        << " rolls at R=" << (*rodStart)->position().perp()
                        << ", phi=" << float((*rodStart)->position().phi());

    }
    rodStart = separ;
  }
}


bool MuonRPCDetLayerGeometryBuilder::isFront(const RPCDetId & rpcId)
{
  // ME1/2 is always in back
  //  if(rpcId.station() == 1 && rpcId.ring() == 2)  return false;

  bool result = false;
  int ring = rpcId.ring();
  int station = rpcId.station();
  // 20 degree rings are a little weird! not anymore from 17x
  if(ring == 1 && station > 1)
  {
    // RE2/1 RE3/1  Upscope Geometry
    /* goes (sector) (subsector)            1/3
    1 1 back   // front 
    1 2 front  // back  
    1 3 front  // front 
    2 1 front  // back  
    2 2 back   // from  
    2 3 back   // back  
                        
    */
    result = (rpcId.subsector() != 2);
    if(rpcId.sector()%2 == 0) result = !result;
    return result;
  }
  else
  {
    // 10 degree rings have odd subsectors in front
    result = (rpcId.subsector()%2 == 0);
  }
  return result;
}


