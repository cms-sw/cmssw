#include <RecoMuon/DetLayers/src/MuonCSCDetLayerGeometryBuilder.h>

#include <RecoMuon/DetLayers/interface/MuRingForwardDoubleLayer.h>
#include <RecoMuon/DetLayers/interface/MuDetRing.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/CommonDetUnit/interface/DetSorting.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

using namespace std;

pair<vector<DetLayer*>, vector<DetLayer*> > 
MuonCSCDetLayerGeometryBuilder::buildLayers(const CSCGeometry& geo) {

  vector<DetLayer*> result[2]; // one for each endcap
  
  for(int i=0; i<2; i++) {        
    
    int endcap = i+1;
    
    // ME/1/1a (= station 1, ring 4) and ME/1/1b (= station 1, ring 1)
    {
      vector<int> rings;
      rings.push_back(4);
      rings.push_back(1);
      
      MuRingForwardDoubleLayer* layer = buildLayer(endcap, 1, rings, geo);          
      if (layer) result[i].push_back(layer);  
    }
    
    // ME/1/2 and 1/3 (= station 1, ring 2 and 3)
    {
      vector<int> rings;
      rings.push_back(2);
      rings.push_back(3);
      
      MuRingForwardDoubleLayer* layer = buildLayer(endcap, 1, rings, geo);          
      if (layer) result[i].push_back(layer);  
    }    
    
    // Stations 2,3,4
    for(int station = 2; station <= CSCDetId::maxStationId(); station++) {
      vector<int> rings;      
      for(int ring = CSCDetId::minRingId(); ring <= CSCDetId::maxRingId(); ring++) {
        rings.push_back(ring);
      }
      MuRingForwardDoubleLayer* layer = buildLayer(endcap, station, rings, geo);          
      if (layer) result[i].push_back(layer);
    }
  }
  pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(result[0], result[1]); 
  return res_pair;
}

MuRingForwardDoubleLayer* MuonCSCDetLayerGeometryBuilder::buildLayer(int endcap,
                                                               int station,
                                                               vector<int>& rings,
                                                               const CSCGeometry& geo) {
  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonCSCDetLayerGeometryBuilder";
  MuRingForwardDoubleLayer* result=0;
  
  vector<const ForwardDetRing*> frontRings, backRings;
  
  for (vector<int>::iterator ring = rings.begin(); ring!=rings.end(); ring++) {    
    vector<const GeomDet*> frontGeomDets, backGeomDets;
    for(int chamber = CSCDetId::minChamberId(); chamber <= CSCDetId::maxChamberId(); chamber++) {
      CSCDetId detId(endcap, station, (*ring), chamber, 0);
      const GeomDet* geomDet = geo.idToDet(detId);
      // we sometimes loop over more chambers than there are in ring
      bool isInFront = isFront(station, *ring, chamber);
      if(geomDet != 0)
      {
        if(isInFront)
        {
          frontGeomDets.push_back(geomDet);
        }
        else
        {
          backGeomDets.push_back(geomDet);
        }
        LogTrace(metname) << "get CSC chamber "
                          <<  CSCDetId(endcap, station, (*ring), chamber, 0)
                          << " at R=" << geomDet->position().perp()
                          << ", phi=" << geomDet->position().phi()
                          << ", z= " << geomDet->position().z() 
                          << " isFront? " << isInFront;
      }
    }

    if(!backGeomDets.empty())
    {
      backRings.push_back(makeDetRing(backGeomDets));
    }

    if(!frontGeomDets.empty())
    {
      frontRings.push_back(makeDetRing(frontGeomDets));
      assert(!backGeomDets.empty());
      float frontz = frontRings[0]->position().z();
      float backz  = backRings[0]->position().z();
      assert(fabs(frontz) < fabs(backz));
    }
  }
  
  // How should they be sorted?
  //    precomputed_value_sort(muDetRods.begin(), muDetRods.end(), geomsort::ExtractZ<GeometricSearchDet,float>());
  result = new MuRingForwardDoubleLayer(frontRings, backRings);  
  LogTrace(metname) << "New MuRingForwardLayer with " << frontRings.size() 
                    << " and " << backRings.size()
                    << " rings, at Z " << result->position().z()
                    << " R1: " << result->specificSurface().innerRadius()
                    << " R2: " << result->specificSurface().outerRadius(); 
  return result;
}


bool MuonCSCDetLayerGeometryBuilder::isFront(int station, int ring, int chamber)
{
  bool result = false;
  
  bool isOverlapping = !(station == 1 && ring == 3);
  // not overlapping means back
  if(isOverlapping)
  {
    bool isEven = (chamber%2==0);
    // odd chambers are bolted to the iron, which faces
    // forward in 1&2, backward in 3&4, so...
    result = (station<3) ? isEven : !isEven;
  }
  return result;
}



MuDetRing * MuonCSCDetLayerGeometryBuilder::makeDetRing(vector<const GeomDet*> & geomDets)
{
    const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonCSCDetLayerGeometryBuilder";


    precomputed_value_sort(geomDets.begin(), geomDets.end(), geomsort::DetPhi());
    MuDetRing * result = new MuDetRing(geomDets);
    LogTrace(metname) << "New MuDetRing with " << geomDets.size()
                        << " chambers at z="<< result->position().z()
                        << " R1: " << result->specificSurface().innerRadius()
                        << " R2: " << result->specificSurface().outerRadius();
    return result;
}

