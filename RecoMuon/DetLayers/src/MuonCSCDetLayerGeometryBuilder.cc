#include <RecoMuon/DetLayers/src/MuonCSCDetLayerGeometryBuilder.h>

#include <RecoMuon/DetLayers/interface/MuRingForwardLayer.h>
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
      
      MuRingForwardLayer* layer = buildLayer(endcap, 1, rings, geo);          
      if (layer) result[i].push_back(layer);  
    }
    
    // ME/1/2 and 1/3 (= station 1, ring 2 and 3)
    {
      vector<int> rings;
      rings.push_back(2);
      rings.push_back(3);
      
      MuRingForwardLayer* layer = buildLayer(endcap, 1, rings, geo);          
      if (layer) result[i].push_back(layer);  
    }    
    
    // Stations 2,3,4
    for(int station = 2; station <= CSCDetId::maxStationId(); station++) {
      vector<int> rings;      
      for(int ring = CSCDetId::minRingId(); ring <= CSCDetId::maxRingId(); ring++) {
        rings.push_back(ring);
      }
      MuRingForwardLayer* layer = buildLayer(endcap, station, rings, geo);          
      if (layer) result[i].push_back(layer);
    }
  }
  pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(result[0], result[1]); 
  return res_pair;
}

MuRingForwardLayer* MuonCSCDetLayerGeometryBuilder::buildLayer(int endcap,
                                                               int station,
                                                               vector<int>& rings,
                                                               const CSCGeometry& geo) {
  const std::string metname = "Muon|CSC|RecoMuonDetLayers";
  MuRingForwardLayer* result=0;
  
  vector<const ForwardDetRing*> muDetRings;
  
  for (vector<int>::iterator ring = rings.begin(); ring!=rings.end(); ring++) {    
    vector<const GeomDet*> geomDets;
    for(int chamber = CSCDetId::minChamberId(); chamber <= CSCDetId::maxChamberId(); chamber++) {

      const GeomDet* geomDet = geo.idToDet(CSCDetId(endcap, station, (*ring), chamber, 0));
      if (geomDet) {
        geomDets.push_back(geomDet);
        LogTrace(metname) << "get CSC chamber "
                          <<  CSCDetId(endcap, station, (*ring), chamber, 0)
                          << " at R=" << geomDet->position().perp()
                          << ", phi=" << geomDet->position().phi();
      }
    }
    
    if (geomDets.size()!=0) {
      precomputed_value_sort(geomDets.begin(), geomDets.end(), geomsort::DetPhi());
      muDetRings.push_back(new MuDetRing(geomDets));
      LogTrace(metname) << "New MuDetRing with " << geomDets.size()
                        << " chambers at z="<< muDetRings.back()->position().z()
                        << " R1: " << muDetRings.back()->specificSurface().innerRadius()
                        << " R2: " << muDetRings.back()->specificSurface().outerRadius(); 
    }
  }
  
  if (muDetRings.size()!=0) {
    // How should they be sorted?
    //    precomputed_value_sort(muDetRods.begin(), muDetRods.end(), geomsort::ExtractZ<GeometricSearchDet,float>());
    result = new MuRingForwardLayer(muDetRings);  
    LogTrace(metname) << "New MuRingForwardLayer with " << muDetRings.size() 
                      << " rings, at Z " << result->position().z()
                      << " R1: " << result->specificSurface().innerRadius()
                      << " R2: " << result->specificSurface().outerRadius(); 
  }
  
  return result;
}
