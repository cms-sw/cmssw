#include <RecoMuon/DetLayers/src/MuonDTDetLayerGeometryBuilder.h>

#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <RecoMuon/DetLayers/interface/MuRodBarrelLayer.h>
#include <RecoMuon/DetLayers/interface/MuDetRod.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/CommonDetUnit/interface/DetSorting.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

using namespace std;

MuonDTDetLayerGeometryBuilder::MuonDTDetLayerGeometryBuilder() {
}

MuonDTDetLayerGeometryBuilder::~MuonDTDetLayerGeometryBuilder() {
}

vector<DetLayer*> 
MuonDTDetLayerGeometryBuilder::buildLayers(const DTGeometry& geo) {
        
  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonDTDetLayerGeometryBuilder";

  vector<DetLayer*> detlayers;
  vector<MuRodBarrelLayer*> result;
            
  for(int station = DTChamberId::minStationId; station <= DTChamberId::maxStationId; station++) {
    
    vector<const DetRod*> muDetRods;
    for(int sector = DTChamberId::minSectorId; sector <= DTChamberId::maxSectorId; sector++) {
      
      vector<const GeomDet*> geomDets;
      for(int wheel = DTChamberId::minWheelId; wheel <= DTChamberId::maxWheelId; wheel++) {		  
        const GeomDet* geomDet = geo.idToDet(DTChamberId(wheel, station, sector));
        if (geomDet) {
          geomDets.push_back(geomDet);
          LogTrace(metname) << "get DT chamber " <<  DTChamberId(wheel, station, sector)
                            << " at R=" << geomDet->position().perp()
                            << ", phi=" << geomDet->position().phi() ;
        }
      }
      
      if (geomDets.size()!=0) {
        precomputed_value_sort(geomDets.begin(), geomDets.end(), geomsort::DetZ());
        muDetRods.push_back(new MuDetRod(geomDets));
        LogTrace(metname) << "  New MuDetRod with " << geomDets.size()
                          << " chambers at R=" << muDetRods.back()->position().perp()
                          << ", phi=" << muDetRods.back()->position().phi();
      }
    }
    precomputed_value_sort(muDetRods.begin(), muDetRods.end(), geomsort::ExtractPhi<GeometricSearchDet,float>());
    result.push_back(new MuRodBarrelLayer(muDetRods));  
    LogDebug(metname) << "    New MuRodBarrelLayer with " << muDetRods.size()
                      << " rods, at R " << result.back()->specificSurface().radius();
  }
  
  for(vector<MuRodBarrelLayer*>::const_iterator it = result.begin(); it != result.end(); it++)
    detlayers.push_back((DetLayer*)(*it));
  
  return detlayers;
}
