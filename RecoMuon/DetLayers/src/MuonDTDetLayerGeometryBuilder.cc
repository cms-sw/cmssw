#include <RecoMuon/DetLayers/src/MuonDTDetLayerGeometryBuilder.h>

#include <RecoMuon/DetLayers/interface/MuDetRod.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

MuonDTDetLayerGeometryBuilder::MuonDTDetLayerGeometryBuilder() {
}

MuonDTDetLayerGeometryBuilder::~MuonDTDetLayerGeometryBuilder() {
}

vector<DetLayer*> 
    MuonDTDetLayerGeometryBuilder::buildLayers(const DTGeometry& geo) {
        
    vector<DetLayer*> detlayers;
    vector<MuRodBarrelLayer*> result;
    vector<const DetRod*> muDetRings;
            
    for(int station = DTChamberId::minStationId; station <= DTChamberId::maxStationId; station++) {

        for(int wheel = DTChamberId::minWheelId; wheel <= DTChamberId::maxWheelId; wheel++) {
                
            vector<const GeomDet*> geomDets;
            for(int sector = DTChamberId::minSectorId; sector <= DTChamberId::maxSectorId; sector++) {
		  
                const GeomDet* geomDet = geo.idToDet(DTChamberId(wheel, station, sector));
	            if (geomDet) {
		            geomDets.push_back(geomDet);
		            LogDebug("RecoMuonDetLayers") << "get DT chamber " <<  DTChamberId(wheel, station, sector) << " " << geomDet;
		        }
            }
                
		    if (geomDets.size()!=0) {
                muDetRings.push_back(new MuDetRod(geomDets));
		        LogDebug("RecoMuonDetLayers") << "New wheel with " << geomDets.size() << " chambers";
		    }
        }
                
        result.push_back(new MuRodBarrelLayer(muDetRings));  
	    LogDebug("RecoMuonDetLayers") << "New layer with " << muDetRings.size() << " wheels";
        muDetRings.clear();  
    }    

    for(vector<MuRodBarrelLayer*>::const_iterator it = result.begin(); it != result.end(); it++)
            detlayers.push_back((DetLayer*)(*it));

    return detlayers;
}
