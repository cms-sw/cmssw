#include "RecoMuon/DetLayers/src/MuonCSCDetLayerGeometryBuilder.h"

#include "RecoMuon/DetLayers/interface/MuDetRing.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include <iostream>

MuonCSCDetLayerGeometryBuilder::MuonCSCDetLayerGeometryBuilder() {
}

MuonCSCDetLayerGeometryBuilder::~MuonCSCDetLayerGeometryBuilder() {
}

vector<MuRingForwardLayer*> MuonCSCDetLayerGeometryBuilder::buildLayers(const CSCGeometry& geo) {
        
    vector<MuRingForwardLayer*> result;
        
    for(int endcap = CSCDetId::minEndcapId(); endcap != CSCDetId::maxEndcapId(); endcap++) {
        for(int station = CSCDetId::minStationId(); station != CSCDetId::maxStationId(); station++) {
                
            vector<const ForwardDetRing*> muDetRings;
            for(int ring = CSCDetId::minRingId(); ring != CSCDetId::maxRingId(); ring++) {
    
                vector<const GeomDet*> geomDets;
                for(int chamber = CSCDetId::minChamberId(); chamber != CSCDetId::maxChamberId(); chamber++) {
		  //                    for(int layer = CSCDetId::minLayerId(); layer != CSCDetId::maxLayerId(); layer++) {

		  
		  const GeomDet* geomDet = geo.idToDet(CSCDetId(endcap, station, ring, chamber, 0));
		  if (geomDet) {
		    geomDets.push_back(geomDet);
		    cout << "get CSC chamber " <<  CSCDetId(endcap, station, ring, chamber, 0) << " " << geomDet << endl;
		  }
                }
                
		if (geomDets.size()!=0) {
                muDetRings.push_back(new MuDetRing(geomDets));
		cout << "New ring with " << geomDets.size() << " chambers" << endl;
		}
            }
                
            result.push_back(new MuRingForwardLayer(muDetRings));    
	    cout << "New layer with " << muDetRings.size() << " rings" << endl;
        }
    }    
    
    return result;
}
