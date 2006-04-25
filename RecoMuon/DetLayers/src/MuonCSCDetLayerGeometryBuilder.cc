#include "RecoMuon/DetLayers/src/MuonCSCDetLayerGeometryBuilder.h"

#include "RecoMuon/DetLayers/interface/MuDetRing.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

MuonCSCDetLayerGeometryBuilder::MuonCSCDetLayerGeometryBuilder() {
}

MuonCSCDetLayerGeometryBuilder::~MuonCSCDetLayerGeometryBuilder() {
}

vector<MuRingForwardLayer*> MuonCSCDetLayerGeometryBuilder::buildLayers(const CSCGeometry& geo) {
        
    vector<MuRingForwardLayer*> result;
    const CSCDetId cscDetId;
        
    for(int endcap = cscDetId.minEndcapId(); endcap != cscDetId.maxEndcapId(); endcap++) {
        for(int station = cscDetId.minStationId(); station != cscDetId.maxStationId(); station++) {
                
            vector<MuDetRing> muDetRings;
            for(int ring = cscDetId.minRingId(); ring != cscDetId.maxRingId(); ring++) {
    
                vector<GeomDet*> geomDets;
                for(int chamber = cscDetId.minChamberId(); chamber != cscDetId.maxChamberId(); chamber++) {
                    for(int layer = cscDetId.minLayerId(); layer != cscDetId.maxLayerId(); layer++) {

                        GeomDet* geomDet = geo->idToDet(cscDetId.rawIdMaker(endcap, station, ring, chamber, layer));
                        geomDets.push_back(geomDet);
                    }    
                }        
                
                muDetRings.push_back(new MuDetRing(geomDets));
            }
                
            result.push_back(new MuRingForwardLayer(muDetRings));    
        }
    }    
    
    return result;
}
