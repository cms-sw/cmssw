#include <RecoMuon/DetLayers/src/MuonCSCDetLayerGeometryBuilder.h>

#include <RecoMuon/DetLayers/interface/MuDetRing.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

MuonCSCDetLayerGeometryBuilder::MuonCSCDetLayerGeometryBuilder() {
}

MuonCSCDetLayerGeometryBuilder::~MuonCSCDetLayerGeometryBuilder() {
}

pair<vector<DetLayer*>, vector<DetLayer*> > 
    MuonCSCDetLayerGeometryBuilder::buildLayers(const CSCGeometry& geo) {
        
    vector<MuRingForwardLayer*> result[2];
    vector<DetLayer*> detlayers[2];
    vector<const ForwardDetRing*> muDetRings;
            
    for(int i=0; i<2; i++) {
        
        int endcap = i+1;

        for(int ring = 2; ring <= 3; ring++) {
                
            vector<const GeomDet*> geomDets;
            for(int chamber = CSCDetId::minChamberId(); chamber <= CSCDetId::maxChamberId(); chamber++) {
		  
                const GeomDet* geomDet = geo.idToDet(CSCDetId(endcap, 1, ring, chamber, 0));
	            if (geomDet) {
		            geomDets.push_back(geomDet);
		            LogDebug("RecoMuonDetLayers") << "get CSC chamber " <<  CSCDetId(endcap, 1, ring, chamber, 0) << " " << geomDet;
		        }
            }
                
		    if (geomDets.size()!=0) {
                muDetRings.push_back(new MuDetRing(geomDets));
		        LogDebug("RecoMuonDetLayers") << "New ring with " << geomDets.size() << " chambers";
		    }
        }
                
        result[i].push_back(new MuRingForwardLayer(muDetRings));  
	    LogDebug("RecoMuonDetLayers") << "New layer with " << muDetRings.size() << " rings";
        muDetRings.clear();  
            
        for(int ring = 1; ring <= 4; ring+=4) {
                
            vector<const GeomDet*> geomDets;
            for(int chamber = CSCDetId::minChamberId(); chamber <= CSCDetId::maxChamberId(); chamber++) {
		  
                const GeomDet* geomDet = geo.idToDet(CSCDetId(endcap, 1, ring, chamber, 0));
	            if (geomDet) {
	                geomDets.push_back(geomDet);
	                LogDebug("RecoMuonDetLayers") << "get CSC chamber " <<  CSCDetId(endcap, station, ring, chamber, 0) << " " << geomDet;
	            }
            }
                
	        if (geomDets.size()!=0) {
                muDetRings.push_back(new MuDetRing(geomDets));
	            LogDebug("RecoMuonDetLayers") << "New ring with " << geomDets.size() << " chambers";
	        }
        }
                
        result[i].push_back(new MuRingForwardLayer(muDetRings));    
	    LogDebug("RecoMuonDetLayers") << "New layer with " << muDetRings.size() << " rings" << endl;
        muDetRings.clear();
    
        for(int station = 2; station <= CSCDetId::maxStationId(); station++) {

            for(int ring = CSCDetId::minRingId(); ring <= CSCDetId::maxRingId(); ring++) {
                
                vector<const GeomDet*> geomDets;
                for(int chamber = CSCDetId::minChamberId(); chamber <= CSCDetId::maxChamberId(); chamber++) {
		  
		            const GeomDet* geomDet = geo.idToDet(CSCDetId(endcap, station, ring, chamber, 0));
		            if (geomDet) {
		                geomDets.push_back(geomDet);
		                LogDebug("RecoMuonDetLayers") << "get CSC chamber " <<  CSCDetId(endcap, station, ring, chamber, 0) << " " << geomDet;
		            }
                }
                
		        if (geomDets.size()!=0) {
                    muDetRings.push_back(new MuDetRing(geomDets));
		            LogDebug("RecoMuonDetLayers") << "New ring with " << geomDets.size() << " chambers";
		        }
            }
                
            result[i].push_back(new MuRingForwardLayer(muDetRings));    
	        LogDebug("RecoMuonDetLayers") << "New layer with " << muDetRings.size() << " rings";
            muDetRings.clear();
        }
        
        for(vector<MuRingForwardLayer*>::const_iterator it = result[i].begin(); it != result[i].end(); it++)
                detlayers[i].push_back((DetLayer*)(*it));
    }    
    
    pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(detlayers[0], detlayers[1]); 
    return res_pair;
}
