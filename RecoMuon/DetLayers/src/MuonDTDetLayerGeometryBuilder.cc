#include <RecoMuon/DetLayers/src/MuonDTDetLayerGeometryBuilder.h>

#include <RecoMuon/DetLayers/interface/MuDetRing.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

MuonDTDetLayerGeometryBuilder::MuonDTDetLayerGeometryBuilder() {
}

MuonDTDetLayerGeometryBuilder::~MuonDTDetLayerGeometryBuilder() {
}

pair<vector<DetLayer*>, vector<DetLayer*> > 
    MuonDTDetLayerGeometryBuilder::buildLayers(const DTGeometry& geo) {
        
    vector<MuRingForwardLayer*> result[2];
    vector<DetLayer*> detlayers[2];
    vector<const ForwardDetRing*> muDetRings;
/*            
    for(int endcap = DTDetId::minEndcapId(); endcap != DTDetId::maxEndcapId(); endcap++) {

        for(int ring = 2; ring <= 3; ring++) {
                
            vector<const GeomDet*> geomDets;
            for(int chamber = DTDetId::minChamberId(); chamber <= DTDetId::maxChamberId(); chamber++) {
		  
                const GeomDet* geomDet = geo.idToDet(DTDetId(endcap, 1, ring, chamber, 0));
	            if (geomDet) {
		            geomDets.push_back(geomDet);
		            LogDebug("xxx") << "get DT chamber " <<  DTDetId(endcap, 1, ring, chamber, 0) << " " << geomDet;
		        }
            }
                
		    if (geomDets.size()!=0) {
                muDetRings.push_back(new MuDetRing(geomDets));
		        LogDebug("xxx") << "New ring with " << geomDets.size() << " chambers";
		    }
        }
                
        result[endcap].push_back(new MuRingForwardLayer(muDetRings));  
	    LogDebug("xxx") << "New layer with " << muDetRings.size() << " rings";
        muDetRings.clear();  
            
        for(int ring = 1; ring <= 4; ring+=4) {
                
            vector<const GeomDet*> geomDets;
            for(int chamber = DTDetId::minChamberId(); chamber <= DTDetId::maxChamberId(); chamber++) {
		  
                const GeomDet* geomDet = geo.idToDet(DTDetId(endcap, 1, ring, chamber, 0));
	            if (geomDet) {
	                geomDets.push_back(geomDet);
	                //cout << "get DT chamber " <<  DTDetId(endcap, station, ring, chamber, 0) << " " << geomDet << endl;
	            }
            }
                
	        if (geomDets.size()!=0) {
                muDetRings.push_back(new MuDetRing(geomDets));
	            LogDebug("xxx") << "New ring with " << geomDets.size() << " chambers";
	        }
        }
                
        result[endcap].push_back(new MuRingForwardLayer(muDetRings));    
	    LogDebug("xxx") << "New layer with " << muDetRings.size() << " rings" << endl;
        muDetRings.clear();
    
        for(int station = 2; station <= DTDetId::maxStationId(); station++) {

            for(int ring = DTDetId::minRingId(); ring <= DTDetId::maxRingId(); ring++) {
                
                vector<const GeomDet*> geomDets;
                for(int chamber = DTDetId::minChamberId(); chamber <= DTDetId::maxChamberId(); chamber++) {
		  
		            const GeomDet* geomDet = geo.idToDet(DTDetId(endcap, station, ring, chamber, 0));
		            if (geomDet) {
		                geomDets.push_back(geomDet);
		                LogDebug("xxx") << "get DT chamber " <<  DTDetId(endcap, station, ring, chamber, 0) << " " << geomDet;
		            }
                }
                
		        if (geomDets.size()!=0) {
                    muDetRings.push_back(new MuDetRing(geomDets));
		            LogDebug("xxx") << "New ring with " << geomDets.size() << " chambers";
		        }
            }
                
            result[endcap].push_back(new MuRingForwardLayer(muDetRings));    
	        LogDebug("xxx") << "New layer with " << muDetRings.size() << " rings";
            muDetRings.clear();
        }
        
        for(vector<MuRingForwardLayer*>::const_iterator it = result[endcap].begin(); it != result[endcap].end(); it++)
                detlayers[endcap].push_back((DetLayer*)(*it));
    }    
    */
    pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(detlayers[0], detlayers[1]); 
    return res_pair;
}
