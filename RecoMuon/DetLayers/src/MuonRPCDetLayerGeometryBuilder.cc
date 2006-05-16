#include <RecoMuon/DetLayers/src/MuonRPCDetLayerGeometryBuilder.h>

#include <RecoMuon/DetLayers/interface/MuDetRing.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

MuonRPCDetLayerGeometryBuilder::MuonRPCDetLayerGeometryBuilder() {
}

MuonRPCDetLayerGeometryBuilder::~MuonRPCDetLayerGeometryBuilder() {
}

//pair<vector<MuRingForwardLayer*>, vector<MuRingForwardLayer*> > 
pair<vector<DetLayer*>, vector<DetLayer*> > 
    MuonRPCDetLayerGeometryBuilder::buildLayers(const RPCGeometry& geo) {
        
  //    vector<MuRingForwardLayer*> result[2];
    vector<DetLayer*> detlayers[2];
    vector<const ForwardDetRing*> muDetRings;
  /*          
    for(int endcap = RPCDetId::minEndcapId(); endcap != RPCDetId::maxEndcapId(); endcap++) {

        for(int ring = 2; ring <= 3; ring++) {
                
            vector<const GeomDet*> geomDets;
            for(int chamber = RPCDetId::minChamberId(); chamber <= RPCDetId::maxChamberId(); chamber++) {
		  
                const GeomDet* geomDet = geo.idToDet(RPCDetId(endcap, 1, ring, chamber, 0));
	            if (geomDet) {
		            geomDets.push_back(geomDet);
		            LogDebug("xxx") << "get RPC chamber " <<  RPCDetId(endcap, 1, ring, chamber, 0) << " " << geomDet;
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
            for(int chamber = RPCDetId::minChamberId(); chamber <= RPCDetId::maxChamberId(); chamber++) {
		  
                const GeomDet* geomDet = geo.idToDet(RPCDetId(endcap, 1, ring, chamber, 0));
	            if (geomDet) {
	                geomDets.push_back(geomDet);
	                //cout << "get RPC chamber " <<  RPCDetId(endcap, station, ring, chamber, 0) << " " << geomDet << endl;
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
    
        for(int station = 2; station <= RPCDetId::maxStationId(); station++) {

            for(int ring = RPCDetId::minRingId(); ring <= RPCDetId::maxRingId(); ring++) {
                
                vector<const GeomDet*> geomDets;
                for(int chamber = RPCDetId::minChamberId(); chamber <= RPCDetId::maxChamberId(); chamber++) {
		  
		            const GeomDet* geomDet = geo.idToDet(RPCDetId(endcap, station, ring, chamber, 0));
		            if (geomDet) {
		                geomDets.push_back(geomDet);
		                LogDebug("xxx") << "get RPC chamber " <<  RPCDetId(endcap, station, ring, chamber, 0) << " " << geomDet;
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
