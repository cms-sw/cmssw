#include "L1Trigger/L1TMuonCPPF/interface/RecHitProcessor.h"

RecHitProcessor::RecHitProcessor() {
}

RecHitProcessor::~RecHitProcessor() {
}

void RecHitProcessor::processLook(
				  const edm::Event& iEvent,
				  const edm::EventSetup& iSetup,
				  const edm::EDGetToken& recHitToken,
				  std::vector<RecHitProcessor::CppfItem>& CppfVec1,
				  l1t::CPPFDigiCollection& cppfDigis,
				  const int MaxClusterSize
				  ) const {
  
  edm::Handle<RPCRecHitCollection> recHits;
  iEvent.getByToken(recHitToken, recHits);
  
  edm::ESHandle<RPCGeometry> rpcGeom;
  iSetup.get<MuonGeometryRecord>().get(rpcGeom);
  
  // The loop is over the detector container in the rpc geometry collection. We are interested in the RPDdetID (inside of RPCChamber vectors), specifically, the RPCrechits. to assignment the CPPFDigis.
  for ( TrackingGeometry::DetContainer::const_iterator iDet = rpcGeom->dets().begin(); iDet < rpcGeom->dets().end(); iDet++ ) {
   
  //  we do a cast over the class RPCChamber to obtain the RPCroll vectors, inside of them, the RPCRechits are found. in other words, the method ->rolls() does not exist for other kind of vector within DetContainer and we can not obtain the rpcrechits in a suitable way. 
    if (dynamic_cast<const RPCChamber*>( *iDet ) == nullptr ) continue;
    
    auto chamb = dynamic_cast<const RPCChamber* >( *iDet ); 

    std::vector<const RPCRoll*> rolls = (chamb->rolls());
 
    // Loop over rolls in the chamber
    for(auto& iRoll : rolls){
      
      RPCDetId rpcId = (*iRoll).id();	
      
        
    
      typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
      rangeRecHits recHitCollection =  recHits->get(rpcId);
      
      
      
      //Loop over the RPC digis
      for (RPCRecHitCollection::const_iterator rechit_it = recHitCollection.first; rechit_it != recHitCollection.second; rechit_it++) {	
	
	//const RPCDetId& rpcId = rechit_it->rpcId();
	int rawId = rpcId.rawId();
	//int station = rpcId.station();
	int Bx = rechit_it->BunchX(); 
	int isValid = rechit_it->isValid();
	int firststrip = rechit_it->firstClusterStrip();
	int clustersize = rechit_it->clusterSize();
	LocalPoint lPos = rechit_it->localPosition();
	const RPCRoll* roll = rpcGeom->roll(rpcId);
	const BoundPlane& rollSurface = roll->surface();
	GlobalPoint gPos = rollSurface.toGlobal(lPos);
	float global_theta = emtf::rad_to_deg(gPos.theta().value());
	float global_phi   = emtf::rad_to_deg(gPos.phi().value());
	//::::::::::::::::::::::::::::
	//Establish the average position of the rechit    
	int rechitstrip = firststrip;
	
        if(clustersize > 2) {
	  int medium = 0;
	  if (clustersize % 2 == 0) medium = 0.5*(clustersize); 
	  else medium = 0.5*(clustersize-1);
	  rechitstrip += medium; 
	} 
	
	if(clustersize > MaxClusterSize) continue;	
	//This is just for test CPPFDigis with the RPC Geometry, It must be "true" in the normal runs 
	bool Geo = true;
	////:::::::::::::::::::::::::::::::::::::::::::::::::
	//Set the EMTF Sector 	
	int EMTFsector1 = 0;	
	int EMTFsector2 = 0;
	
	//sector 1
	if ((global_phi > 15.) && (global_phi <= 16.3)) {
	  EMTFsector1 = 1;
	  EMTFsector2 = 6;
	}
	else if ((global_phi > 16.3) && (global_phi <= 53.)) {
	  EMTFsector1 = 1;
	  EMTFsector2 = 0;
	}
	else if ((global_phi > 53.) && (global_phi <= 75.)) {
	  EMTFsector1 = 1;
	  EMTFsector2 = 2;
	}
	//sector 2 
	else if ((global_phi > 75.) && (global_phi <= 76.3)) {
	  EMTFsector1 = 1;
	  EMTFsector2 = 2;
	}
	else if ((global_phi > 76.3) && (global_phi <= 113.)) {
	  EMTFsector1 = 2;
	  EMTFsector2 = 0;
	}
	else if ((global_phi > 113.) && (global_phi <= 135.)) {
	  EMTFsector1 = 2;
	  EMTFsector2 = 3;
	}
	//sector 3
	//less than 180
	else if ((global_phi > 135.) && (global_phi <= 136.3)) {
	  EMTFsector1 = 2;
	  EMTFsector2 = 3;
	}
	else if ((global_phi > 136.3) && (global_phi <= 173.)) {
	  EMTFsector1 = 3;
	  EMTFsector2 = 0;
	}
	else if ((global_phi > 173.) && (global_phi <= 180.)) {
	  EMTFsector1 = 3;
	  EMTFsector2 = 4;
	}
	//Greater than -180
	else if ((global_phi < -165.) && (global_phi >= -180.)) {
	  EMTFsector1 = 3;
	  EMTFsector2 = 4;
	}
	//Fourth sector
	else if ((global_phi > -165.) && (global_phi <= -163.7)) {
	  EMTFsector1 = 3;
	  EMTFsector2 = 4;
	}
	else if ((global_phi > -163.7) && (global_phi <= -127.)) {
	  EMTFsector1 = 4;
	  EMTFsector2 = 0;
	}
	else if ((global_phi > -127.) && (global_phi <= -105.)) {
	  EMTFsector1 = 4;
	  EMTFsector2 = 5;
	}
	//fifth sector
	else if ((global_phi > -105.) && (global_phi <= -103.7)) {
	  EMTFsector1 = 4;
	  EMTFsector2 = 5;
	}
	else if ((global_phi > -103.7) && (global_phi <= -67.)) {
	  EMTFsector1 = 5;
	  EMTFsector2 = 0;
	}
	else if ((global_phi > -67.) && (global_phi <= -45.)) {
	  EMTFsector1 = 5;
	  EMTFsector2 = 6;
	} 
	//sixth sector
	else if ((global_phi > -45.) && (global_phi <= -43.7)) {
	  EMTFsector1 = 5;
	  EMTFsector2 = 6;
	} 
	else if ((global_phi > -43.7) && (global_phi <= -7.)) {
	  EMTFsector1 = 6;
	  EMTFsector2 = 0;
	}
	else if ((global_phi > -7.) && (global_phi <= 15.)) {
	  EMTFsector1 = 6;
	  EMTFsector2 = 1;
	} 
	
	
	// std::vector<RecHitProcessor::CppfItem>::iterator it;
	// for(it = CppfVec1.begin(); it != CppfVec1.end(); it++){
	//	if( (*it).rawId == rawId) if(Geo_true) std::cout << (*it).rawId << "rawid" << rawId << std::endl;
	//	}
	//Loop over the look up table    
	double EMTFLink1 = 0.;
	double EMTFLink2 = 0.;
	
        std::vector<RecHitProcessor::CppfItem>::iterator cppf1;
        std::vector<RecHitProcessor::CppfItem>::iterator cppf;
        for(cppf1 = CppfVec1.begin(); cppf1 != CppfVec1.end(); cppf1++){
	  
          
	  
	  //Condition to save the CPPFDigi
	  if(((*cppf1).rawId == rawId) && ((*cppf1).strip == rechitstrip)){
	    
	    int old_strip = (*cppf1).strip;
            int before = 0;
	    int after = 0;
	  
            if(cppf1 != CppfVec1.begin())	    
	    	before = (*(cppf1-2)).strip;
	    
	    else if (cppf1 == CppfVec1.begin())
		before = (*cppf1).strip;
		
            if(cppf1 != CppfVec1.end())
 	    	after = (*(cppf1+2)).strip;
		
            else if (cppf1 == CppfVec1.end())
		after = (*cppf1).strip;	

	    cppf = cppf1;
	    
	    if(clustersize == 2){
	      
	      if(firststrip == 1){
		if(before < after) cppf=(cppf1-1);
                else if (before > after) cppf=(cppf1+1); 
	      }
	      else if(firststrip > 1){
		if(before < after) cppf=(cppf1+1);
		else if (before > after) cppf=(cppf1-1);
	      }
	      
	    }
	    //Using the RPCGeometry	
	    if(Geo){
	      std::shared_ptr<l1t::CPPFDigi> MainVariables1(new l1t::CPPFDigi(rpcId, Bx , (*cppf).int_phi, (*cppf).int_theta, isValid, (*cppf).lb, (*cppf).halfchannel, EMTFsector1, EMTFLink1, old_strip, clustersize, global_phi, global_theta));
	      std::shared_ptr<l1t::CPPFDigi> MainVariables2(new l1t::CPPFDigi(rpcId, Bx , (*cppf).int_phi, (*cppf).int_theta, isValid, (*cppf).lb, (*cppf).halfchannel, EMTFsector2, EMTFLink2, old_strip, clustersize, global_phi, global_theta));

	      if ((EMTFsector1 > 0) && (EMTFsector2 == 0)){
		cppfDigis.push_back(*MainVariables1.get());
	      } 
	      else if ((EMTFsector1 > 0) && (EMTFsector2 > 0)){
		cppfDigis.push_back(*MainVariables1.get());
		cppfDigis.push_back(*MainVariables2.get());
	      }
	      else if ((EMTFsector1 == 0) && (EMTFsector2 == 0)) {
		continue; 
	      } 
	    } //Geo is true
	    else {
	      global_phi = 0.;
	      global_theta = 0.;
	      std::shared_ptr<l1t::CPPFDigi> MainVariables1(new l1t::CPPFDigi(rpcId, Bx , (*cppf).int_phi, (*cppf).int_theta, isValid, (*cppf).lb, (*cppf).halfchannel, EMTFsector1, EMTFLink1, old_strip, clustersize, global_phi, global_theta));
	      std::shared_ptr<l1t::CPPFDigi> MainVariables2(new l1t::CPPFDigi(rpcId, Bx , (*cppf).int_phi, (*cppf).int_theta, isValid, (*cppf).lb, (*cppf).halfchannel, EMTFsector2, EMTFLink2, old_strip, clustersize, global_phi, global_theta));
	      if ((EMTFsector1 > 0) && (EMTFsector2 == 0)){
		cppfDigis.push_back(*MainVariables1.get());
	      } 
	      else if ((EMTFsector1 > 0) && (EMTFsector2 > 0)){
		cppfDigis.push_back(*MainVariables1.get());
		cppfDigis.push_back(*MainVariables2.get());
	      }
	      else if ((EMTFsector1 == 0) && (EMTFsector2 == 0)) {
		continue;
	      } 
	    }
	  } //Condition to save the CPPFDigi 
	} //Loop over the LUTVector
      } //Loop over the recHits
    } // End loop: for (std::vector<const RPCRoll*>::const_iterator r = rolls.begin(); r != rolls.end(); ++r)
  } // End loop: for (TrackingGeometry::DetContainer::const_iterator iDet = rpcGeom->dets().begin(); iDet < rpcGeom->dets().end(); iDet++)  
  
}


void RecHitProcessor::process(
			      const edm::Event& iEvent,
			      const edm::EventSetup& iSetup,
			      const edm::EDGetToken& recHitToken,
			      l1t::CPPFDigiCollection& cppfDigis
			      ) const {
  
  // Get the RPC Geometry
  edm::ESHandle<RPCGeometry> rpcGeom;
  iSetup.get<MuonGeometryRecord>().get(rpcGeom);
  
  // Get the RecHits from the event
  edm::Handle<RPCRecHitCollection> recHits;
  iEvent.getByToken(recHitToken, recHits);
  
 
  // The loop is over the detector container in the rpc geometry collection. We are interested in the RPDdetID (inside of RPCChamber vectors), specifically, the RPCrechits. to assignment the CPPFDigis.
  for ( TrackingGeometry::DetContainer::const_iterator iDet = rpcGeom->dets().begin(); iDet < rpcGeom->dets().end(); iDet++ ) {
  
  //  we do a cast over the class RPCChamber to obtain the RPCroll vectors, inside of them, the RPCRechits are found. in other words, the method ->rolls() does not exist for other kind of vector within DetContainer and we can not obtain the rpcrechits in a suitable way.   
    if (dynamic_cast<const RPCChamber*>( *iDet ) == nullptr ) continue;
    
    auto chamb = dynamic_cast<const RPCChamber* >( *iDet ); 
    std::vector<const RPCRoll*> rolls = (chamb->rolls());
    
    // Loop over rolls in the chamber
    for(auto& iRoll : rolls){
      
      RPCDetId rpcId = (*iRoll).id();	
      
      typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
      rangeRecHits recHitCollection =  recHits->get(rpcId);
      
      
      for (RPCRecHitCollection::const_iterator rechit_it = recHitCollection.first; rechit_it != recHitCollection.second; rechit_it++) {	  
	
        //const RPCDetId& rpcId = rechit_it->rpcId();
        //int rawId = rpcId.rawId();
        int region = rpcId.region();
        //int station = rpcId.station();
        int Bx = rechit_it->BunchX(); 
        int isValid = rechit_it->isValid();
        int firststrip = rechit_it->firstClusterStrip();
        int clustersize = rechit_it->clusterSize();
        LocalPoint lPos = rechit_it->localPosition();
        const RPCRoll* roll = rpcGeom->roll(rpcId);
        const BoundPlane& rollSurface = roll->surface();
        GlobalPoint gPos = rollSurface.toGlobal(lPos);
        float global_theta = emtf::rad_to_deg(gPos.theta().value());
        float global_phi   = emtf::rad_to_deg(gPos.phi().value());
        //Endcap region only
	
        if (region != 0) {
	  
	  int int_theta = (region == -1 ? 180. * 32. / 36.5 : 0.)
	    + (float)region * global_theta * 32. / 36.5   
	    - 8.5 * 32 / 36.5;
	  
	  if(region == 1) {
	    if(global_theta < 8.5) int_theta = 0;
	    if(global_theta > 45.) int_theta = 31;
	  } 
	  else if(region == -1) {
	    if(global_theta < 135.) int_theta = 31;
	    if(global_theta > 171.5) int_theta = 0;
	  } 
	  
	  //Local EMTF
	  double local_phi = 0.;
	  int EMTFsector1 = 0;
	  int EMTFsector2 = 0;
	  
	  //sector 1
	  if ((global_phi > 15.) && (global_phi <= 16.3)) {
	    local_phi = global_phi-15.;
	    EMTFsector1 = 1;
	    EMTFsector2 = 6;
	  }
	  else if ((global_phi > 16.3) && (global_phi <= 53.)) {
	    local_phi = global_phi-15.;
	    EMTFsector1 = 1;
	    EMTFsector2 = 0;
	  }
	  else if ((global_phi > 53.) && (global_phi <= 75.)) {
	    local_phi = global_phi-15.;
	    EMTFsector1 = 1;
	    EMTFsector2 = 2;
	  }
	  //sector 2 
	  else if ((global_phi > 75.) && (global_phi <= 76.3)) {
	    local_phi = global_phi-15.;
	    EMTFsector1 = 1;
	    EMTFsector2 = 2;
	  }
	  else if ((global_phi > 76.3) && (global_phi <= 113.)) {
	    local_phi = global_phi-75.;
	    EMTFsector1 = 2;
	    EMTFsector2 = 0;
	  }
	  else if ((global_phi > 113.) && (global_phi <= 135.)) {
	    local_phi = global_phi-75.;
	    EMTFsector1 = 2;
	    EMTFsector2 = 3;
	  }
	  //sector 3
	  //less than 180
	  else if ((global_phi > 135.) && (global_phi <= 136.3)) {
	    local_phi = global_phi-75.;
	    EMTFsector1 = 2;
	    EMTFsector2 = 3;
	  }
	  else if ((global_phi > 136.3) && (global_phi <= 173.)) {
	    local_phi = global_phi-135.;
	    EMTFsector1 = 3;
	    EMTFsector2 = 0;
	  }
	  else if ((global_phi > 173.) && (global_phi <= 180.)) {
	    local_phi = global_phi-135.;
	    EMTFsector1 = 3;
	    EMTFsector2 = 4;
	  }
	  //Greater than -180
	  else if ((global_phi < -165.) && (global_phi >= -180.)) {
	    local_phi = global_phi+225.;
	    EMTFsector1 = 3;
	    EMTFsector2 = 4;
	  }
	  //Fourth sector
	  else if ((global_phi > -165.) && (global_phi <= -163.7)) {
	    local_phi = global_phi+225.;
	    EMTFsector1 = 3;
	    EMTFsector2 = 4;
	  }
	  else if ((global_phi > -163.7) && (global_phi <= -127.)) {
	    local_phi = global_phi+165.;
	    EMTFsector1 = 4;
	    EMTFsector2 = 0;
	  }
	  else if ((global_phi > -127.) && (global_phi <= -105.)) {
	    local_phi = global_phi+165.;
	    EMTFsector1 = 4;
	    EMTFsector2 = 5;
	  }
	  //fifth sector
	  else if ((global_phi > -105.) && (global_phi <= -103.7)) {
	    local_phi = global_phi+165.;
	    EMTFsector1 = 4;
	    EMTFsector2 = 5;
	  }
	  else if ((global_phi > -103.7) && (global_phi <= -67.)) {
	    local_phi = global_phi+105.;
	    EMTFsector1 = 5;
	    EMTFsector2 = 0;
	  }
	  else if ((global_phi > -67.) && (global_phi <= -45.)) {
	    local_phi = global_phi+105.;
	    EMTFsector1 = 5;
	    EMTFsector2 = 6;
	  } 
	  //sixth sector
	  else if ((global_phi > -45.) && (global_phi <= -43.7)) {
	    local_phi = global_phi+105.;
	    EMTFsector1 = 5;
	    EMTFsector2 = 6;
	  } 
	  else if ((global_phi > -43.7) && (global_phi <= -7.)) {
	    local_phi = global_phi+45.;
	    EMTFsector1 = 6;
	    EMTFsector2 = 0;
	  }
	  else if ((global_phi > -7.) && (global_phi <= 15.)) {
	    local_phi = global_phi+45.;
	    EMTFsector1 = 6;
	    EMTFsector2 = 1;
	  } 
	  
	  int int_phi = int((local_phi + 22.0 )*15. + .5); 
	  
	  double EMTFLink1 = 0.;
	  double EMTFLink2 = 0.;
          double lb = 0.;
	  double halfchannel = 0.;
	  
	  //Invalid hit
	  if (isValid == 0) int_phi = 2047;	 
	  //Right integers range
	  assert(0 <= int_phi && int_phi < 1250);
	  assert(0 <= int_theta && int_theta < 32);
	  
	  std::shared_ptr<l1t::CPPFDigi> MainVariables1(new l1t::CPPFDigi(rpcId, Bx , int_phi, int_theta, isValid, lb, halfchannel, EMTFsector1, EMTFLink1, firststrip, clustersize, global_phi, global_theta));
	  std::shared_ptr<l1t::CPPFDigi> MainVariables2(new l1t::CPPFDigi(rpcId, Bx , int_phi, int_theta, isValid, lb, halfchannel, EMTFsector2, EMTFLink2, firststrip, clustersize, global_phi, global_theta));
	  if(int_theta == 31) continue;
          if ((EMTFsector1 > 0) && (EMTFsector2 == 0)){
	    cppfDigis.push_back(*MainVariables1.get());
	  } 
          if ((EMTFsector1 > 0) && (EMTFsector2 > 0)){
	    cppfDigis.push_back(*MainVariables1.get());
	    cppfDigis.push_back(*MainVariables2.get());
	  }
	  if ((EMTFsector1 == 0) && (EMTFsector2 == 0)){
	    continue;
	  } 
        } // No barrel rechits	
	
      } // End loop: for (RPCRecHitCollection::const_iterator recHit = recHitCollection.first; recHit != recHitCollection.second; recHit++)
      
    } // End loop: for (std::vector<const RPCRoll*>::const_iterator r = rolls.begin(); r != rolls.end(); ++r)
  } // End loop: for (TrackingGeometry::DetContainer::const_iterator iDet = rpcGeom->dets().begin(); iDet < rpcGeom->dets().end(); iDet++)
} // End function: void RecHitProcessor::process()


  


