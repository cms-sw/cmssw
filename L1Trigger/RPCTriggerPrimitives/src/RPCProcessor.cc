#include "L1Trigger/RPCTriggerPrimitives/interface/RPCProcessor.h"

RPCProcessor::RPCProcessor(){
}

RPCProcessor::~RPCProcessor(){
}

void RPCProcessor::Process(const edm::Event& iEvent,
			   const edm::EventSetup& iSetup,
			   const edm::EDGetToken& RPCDigiToken,
			   RPCRecHitCollection& primitivedigi,
			   std::unique_ptr<RPCMaskedStrips>& theRPCMaskedStripsObj,
                           std::unique_ptr<RPCDeadStrips>& theRPCDeadStripsObj,
			   std::unique_ptr<RPCRecHitBaseAlgo>& theAlgo,
                           std::map<std::string, std::string> LBName_ChamberID_Map_1, 
	                   std::map<std::string, std::string> LBID_ChamberID_Map_1, 
            		   std::map<std::string, std::string> LBName_ChamberID_Map_2, 
          		   std::map<std::string, std::string> LBID_ChamberID_Map_2,
                           bool ApplyLinkBoardCut_,             
		           int LinkboardCut,
               		   int ClusterSizeCut ) const{
  
  
  // Get the RPC Geometry
  edm::ESHandle<RPCGeometry> rpcGeom;
  iSetup.get<MuonGeometryRecord>().get(rpcGeom);
  // Get the RPC Digis
  edm::Handle<RPCDigiCollection> rpcdigis;
  iEvent.getByToken(RPCDigiToken, rpcdigis);
  // Pass the EventSetup to the algo
  theAlgo->setES(iSetup); 
  
  
  std::map<std::string, int> FirstLinkBarrel;
  std::map<std::string, int> FirstLinkEndCap;
  FirstLinkBarrel.clear();
  FirstLinkEndCap.clear();
  
  bool PassLinkCutBarrel=true;
  bool PassLinkCutEndCap=true;
  
  
  for ( auto rpcdgIt = rpcdigis->begin(); rpcdgIt != rpcdigis->end(); ++rpcdgIt ) {
    
    // The layerId
    const RPCDetId& rpcId = (*rpcdgIt).first;
    
    // Get the GeomDet from the setup
    const RPCRoll* roll = rpcGeom->roll(rpcId);
    if (roll == nullptr){
      edm::LogError("BadDigiInput")<<"Failed to find RPCRoll for ID "<<rpcId;
      continue;
    }
    
    
    // Get the iterators over the digis associated with this LayerId
    const RPCDigiCollection::Range& range = (*rpcdgIt).second;
    
    RollMask mask;
    const int rawId = rpcId.rawId();
    
    for ( const auto& tomask : theRPCMaskedStripsObj->MaskVec ) {
      if ( tomask.rawId == rawId ) {
        const int bit = tomask.strip;
        mask.set(bit-1);
      }
    }
    
    for ( const auto& tomask : theRPCDeadStripsObj->DeadVec ) {
      if ( tomask.rawId == rawId ) {
        const int bit = tomask.strip;
        mask.set(bit-1);
      }
    }
    
    
    // Call the reconstruction algorithm    
    edm::OwnVector<RPCRecHit> recHits = theAlgo->reconstruct(*roll, rpcId, range, mask);
    
    // LocalError tmpErr;
    // LocalPoint point; 
    // auto digi_pointer = std::make_shared<RPCRecHit>(  RPCRecHit(rpcId, 0, 0, 0, point, tmpErr) );
    // recHits.push_back(*digi_pointer.get());
    
    // Apply extra Cuts section.
    
    
    //Final rechit vector
    edm::OwnVector<RPCRecHit> recHit_output; 
    
    
    //Loop over the recHit vector
    for(auto &own : recHits){
      
      const RPCDetId& rpcId_ = own.rpcId();          
      const int region_ = rpcId_.region();
      const int ring_ = rpcId_.ring();
      const int station_ = rpcId_.station();
      const int sector_ = rpcId_.sector();
      const int subsector_ = rpcId_.subsector();
      const int layer_ = rpcId_.layer();
      const int roll_eta_ = rpcId_.roll();
      
      //Apply linkboard cut   
      if(ApplyLinkBoardCut_==true){
	
	std::string StringBarrel="";
	std::string StringEndCap="";
	
	std::string LBNameEndCap="";
	std::string LBNameBarrel="";
	
	/// Region id: 0 for Barrel, +/-1 For +/- Endcap
	if(region_ == 0){
	  StringBarrel=GetStringBarrel(ring_, station_, sector_, layer_, subsector_, roll_eta_);
	  std::string namemap1 = LBName_ChamberID_Map_1[StringBarrel]; 
	  std::string namemap2 = LBName_ChamberID_Map_2[StringBarrel]; 
	  LBNameBarrel=namemap1+namemap2;
	  
	} else{ 
	  //ChamberID only for EndCap region
	  int nsub = 6;
	  (ring_ == 1 && station_ > 1) ? nsub = 3 : nsub = 6;
	  const int chamberID = subsector_ + nsub * ( sector_ - 1);
	  
	  StringEndCap=GetStringEndCap(station_, ring_, chamberID);
	  //Getting linkboard name from map
	  LBNameEndCap=LBName_ChamberID_Map_1[StringEndCap];     
	}
	
	
	//maximum two cluster per linkboard     
	if(region_==0){//For Barrel
	  
	  std::map<std::string, int>::iterator it;
	  it=FirstLinkBarrel.find(LBNameBarrel);
	  
	  int repetitions_Barrel=0;
	  if(it==FirstLinkBarrel.end()) {
	    FirstLinkBarrel[LBNameBarrel]=1;
	  }
	  else if (it != FirstLinkBarrel.end()){
	    repetitions_Barrel=FirstLinkBarrel[LBNameBarrel];
	    FirstLinkBarrel[LBNameBarrel]=repetitions_Barrel+1;
	  }
	  
	  PassLinkCutBarrel=ApplyLinkBoardCut(FirstLinkBarrel[LBNameBarrel],LinkboardCut);
	  if(PassLinkCutBarrel) recHit_output.push_back(own);
	  
	} 
	
	else{ //For endcap
	  
	  std::map<std::string, int>::iterator it; 
	  it=FirstLinkEndCap.find(LBNameEndCap);
	  
	  int repetitions_EndCap=0;
	  if(it==FirstLinkEndCap.end()) {
	    FirstLinkEndCap[LBNameEndCap]=1;
	  }
	  else if (it != FirstLinkEndCap.end()){
	    repetitions_EndCap=FirstLinkEndCap[LBNameEndCap];
	    FirstLinkEndCap[LBNameEndCap]=repetitions_EndCap+1;
	  }
	  PassLinkCutEndCap=ApplyLinkBoardCut(FirstLinkEndCap[LBNameEndCap],LinkboardCut);
	  if(PassLinkCutEndCap) recHit_output.push_back(own); 
	  
	} 
	
      } else recHit_output = recHits;
      
    } //loop over temporal recHit vector
    
    
    if (recHit_output.size() != 0){// Just to make sure
      // clustersize cut: 
      recHit_output=ApplyClusterSizeCut(recHit_output, ClusterSizeCut); 
      primitivedigi.put(rpcId, recHit_output.begin(), recHit_output.end());
    }
    
  } // end for loop in RpcDigis
}



edm::OwnVector<RPCRecHit> RPCProcessor::ApplyClusterSizeCut(const edm::OwnVector<RPCRecHit> recHits_, int ClusterSizeCut_){
  
  edm::OwnVector<RPCRecHit> final_;
  
  for(auto &own : recHits_){
    bool passcut=true; 
    if(own.clusterSize() > ClusterSizeCut_){
      passcut = false;
    }
    if(passcut) final_.push_back(own);
  } 
  
  return final_;
}

bool RPCProcessor::ApplyLinkBoardCut(int NClusters, int LinkboardCut){
  
  bool passCutsize = true;
  if(NClusters > LinkboardCut) passCutsize = false; 
  
  return passCutsize;
  
}


std::string RPCProcessor::GetStringBarrel(const int ring_, const int station_, const int sector_, const int layer_, const int subsector_, const int roll_) const {
  
  std::string point="";
  std::map<int, std::string> Wheel;
  Wheel[-2]="W-2/";
  Wheel[-1]="W-1/";
  Wheel[0]="W0/";
  Wheel[1]="W+1/";
  Wheel[2]="W+2/";
  
  point += Wheel[ring_];
  
  std::map<int, std::string> Station;
  Station[1]="RB1";
  Station[2]="RB2";
  Station[3]="RB3";
  Station[4]="RB4";
  
  
  point += Station[station_];
  
  std::map<int, std::string> Layer;
  Layer[1]="in";
  Layer[2]="out";
  //layer 1 is the inner chamber and layer 2 is the outer chamber
  if(station_ == 1 || station_ == 2) point += Layer[layer_];
  
  point += "/";
  
  // Including the sector
  point+= std::to_string(sector_);
  
  //Taken from CondFormats/RPCObjects/src/ChamberLocationSpec.cc
  
  std::map<int, std::string> SubsecFour;
  SubsecFour[1]="--";
  SubsecFour[2]="-";
  SubsecFour[3]="+";
  SubsecFour[4]="++";
  
  std::map<int, std::string> SubsecTwo;
  SubsecTwo[1]="-";
  SubsecTwo[2]="+";
  
  if(station_ == 3 || station_ == 4){
    
    if(station_ == 4 && sector_ == 4){
      point+=SubsecFour[subsector_];
    }
    
    else if ((station_== 4) && (sector_ == 9 || sector_ == 11)){
      point+="";
    }
    
    else point+=SubsecTwo[subsector_];
    
  }
  
  
  //Taken from DataFormats/MuonDetId/src/RPCDetId.cc
  
  std::map<int, std::string> ROLL;
  ROLL[1]="Backward";
  ROLL[2]="Central";
  ROLL[3]="Forward";
  
  point+=ROLL[roll_];  
  
  return point;
}

std::string RPCProcessor::GetStringEndCap(const int station_, const int ring_, const int chamberID_) const {
  
  std::string point="";
  //Including station number
  std::map<int, std::string> Station;
  Station[-4]="RE-4";
  Station[-3]="RE-3";
  Station[-2]="RE-2";
  Station[-1]="RE-1";
  Station[1]="RE+1";
  Station[2]="RE+2";
  Station[3]="RE+3";
  Station[4]="RE+4";
  
  point+=Station[station_];
  
  //Including ring number
  point+="/"+std::to_string(ring_);
  //Including chamberID number
  point+="/"+std::to_string(chamberID_);
  
  return point;
}


