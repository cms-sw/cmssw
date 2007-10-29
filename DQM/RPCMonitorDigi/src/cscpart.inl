std::cout <<"\t Getting the CSC Geometry"<<std::endl;
edm::ESHandle<CSCGeometry> cscGeo;
iSetup.get<MuonGeometryRecord>().get(cscGeo);
    
std::cout <<"\t Getting the CSC Segments"<<std::endl;
edm::Handle<CSCSegmentCollection> allCSCSegments;
iEvent.getByLabel(cscSegments, allCSCSegments);


if(allCSCSegments->size()>0){
  
  std::cout<<"\t Here we go we have a Segment in the CSCs"<<std::endl;
  std::cout<<"\t Number of Segments in this event = "<<allCSCSegments->size()<<std::endl;
    
  std::map<CSCDetId,int> CSCSegmentsCounter;
  CSCSegmentCollection::const_iterator segment;
    

  //loop over all the CSCSegments to count how many segments per chamber do we have
   
  for (segment = allCSCSegments->begin();segment!=allCSCSegments->end(); ++segment){
    CSCSegmentsCounter[segment->cscDetId()]++;
  }    
    
  std::cout<<"\t loop over all the CSCSegments to extrapolate"<<std::endl;    
  for (segment = allCSCSegments->begin();segment!=allCSCSegments->end(); ++segment){
    
    if(segment->dimension()==4){

      CSCDetId CSCId = segment->cscDetId();
      std::cout<<"\t \t Number of Segments in"<<CSCId<<" are "<<CSCSegmentsCounter[CSCId]<<std::endl;
      
      if(CSCSegmentsCounter[CSCId]==1){
	int cscEndCap = CSCId.endcap();
	int cscStation = CSCId.station();
	int cscRing = CSCId.ring();
	int cscChamber = CSCId.chamber();

	int rpcRegion = 1; if(cscEndCap==2) rpcRegion= -1;//Relacion entre las endcaps
	int rpcRing = cscRing;
	if(rpcRing==4)rpcRing =1;
	int rpcStation = cscStation;
	int rpcSector=0;
	int rpcSubsector=0;

	if(cscRing==1&&cscStation!=1){//las de 18
	  rpcSector = (cscChamber-1)/3+1;
	  rpcSubsector = cscChamber-3*rpcSector+3; 
	  if(rpcSubsector==1)rpcSubsector=2;else if(rpcSubsector==2)rpcSubsector=1;//intercambia 2x1 en las de 18
	
	}else{//las de 36
	  if(cscRing==2&&cscStation==1){
	    cscChamber=cscChamber+1;
	  }
	
	  rpcSector = (cscChamber-2)/6+1;
	  rpcSubsector = cscChamber-6*rpcSector+5;
	
	  if(cscChamber==1) {
	    rpcSector = 6; 
	    rpcSubsector=6;
	  }
	
	  if(cscRing==2&&cscStation==1){
	    cscChamber=cscChamber-1;
	  }


	}

	LocalPoint segmentPosition= segment->localPosition();
	LocalVector segmentDirection=segment->localDirection();
	
	std::cout<<"\t \t We have one segment in this CSC "<<CSCId<<std::endl;
	std::cout<<"\t \t Its direction and postition is"<<segmentDirection<<""<<segmentPosition<<std::endl;
	
	float Xo=segmentPosition.x();
	float Yo=segmentPosition.y();
	float dx=segmentDirection.x();
	float dy=segmentDirection.y();
	float dz=segmentDirection.z();

	std::cout<<"\t \t Getting chamber from Geometry"<<std::endl;
	const CSCChamber* TheChamber=cscGeo->chamber(CSCId); 
	std::cout<<"\t \t Getting ID from Chamber"<<std::endl;
	const CSCDetId TheId=TheChamber->id();
	std::cout<<"\t \t Printing The Id"<<TheId<<std::endl;
	
	std::cout<<"\t \t Loop over all the rolls asociated to this CSC"<<std::endl;
	
	std::set<RPCDetId> rollsForThisCSC = rollstoreCSC[CSCStationIndex(rpcRegion,rpcStation,rpcRing,rpcSector,rpcSubsector)];

	for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisCSC.begin();iteraRoll != rollsForThisCSC.end(); iteraRoll++){
	  const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll);
	  RPCDetId rpcId = rollasociated->id();
	  std::cout<<"\t \t \t We are in the roll getting the surface"<<rpcId<<std::endl;
	  const BoundPlane & RPCSurface = rollasociated->surface(); 

	  GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0,0,0));
	  std::cout<<"\t \t \t Center (0,0,0) of the Roll in Global"<<CenterPointRollGlobal<<std::endl;

	  GlobalPoint CenterPointCSCGlobal = TheChamber->toGlobal(LocalPoint(0,0,0));
	  std::cout<<"\t \t \t Center (0,0,0) of the CSC in Global"<<CenterPointCSCGlobal<<std::endl;
	    
	  float d=sqrt(
		       pow(CenterPointRollGlobal.x()-CenterPointCSCGlobal.x(),2)+
		       pow(CenterPointRollGlobal.y()-CenterPointCSCGlobal.y(),2)+
		       pow(CenterPointRollGlobal.z()-CenterPointCSCGlobal.z(),2)
		       );

	  std::cout<<"\t \t \t d="<<d<<"cm"<<std::endl;
	  
	  if(d>70.0)
	    {
	      std::cout<<"\t \t \t Extrapolation too long, Roll Discarted!!"<<std::endl;
	    }

	  else{
	    std::cout<<"\t \t \t Doing the extrapolation"<<std::endl;
	    std::cout<<"\t \t \t CSC Segment Direction in CSCLocal "<<segmentDirection<<std::endl;
	    std::cout<<"\t \t \t Segment Point in CSCLocal "<<segmentPosition<<std::endl;
	  
	    GlobalPoint segmentPositionInGlobal=TheChamber->toGlobal(segmentPosition); //new way to convert to global
	    std::cout<<"\t \t \t Segment Position in Global"<<segmentPositionInGlobal<<std::endl;
	    
	    LocalPoint CenterRollinCSCFrame = TheChamber->toLocal(CenterPointRollGlobal);
	    std::cout<<"\t \t \t Center (0,0,0) Roll In CSCLocal"<<CenterRollinCSCFrame<<std::endl;
	    
	    float D=CenterRollinCSCFrame.z();
	  
	    std::cout<<"\t \t \t Is D less than MaxD? D="<<D<<"cm"<<std::endl;
	  
	    if(fabs(D)<MaxD){
	      std::cout<<"\t \t \t yes"<<std::endl;
	      float X=Xo+dx*D/dz;
	      float Y=Yo+dy*D/dz;
	      float Z=D;
	    
	      std::cout<<"\t \t \t X Predicted in CSCLocal= "<<X<<"cm"<<std::endl;
	      std::cout<<"\t \t \t Y Predicted in CSCLocal= "<<Y<<"cm"<<std::endl;
	      std::cout<<"\t \t \t Z Predicted in CSCLocal= "<<Z<<"cm"<<std::endl;
	    
	      const TrapezoidalStripTopology* top_=dynamic_cast<const TrapezoidalStripTopology*>(&(rollasociated->topology()));
	      LocalPoint xmin = top_->localPosition(0.);
	      LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
	      float rsize = fabs( xmax.x()-xmin.x() )*0.5;
	      float stripl = top_->stripLength();
	      
	      GlobalPoint GlobalPointExtrapolated=TheChamber->toGlobal(LocalPoint(X,Y,Z));
	      std::cout<<"\t \t \t Point ExtraPolated in Global"<<GlobalPointExtrapolated<< std::endl;
	      
	      LocalPoint PointExtrapolatedRPCFrame = RPCSurface.toLocal(GlobalPointExtrapolated);
	      std::cout<<"\t \t \t Point Extrapolated in RPCLocal"<<PointExtrapolatedRPCFrame<< std::endl;
	      std::cout<<"\t \t \t Does the extrapolation go inside this roll????"<<std::endl;
	      //conditions to find the right roll to extrapolate
	      
	      if(fabs(PointExtrapolatedRPCFrame.z()) < 0.01 && fabs(PointExtrapolatedRPCFrame.x()) < rsize && fabs(PointExtrapolatedRPCFrame.y()) < 
		 stripl*0.5){ 
	        
		std::cout<<"\t \t \t \t yes"<<std::endl;	
		const float stripPredicted = rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
		
		std::cout<<"\t \t \t \t It is "<<stripPredicted<<std::endl;	
		std::cout<<"\t \t \t \t Getting digis asociated to this roll in this event"<<std::endl;
		RPCDigiCollection::Range rpcRangeDigi=rpcDigis->get(rollasociated->id());
		std::cout<<"\t \t \t \t Loop over the digis in this roll"<<std::endl;
		
		bool anycoincidence = false;


		//--------- HISTOGRAM STRIP PREDICTED FROM DT  -------------------

		RPCDetId  rollId = rollasociated->id();
		uint32_t id = rollId.rawId();
		_idList.push_back(id);
	      
		char detUnitLabel[128];
		sprintf(detUnitLabel ,"%d",id);
		sprintf(layerLabel ,"layer%d_subsector%d_roll%d",rollId.layer(),rollId.subsector(),rollId.roll());
	      
		std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(id);
		if (meItr == meCollection.end()){
		  meCollection[id] = bookDetUnitSeg(rollId);
		}
	      
		std::map<std::string, MonitorElement*> meMap=meCollection[id];

		sprintf(meIdCSC,"ExpectedOccupancyFromCSC_%s",detUnitLabel);
		meMap[meIdCSC]->Fill(stripPredicted);

		sprintf(meIdCSC,"ExpectedOccupancy2DFromCSC_%s",detUnitLabel);
		meMap[meIdCSC]->Fill(stripPredicted,Y);


		//--------------------------------------------------------------------
	    
		totalcounter[0]++;
		buff=counter[0];
		buff[rollasociated->id()]++;
		counter[0]=buff;
		

		std::cout<<"\t \t \t \t \t Loop over the digis in this roll "<<std::endl;
	
		for(RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){
		  std::cout<<"\t \t \t \t \t Digi "<<*digiIt<<std::endl;
		  
		  int stripDetected=digiIt->strip();
		  
		  float res = fabs((float)(stripDetected) - stripPredicted);
		  std::cout<<"\t \t \t \t \t Diference "<<res<<std::endl;


		  //---------------Fillin the histograms ------------------	
		  sprintf(meRPC,"RPCResidualsFromCSC_%s",detUnitLabel);
		  meMap[meRPC]->Fill(res);
		
		  sprintf(meRPC,"RPCResiduals2DFromCSC_%s",detUnitLabel);
		  meMap[meRPC]->Fill(res,Y);

		  //-------------------------------------------------------

		  
		  if(res < widestrip){
		    anycoincidence=true;
		    std::cout <<"\t \t \t \t \t \t COINCEDENCE Predict "<<stripPredicted<<" Detect "<<stripDetected<<std::endl;
		    totalcounter[1]++;
		    buff=counter[1];
		    buff[rollasociated->id()]++;
		    counter[1]=buff;
		  
		    //-----------------Filling the histograms --------------
		    sprintf(meRPC,"RealDetectedOccupancyFromCSC_%s",detUnitLabel);
		    meMap[meRPC]->Fill(stripDetected);
		  
		    sprintf(meRPC,"RPCDataOccupancyFromCSC_%s",detUnitLabel);
		    meMap[meRPC]->Fill(stripPredicted);

		    sprintf(meRPC,"RPCDataOccupancy2DFromCSC_%s",detUnitLabel);
		    meMap[meRPC]->Fill(stripPredicted,Y);
		    //-------------------------------------------------------

		    break;
		  }
		}
	      
		if(anycoincidence==false) {
		  std::cout <<"\t \t \t \t \t \t THIS PREDICTION DOESN'T MATCH WITH RPC DATA"<<std::endl;
		  totalcounter[2]++;
		  buff=counter[2];
		  buff[rollasociated->id()]++;
		  counter[2]=buff;		

		  ofrej<<"cscEndCap "<<rpcRegion
		       <<"\t cscStation "<<cscStation
		       <<"\t cscRing "<<cscRing			   
		       <<"\t cscChamber "<<cscChamber
		       <<"\t Roll "<<rollasociated->id()
		       <<"\t Event "
		       <<iEvent.id().event()
		       <<std::endl;
		}
	      
	      }
	      else {
		std::cout<<"\t \t \t \t no"<<std::endl;
	      }//Condition for the right match
	    }//if extrapolation distance D is not too long
	  }//if the distance between detectors is not too long
	}//loop over the CSCStationIndex
      }//if we have just one segment
    }//Is the segment 4D?
  }// loop over all the CSC segments
}// Is there any CSC segment in the event?
else{
  std::cout<<"This Event doesn't have any CSCSegment"<<std::endl;
}
