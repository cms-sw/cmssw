std::cout <<"\t Getting the CSC Geometry"<<std::endl;
edm::ESHandle<CSCGeometry> cscGeo;
iSetup.get<MuonGeometryRecord>().get(cscGeo);
    
std::cout <<"\t Getting the CSC Segments"<<std::endl;
edm::Handle<CSCSegmentCollection> allCSCSegments;
iEvent.getByLabel(cscSegments, allCSCSegments);

if(allCSCSegments->size()>0){
  std::cout<<"\t Number of CSC Segments in this event = "<<allCSCSegments->size()<<std::endl;
    
  std::map<CSCDetId,int> CSCSegmentsCounter;
  CSCSegmentCollection::const_iterator segment;
     
  for (segment = allCSCSegments->begin();segment!=allCSCSegments->end(); ++segment){
    CSCSegmentsCounter[segment->cscDetId()]++;
  }    
    
  std::cout<<"\t loop over all the CSCSegments "<<std::endl;    
  for (segment = allCSCSegments->begin();segment!=allCSCSegments->end(); ++segment){
              
    CSCDetId CSCId = segment->cscDetId();
    
    std::cout<<"\t \t This Segment is in Chamber id: "<<CSCId<<std::endl;
    std::cout<<"\t \t Number of segments in this CSC = "<<CSCSegmentsCounter[CSCId]<<std::endl;
    std::cout<<"\t \t Is the only one in this CSC? and is not ind the ring 1 or station 4?"<<std::endl;

    if(CSCSegmentsCounter[CSCId]==1 && CSCId.station()!=4 && CSCId.ring()!=1){
      std::cout<<"\t \t yes"<<std::endl;
     
      int cscEndCap = CSCId.endcap();
      int cscStation = CSCId.station();
      int cscRing = CSCId.ring();
      int cscChamber = CSCId.chamber();
      int rpcRegion = 1; if(cscEndCap==2) rpcRegion= -1;//Relacion entre las endcaps
      int rpcRing = cscRing;
      if(cscRing==4)rpcRing =1;
      int rpcStation = cscStation;
      int rpcSegment = 0;
	
      if(cscStation!=1&&cscRing==1){//las de 18 CSC
	rpcSegment = CSCId.chamber();
      }
      else{//las de 36 CSC
	rpcSegment = (CSCId.chamber()==1) ? 36 : CSCId.chamber()-1;
      }
     
      LocalPoint segmentPosition= segment->localPosition();
      LocalVector segmentDirection=segment->localDirection();
	
      
      if(segment->dimension()==4){
	std::cout<<"\t \t yes"<<std::endl;
	std::cout<<"\t \t CSC Segment Dimension "<<segment->dimension()<<std::endl; 
      
	float Xo=segmentPosition.x();
	float Yo=segmentPosition.y();
	float Zo=segmentPosition.z();
	float dx=segmentDirection.x();
	float dy=segmentDirection.y();
	float dz=segmentDirection.z();

	std::cout<<"\t \t Getting chamber from Geometry"<<std::endl;
	const CSCChamber* TheChamber=cscGeo->chamber(CSCId); 
	std::cout<<"\t \t Getting ID from Chamber"<<std::endl;
	const CSCDetId TheId=TheChamber->id();
	std::cout<<"\t \t Printing The Id"<<TheId<<std::endl;

	std::cout<<"\t \t Loop over all the rolls asociated to this CSC"<<std::endl;
	std::set<RPCDetId> rollsForThisCSC = rollstoreCSC[CSCStationIndex(rpcRegion,rpcStation,rpcRing,rpcSegment)];
	std::cout<<"\t \t Number of rolls for this CSC = "<<rollsForThisCSC.size()<<std::endl;
	
	
	if(rpcRing!=1&&rpcStation!=4){
	  std::cout<<"Fail for CSCId="<<TheId<<" rpcRegion="<<rpcRegion<<" rpcStation="<<rpcStation<<" rpcRing="<<rpcRing<<" rpcSegment="<<rpcSegment<<std::endl;
	  assert(rollsForThisCSC.size()>=1);
	
	  //Loop over all the rolls
	  for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisCSC.begin();iteraRoll != rollsForThisCSC.end(); iteraRoll++){
	    const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll);
	    RPCDetId rpcId = rollasociated->id();
	    std::cout<<"\t \t \t We are in the roll getting the surface"<<rpcId<<std::endl;
	    const BoundPlane & RPCSurface = rollasociated->surface(); 

	    std::cout<<"\t \t \t RollID: "<<rollasociated->id()<<std::endl;

	    std::cout<<"\t \t \t Doing the extrapolation to this roll"<<std::endl;
	    std::cout<<"\t \t \t CSC Segment Direction in CSCLocal "<<segmentDirection<<std::endl;
	    std::cout<<"\t \t \t CSC Segment Point in CSCLocal "<<segmentPosition<<std::endl;  
	    GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0,0,0));
	    std::cout<<"\t \t \t Center (0,0,0) of the Roll in Global"<<CenterPointRollGlobal<<std::endl;

	    GlobalPoint CenterPointCSCGlobal = TheChamber->toGlobal(LocalPoint(0,0,0));
	    std::cout<<"\t \t \t Center (0,0,0) of the CSC in Global"<<CenterPointCSCGlobal<<std::endl;
	    
	    GlobalPoint segmentPositionInGlobal=TheChamber->toGlobal(segmentPosition); //new way to convert to global
	    std::cout<<"\t \t \t Segment Position in Global"<<segmentPositionInGlobal<<std::endl;
	    
	    LocalPoint CenterRollinCSCFrame = TheChamber->toLocal(CenterPointRollGlobal);
	    std::cout<<"\t \t \t Center (0,0,0) Roll In CSCLocal"<<CenterRollinCSCFrame<<std::endl;
	    
	    float D=CenterRollinCSCFrame.z();
	  	  
	    float X=Xo+dx*D/dz;
	    float Y=Yo+dy*D/dz;
	    float Z=D;

	    const TrapezoidalStripTopology* top_=dynamic_cast<const TrapezoidalStripTopology*>(&(rollasociated->topology()));
	    LocalPoint xmin = top_->localPosition(0.);
	    std::cout<<"\t \t \t xmin of this  Roll "<<xmin<<"cm"<<std::endl;
	    LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
	    std::cout<<"\t \t \t xmax of this  Roll "<<xmax<<"cm"<<std::endl;
	    float rsize = fabs( xmax.x()-xmin.x() );
	    std::cout<<"\t \t \t Roll Size "<<rsize<<"cm"<<std::endl;
	    float stripl = top_->stripLength();
	    float stripw = top_->pitch();
	    std::cout<<"\t \t \t Strip Lenght "<<stripl<<"cm"<<std::endl;
	    std::cout<<"\t \t \t Strip Width "<<stripw<<"cm"<<std::endl;

	    std::cout<<"\t \t \t X Predicted in CSCLocal= "<<X<<"cm"<<std::endl;
	    std::cout<<"\t \t \t Y Predicted in CSCLocal= "<<Y<<"cm"<<std::endl;
	    std::cout<<"\t \t \t Z Predicted in CSCLocal= "<<Z<<"cm"<<std::endl;
	  
	    float extrapolatedDistance = sqrt((X-Xo)*(X-Xo)+(Y-Yo)*(Y-Yo)+(Z-Zo)*(Z-Zo));
	    std::cout<<"\t \t \t Is the distance of extrapolation less than MaxD? ="<<extrapolatedDistance<<"cm"<<"MaxD="<<MaxD<<"cm"<<std::endl;
	    if(extrapolatedDistance<=MaxD){ 
	      std::cout<<"\t \t \t yes"<<std::endl;
	    
	      GlobalPoint GlobalPointExtrapolated=TheChamber->toGlobal(LocalPoint(X,Y,Z));
	      std::cout<<"\t \t \t Point ExtraPolated in Global"<<GlobalPointExtrapolated<< std::endl;
	      
	      LocalPoint PointExtrapolatedRPCFrame = RPCSurface.toLocal(GlobalPointExtrapolated);
	      std::cout<<"\t \t \t Point Extrapolated in RPCLocal"<<PointExtrapolatedRPCFrame<< std::endl;
	      std::cout<<"\t \t \t Does the extrapolation go inside this roll????"<<std::endl;
	    	      
	      if(fabs(PointExtrapolatedRPCFrame.z()) < 0.01 && 
		 fabs(PointExtrapolatedRPCFrame.x()) < rsize*0.5 && 
		 fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){ 
		RPCDetId  rollId = rollasociated->id();
		std::cout<<"\t \t \t \t yes"<<std::endl;	
		const float stripPredicted = rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 

		std::cout<<"\t \t \t \t Candidate"<<rollId<<" "<<"(from CSC Segment) STRIP---> "<<stripPredicted<< std::endl;

		int stripDetected = 0;
	      
		std::cout<<"\t \t \t \t Getting digis asociated to this roll in this event"<<std::endl;
		RPCDigiCollection::Range rpcRangeDigi=rpcDigis->get(rollasociated->id());

		int stripCounter = 0;
		

		//--------- HISTOGRAM STRIP PREDICTED FROM CSC  -------------------

		RPCGeomServ rpcsrv(rollId);
		std::string nameRoll = rpcsrv.name();
		std::cout<<"\t \t \t \t The RPCName is "<<nameRoll<<std::endl;
		_idList.push_back(nameRoll);

		char detUnitLabel[128];
		sprintf(detUnitLabel ,"%s",nameRoll.c_str());
		sprintf(layerLabel ,"%s",nameRoll.c_str());
		
	      
		std::map<std::string, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(nameRoll);

		if (meItr == meCollection.end()){
		  meCollection[nameRoll] = bookDetUnitSeg(rollId);
		}
	      
		std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];

		sprintf(meIdCSC,"ExpectedOccupancyFromCSC_%s",detUnitLabel);
		meMap[meIdCSC]->Fill(stripPredicted);

		sprintf(meIdCSC,"ExpectedOccupancy2DFromCSC_%s",detUnitLabel);
		meMap[meIdCSC]->Fill(stripPredicted,Y);

		//--------------------------------------------------------------------
	    
		totalcounter[0]++;
		buff=counter[0];
		buff[rollasociated->id()]++;
		counter[0]=buff;
		
		bool anycoincidence = false;
		double sumStripDetected = 0.;

		std::cout<<"\t \t \t \t \t Loop over the digis in this roll looki ng for the Average"<<std::endl;
	      
		for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){
		  stripCounter++;
		  stripDetected=digiIt->strip(); 
		  sumStripDetected=sumStripDetected+stripDetected;
		  std::cout<<"\t \t \t \t \t \t Digi "<<*digiIt<<"\t Detected="<<stripDetected<<" Predicted="<<stripPredicted<<"\t SumStrip= "<<sumStripDetected<<std::endl;
		}

		std::cout<<"\t \t \t \t \t Sum of strips "<<sumStripDetected<<std::endl;

		double meanStripDetected=sumStripDetected/((double)stripCounter);

		std::cout<<"\t \t \t \t \t Number of strips "<<stripCounter<<" Strip Average Detected"<<meanStripDetected<<std::endl;

		LocalPoint meanstripDetectedLocalPoint = top_->localPosition((float)(meanStripDetected));
	      
		float meanrescms = PointExtrapolatedRPCFrame.x()-meanstripDetectedLocalPoint.x();          
		float meanrescmsY = PointExtrapolatedRPCFrame.y()-meanstripDetectedLocalPoint.y();

		std::cout<<"\t \t \t \t \t PointExtrapolatedRPCFrame.x="<<PointExtrapolatedRPCFrame.x()<<" meanstripDetectedLocalPoint.x="<<meanstripDetectedLocalPoint.x()<<std::endl;

		if(fabs(meanrescms) < MinimalResidual){

		  std::cout<<"\t \t \t \t \t MeanRes="<<meanrescms<<"cm  MinimalResidual="<<MinimalResidual<<"cm"<<std::endl;

		  //----GLOBAL HISTOGRAM----
		  std::cout<<"\t \t \t \t \t Filling the Global Histogram with= "<<meanrescms<<std::endl;
		  //if(rollId.layer()==1&&rollId.station()==1&&rollId.ring()==0) 
		  hGlobalRes->Fill(meanrescms+0.5*stripw);
		  //if(rollId.layer()==1&&rollId.station()==1&&rollId.ring()==0&&stripCounter==2) hGlobalResClu1->Fill(meanrescms+0.5*stripw);
		  //if(rollId.layer()==1&&rollId.station()==1&&rollId.ring()==0&&stripCounter==4) hGlobalResClu2->Fill(meanrescms+0.5*stripw);
		  //if(rollId.layer()==1&&rollId.station()==1&&rollId.ring()==0&&stripCounter==6) hGlobalResClu3->Fill(meanrescms+0.5*stripw);
		  //if(rollId.layer()==1&&rollId.station()==1&&rollId.ring()==0&&stripCounter==8) hGlobalResClu4->Fill(meanrescms+0.5*stripw);
		  hGlobalResY->Fill(meanrescmsY);
		  //------------------------


		  sprintf(meIdRPC,"RPCResidualsFromCSC_%s",detUnitLabel);
		  meMap[meIdRPC]->Fill(meanrescms);
		
		  sprintf(meIdRPC,"RPCResiduals2DFromCSC_%s",detUnitLabel);
		  meMap[meIdRPC]->Fill(meanrescms,Y);
		
		  
		  std::cout <<"\t \t \t \t \t \t COINCEDENCE Predict "<<stripPredicted<<" Detect "<<stripDetected<<std::endl;
		  anycoincidence=true;
		  std::cout <<"\t \t \t \t \t Increassing CSC counter"<<std::endl;
		  totalcounter[1]++;
		  buff=counter[1];
		  buff[rollId]++;
		  counter[1]=buff;
		
		  sprintf(meRPC,"RealDetectedOccupancyFromCSC_%s",detUnitLabel);
		  meMap[meRPC]->Fill(meanStripDetected);
		
		  sprintf(meRPC,"RPCDataOccupancyFromCSC_%s",detUnitLabel);
		  meMap[meRPC]->Fill(stripPredicted);
		
		  sprintf(meRPC,"RPCDataOccupancy2DFromCSC_%s",detUnitLabel);
		  meMap[meRPC]->Fill(stripPredicted,Y);
		}
	      
		if(anycoincidence==false) {
		  std::cout <<"\t \t \t \t \t \t THIS PREDICTION DOESN'T MATCH WITH RPC DATA"<<std::endl;
		  totalcounter[2]++;
		  buff=counter[2];
		  buff[rollasociated->id()]++;
		  counter[2]=buff;
		  std::cout << "\t \t \t \t \t One for counterFAIL"<<std::endl;
		
		  ofrej<<"CSCS EndCap "<<rpcRegion
		       <<"\t cscStation "<<cscStation
		       <<"\t cscRing "<<cscRing			   
		       <<"\t cscChamber "<<cscChamber
		       <<"\t Roll "<<rollasociated->id()
		       <<"\t Event "<<iEvent.id().event()
		       <<"\t CSCId "<<CSCId
		       <<"\t CSCId.ring  "<<CSCId.ring()
		       <<" \t cscRing "<<cscRing
		       <<std::endl;
		}
	      }else {
		std::cout<<"\t \t \t \t No the prediction is outside of this roll"<<std::endl;
	      }//Condition for the right match
	    }else{//if extrapolation distance D is not too long
	      std::cout<<"\t \t \t No, Exrtrapolation too long!, canceled"<<std::endl;
	    }//D so big
	  }//loop over the rolls asociated 
	}//Condition over the startup geometry!!!!
      }//Is the segment 4D?
    }else{
      std::cout<<"\t \t More than one segment in this chamber, or we are in Station Ring 1 or in Station 4"<<std::endl;
    }
  }
}
else{
  std::cout<<"This Event doesn't have any CSCSegment"<<std::endl;
}
