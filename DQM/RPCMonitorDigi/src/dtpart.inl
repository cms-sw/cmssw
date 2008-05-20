std::cout <<"\t Getting the DT Geometry"<<std::endl;
edm::ESHandle<DTGeometry> dtGeo;
iSetup.get<MuonGeometryRecord>().get(dtGeo);
    
std::cout <<"\t Getting the DT Segments"<<std::endl;
edm::Handle<DTRecSegment4DCollection> all4DSegments;
iEvent.getByLabel(dt4DSegments, all4DSegments);

if(all4DSegments->size()>0){
  std::cout<<"\t Number of Segments in this event = "<<all4DSegments->size()<<std::endl;
  
  std::map<DTChamberId,int> scounter;
  DTRecSegment4DCollection::const_iterator segment;  
  
  for (segment = all4DSegments->begin();segment!=all4DSegments->end(); ++segment){
    scounter[segment->chamberId()]++;
  }    
  
  std::cout<<"\t Loop over all the 4D Segments"<<std::endl;
  for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ 
    
    DTChamberId DTId = segment->chamberId();
    
    std::cout<<"\t \t This Segment is in Chamber id: "<<DTId<<std::endl;
    std::cout<<"\t \t Number of segments in this DT = "<<scounter[DTId]<<std::endl;
    std::cout<<"\t \t Is the only one in this DT?"<<std::endl;
    
    if(scounter[DTId]==1 && DTId.station()!=4){	
      std::cout<<"\t \t yes"<<std::endl;
      int dtWheel = DTId.wheel();
      int dtStation = DTId.station();
      int dtSector = DTId.sector();      
      LocalPoint segmentPosition= segment->localPosition();
      LocalVector segmentDirection=segment->localDirection();
      
      const GeomDet* gdet=dtGeo->idToDet(segment->geographicalId());
      const BoundPlane & DTSurface = gdet->surface();
      
      //check if the dimension of the segment is 4 
      std::cout<<"\t \t Is the segment 4D?"<<std::endl;
      
      if(segment->dimension()==4){
	std::cout<<"\t \t yes"<<std::endl;
	std::cout<<"\t \t DT Segment Dimension "<<segment->dimension()<<std::endl; 
	
	float Xo=segmentPosition.x();
	float Yo=segmentPosition.y();
	float Zo=segmentPosition.z();
	float dx=segmentDirection.x();
	float dy=segmentDirection.y();
	float dz=segmentDirection.z();
	std::cout<<"\t \t Loop over all the rolls asociated to this DT"<<std::endl;
	
	std::set<RPCDetId> rollsForThisDT = rollstoreDT[DTStationIndex(0,dtWheel,dtSector,dtStation)];
	
	std::cout<<"\t \t Number of rolls for this DT = "<<rollsForThisDT.size()<<std::endl;
        assert(rollsForThisDT.size()>=1);

	//Loop over all the rolls
	
	for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisDT.begin();iteraRoll != rollsForThisDT.end(); iteraRoll++){
	  const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll);
	  const BoundPlane & RPCSurface = rollasociated->surface(); 
	  
	  std::cout<<"\t \t \t RollID: "<<rollasociated->id()<<std::endl;
	  std::cout<<"\t \t \t Doing the extrapolation to this roll"<<std::endl;

	  std::cout<<"\t \t \t DT Segment Direction in DTLocal "<<segmentDirection<<std::endl;
	  std::cout<<"\t \t \t DT Segment Point in DTLocal "<<segmentPosition<<std::endl;
	  
	  GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0,0,0));
	  std::cout<<"\t \t \t Center (0,0,0) of the Roll in Global"<<CenterPointRollGlobal<<std::endl;
	  
	  LocalPoint CenterRollinDTFrame = DTSurface.toLocal(CenterPointRollGlobal);
	  std::cout<<"\t \t \t Center (0,0,0) Roll In DTLocal"<<CenterRollinDTFrame<<std::endl;
	    
	  float D=CenterRollinDTFrame.z();
	  
	  float X=Xo+dx*D/dz;
	  float Y=Yo+dy*D/dz;
	  float Z=D;
	
	  const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&(rollasociated->topology()));
	  LocalPoint xmin = top_->localPosition(0.);
	  std::cout<<"\t \t \t xmin of this  Roll "<<xmin<<"cm"<<std::endl;
	  LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
	  std::cout<<"\t \t \t xmax of this  Roll "<<xmax<<"cm"<<std::endl;
	  float rsize = fabs( xmax.x()-xmin.x() )*0.5;
	  std::cout<<"\t \t \t Roll Size "<<rsize<<"cm"<<std::endl;
	  float stripl = top_->stripLength();
	  std::cout<<"\t \t \t Strip Lenght "<<stripl<<"cm"<<std::endl;
	  
	  
	  std::cout<<"\t \t \t X Predicted in DTLocal= "<<X<<"cm"<<std::endl;
	  std::cout<<"\t \t \t Y Predicted in DTLocal= "<<Y<<"cm"<<std::endl;
	  std::cout<<"\t \t \t Z Predicted in DTLocal= "<<Z<<"cm"<<std::endl;
	  
	  float extrapolatedDistance = sqrt((X-Xo)*(X-Xo)+(Y-Yo)*(Y-Yo)+(Z-Zo)*(Z-Zo));
	  std::cout<<"\t \t \t Is the distance of extrapolation less than MaxD? ="<<extrapolatedDistance<<"cm"<<"MaxD="<<MaxD<<"cm"<<std::endl;
	  if(extrapolatedDistance<=MaxD){ 
	    std::cout<<"\t \t \t yes"<<std::endl;
	    
	    GlobalPoint GlobalPointExtrapolated = DTSurface.toGlobal(LocalPoint(X,Y,Z));
	    std::cout<<"\t \t \t Point ExtraPolated in Global"<<GlobalPointExtrapolated<< std::endl;
	    
	    LocalPoint PointExtrapolatedRPCFrame = RPCSurface.toLocal(GlobalPointExtrapolated);
	    std::cout<<"\t \t \t Point Extrapolated in RPCLocal"<<PointExtrapolatedRPCFrame<< std::endl;
	    std::cout<<"\t \t \t Does the extrapolation go inside this roll?"<<std::endl;
	    
	    if(fabs(PointExtrapolatedRPCFrame.z()) < 0.01 && fabs(PointExtrapolatedRPCFrame.x()) < rsize && fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){
	      RPCDetId  rollId = rollasociated->id();
	      std::cout<<"\t \t \t \t yes"<<std::endl;	
	      const float stripPredicted = 
		rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
		
	      std::cout<<"\t \t \t \t Candidate"<<rollasociated->id()<<" "<<"(from DT Segment) STRIP---> "<<stripPredicted<< std::endl;
		
	      int stripDetected = 0;
	      int minstripDetected = 0;

	      std::cout<<"\t \t \t \t Getting Roll Asociated"<<std::endl;	
	      RPCDigiCollection::Range rpcRangeDigi=rpcDigis->get(rollasociated->id());
		
	      int stripCounter = 0;
		
		
	      //--------- HISTOGRAM STRIP PREDICTED FROM DT  -------------------
		
	      RPCGeomServ rpcsrv(rollId);
	      std::string nameRoll = rpcsrv.name();
	      std::cout<<"\t \t \t \t The RPCName is"<<nameRoll<<std::endl;
	      _idList.push_back(nameRoll);
		
	      char detUnitLabel[128];
	      sprintf(detUnitLabel ,"%s",nameRoll.c_str());
	      sprintf(layerLabel ,"%s",nameRoll.c_str());
		
	      std::cout<<"\t \t \t \t Finding Id"<<nameRoll<<std::endl;
	      std::map<std::string, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(nameRoll);
	      std::cout<<"\t \t \t \t Done Finding Id"<<nameRoll<<std::endl;
		
		
	      if (meItr == meCollection.end()){
		meCollection[nameRoll] = bookDetUnitSeg(rollId);
	      }
		
	      std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];
		
	      sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
	      
	      std::cout<<"\t \t \t \t Filling the histogram"<<meIdDT<<std::endl;
	      meMap[meIdDT]->Fill(stripPredicted);
	      std::cout<<"\t \t \t \t Done Filling the histogram"<<std::endl;
		
	      sprintf(meIdDT,"ExpectedOccupancy2DFromDT_%s",detUnitLabel);
	      meMap[meIdDT]->Fill(stripPredicted,Y);
	       
	      //-----------------------------------------------------
		
	      totalcounter[0]++;
	      buff=counter[0];
	      buff[rollasociated->id()]++;
	      counter[0]=buff;
		
	      bool anycoincidence=false;
		
	      		
	      double minres=999999.;
	      
	      
	      std::cout<<"\t \t \t \t \t Loop over the digis in this roll looking for minres"<<std::endl;

	      for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){
		stripCounter++;
		stripDetected=digiIt->strip();	  
		double res = (double)(stripDetected) - (double)(stripPredicted);
		std::cout<<"\t \t \t \t \t \t Digi "<<*digiIt<<"\t Res = "<<res<<std::endl;//print the digis in the event
		if(fabs(res)<fabs(minres)){
		  minres=res;
		  minstripDetected = stripDetected;
		}
	      }
	      
	      LocalPoint minstripDetectedLocalPoint = top_->localPosition((float)(minstripDetected));
	      
	      float minrescms = PointExtrapolatedRPCFrame.x()-minstripDetectedLocalPoint.x();
	      float minrescmsY = PointExtrapolatedRPCFrame.y()-minstripDetectedLocalPoint.y();
	    	      
	      if(fabs(minres) < widestrip){
		
		std::cout<<"\t \t \t \t \t MinRes = "<<minres<<"  res(cm)="<<minrescms<<std::endl;

		hGlobalRes->Fill(minrescms);
		hGlobalResY->Fill(minrescmsY);
		
		sprintf(meIdRPC,"RPCResidualsFromDT_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(minres);
		
		sprintf(meIdRPC,"RPCResiduals2DFromDT_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(minres,Y);
		
		std::cout <<"\t \t \t \t \t COINCIDENCE Predict "<<stripPredicted<<"  (int)Predicted="<<(int)(stripPredicted)<<"  Detect="<<minstripDetected<<std::endl;
		anycoincidence=true;
		std::cout <<"\t \t \t \t \t Increassing counter"<<std::endl;
		totalcounter[1]++;
		buff=counter[1];
		buff[rollId]++;
		counter[1]=buff;
		
		sprintf(meIdRPC,"RealDetectedOccupancyFromDT_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(minstripDetected);
		
		sprintf(meIdRPC,"RPCDataOccupancyFromDT_%s",detUnitLabel);
		meMap[meIdRPC]->Fill((int)(stripPredicted));
		
		sprintf(meIdRPC,"RPCDataOccupancy2DFromDT_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(stripPredicted,Y);
		
	      }
	      
	      if(anycoincidence==false) {
		std::cout <<"\t \t \t \t \t THIS PREDICTION DOESN'T HAVE ANY CORRESPONDENCE WITH THE DATA"<<std::endl;
		totalcounter[2]++;
		buff=counter[2];
		buff[rollId]++;
		counter[2]=buff;		
		std::cout << "\t \t \t \t \t One for counterFAIL"<<std::endl;
		  
		ofrej<<"DTs Wh "<<dtWheel
		     <<"\t St "<<dtStation
		     <<"\t Se "<<dtSector
		     <<"\t Roll "<<rollasociated->id()
		     <<"\t Event "
		     <<iEvent.id().event()
		     <<std::endl;
	      }
		
	    }
	    else {
	      std::cout<<"\t \t \t \t No the prediction is outside of this roll"<<std::endl;
	    }//Condition for the right match
	  }else{
	    std::cout<<"\t \t \t No, Exrtrapolation too long!, canceled"<<std::endl;
	  }//D so big
	}//loop over all the rolls
      }
    }
    else {
      std::cout<<"\t \t No More than one segment in this chamber, or we are in Station 4"<<std::endl;
    }
  }
}
else {
  std::cout<<"This Event doesn't have any DT4DDSegment"<<std::endl; //is ther more than 1 segment in this event?
}
