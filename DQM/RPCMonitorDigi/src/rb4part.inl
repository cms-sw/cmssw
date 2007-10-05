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
    std::cout<<"\t \t DT Segment Dimension "<<segment->dimension()<<std::endl; 
    std::cout<<"\t \t Is the only in this DT?"<<std::endl;
    
    if(scounter[DTId] == 1){
      std::cout<<"\t \t yes"<<std::endl;
      int dtWheel = DTId.wheel();
      int dtStation = DTId.station();
      int dtSector = DTId.sector();
      
      LocalPoint segmentPosition= segment->localPosition();
      LocalVector segmentDirection=segment->localDirection();
      
      
      //check if the dimension of the segment is 2
      std::cout<<"\t \t Is the segment 2D?"<<std::endl;
      
      
      //DE ACA PARA ARRIBA SE REPITE EN dtpart.inl

      
      if(segment->dimension()==2){
	
	if(dtStation==4){
	  
	  LocalVector segmentDirectionMB4=segmentDirection;
	  LocalPoint segmentPositionMB4=segmentPosition;
	  
	  std::cout<<"MB4 \t 2D in RB4"<<DTId<<" with D="<<segment->dimension()<<segmentPositionMB4<<std::endl;	  
	  bool compatiblesegments=false;
	  Xo=segmentPositionMB4.x();
	  dx=segmentDirectionMB4.x();
	  dz=segmentDirectionMB4.z();
	  std::cout<<"MB4 \t \t Loop over all the segments in MB3"<<std::endl;	  
	  DTRecSegment4DCollection::const_iterator segMB3;  
	  
	  const BoundPlane& DTSurface4 = dtGeo->idToDet(DTId)->surface();
	  
	  for(segMB3=all4DSegments->begin();segMB3!=all4DSegments->end();++segMB3){
	    DTChamberId dtid3 = segMB3->chamberId();
	    
	    if(dtid3.station()==3&&dtid3.sector()==DTId.sector()&&dtid3.wheel()==DTId.wheel()){
	      
	      const GeomDet* gdet3=dtGeo->idToDet(segMB3->geographicalId());
	      const BoundPlane & DTSurface3 = gdet3->surface();
	      
	      float dx3=segMB3->localDirection().x();
	      float dy3=segMB3->localDirection().y();
	      float dz3=segMB3->localDirection().z();
	      
	      LocalVector segDirMB4inMB3Frame=DTSurface3.toLocal(DTSurface4.toGlobal(segmentDirectionMB4));
	      
	      float cosAng=dx*dx3+dz*dz3/sqrt((dx3*dx3+dz3*dz3)*(dx*dx+dz*dz));
	      
	      //assertion(fabs(cosAng)<=1);
	      float MaxCosAng=0.1; //Por ahora Luego en el config file
	      if(cosAng<MaxCosAng){
		compatiblesegments=true;
		std::set<RPCDetId> rollsForThisDT = 
		  rollstoreDT[DTStationIndex(0,dtWheel,dtSector,dtStation)];
		std::cout<<"MB4 \t \t Loop over all the rolls asociated to RB4"<<std::cout;
		for (
		     std::set<RPCDetId>::iterator iteraRoll
		       =rollsForThisDT.begin();iteraRoll != rollsForThisDT.end(); iteraRoll++){
		  
		  const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll); //roll asociado a MB4
		  const BoundPlane & RPCSurfaceRB4 = rollasociated->surface(); //surface MB4
		  const GeomDet* gdet=dtGeo->idToDet(segMB3->geographicalId()); 
		  const BoundPlane & DTSurfaceMB3 = gdet->surface(); // surface MB3
		  
		  GlobalPoint CenterPointRollGlobal=RPCSurfaceRB4.toGlobal(LocalPoint(0,0,0));
		  
		  LocalPoint CenterRollinMB3Frame = DTSurfaceMB3.toLocal(CenterPointRollGlobal);
		  float D=CenterRollinMB3Frame.z();
		  
		  float Xo3=segMB3->localPosition().x();
		  float Yo3=segMB3->localPosition().y();
		  
		  X=Xo3+dx3*D/dz3;
		  Y=Yo3+dy3*D/dz3;
		  Z=D;
		  
		  const RectangularStripTopology* top_
		    =dynamic_cast<const RectangularStripTopology*>(&(rollasociated->topology())); //Topologia roll asociado MB4
		  
		  LocalPoint xmin = top_->localPosition(0.);
		  LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
		  float rsize = fabs( xmax.x()-xmin.x() )*0.5;
		  float stripl = top_->stripLength();
		  
		  GlobalPoint GlobalPointExtrapolated = DTSurfaceMB3.toGlobal(LocalPoint(X,Y,Z));
		  LocalPoint PointExtrapolatedRPCFrame = RPCSurfaceRB4.toLocal(GlobalPointExtrapolated);
		  
		  std::cout<<"MB4 \t \t \t Point Extrapolated.z in RPC Local "<<PointExtrapolatedRPCFrame.z()<<std::cout;
		  
		  if(fabs(PointExtrapolatedRPCFrame.z()) < 0.01  &&
		     fabs(PointExtrapolatedRPCFrame.x()) < rsize &&
		     fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){ 
		    
		    const float stripPredicted=
		      rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
		    
		    RPCDetId  rollId = rollasociated->id();
		    
		    totalcounter[0]++;
		    buff=counter[0];
		    buff[rollId]++;
		    counter[0]=buff;		
		    
		    bool anycoincidence=false;
		    int stripDetected = 0;
		    RPCDigiCollection::Range rpcRangeDigi = rpcDigis->get(rollasociated->id());
		    for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){
		      stripDetected=digiIt->strip();
		      float res = fabs((float)(stripDetected) - stripPredicted);
		      if(res<widestrip){
			std::cout <<"MB4 \t \t \t \t COINCEDENCE Predict "<<stripPredicted<<" Detect "<<stripDetected<<std::endl;
			anycoincidence=true;
			break;
		      }
		    }
		    
		    if(anycoincidence){
		      
		      totalcounter[1]++;
		      buff=counter[1];
		      buff[rollId]++;
		      counter[1]=buff;		
		    }
		    else{
		      totalcounter[2]++;
		      buff=counter[2];
		      buff[rollId]++;
		      counter[2]=buff;		
		      std::cout <<"MB4 \t \t \t \t THIS PREDICTION DOESN'T HAVE ANY CORRESPONDENCE WITH THE DATA"<<std::endl;
		      ofrej<<"Wh "<<dtWheel
			   <<"\t St "<<dtStation
			   <<"\t Se "<<dtSector
			   <<"\t Roll "<<rollasociated->id()
			   <<"\t Event "
			   <<iEvent.id().event()
			   <<std::endl;
		    }
		  }
		}// loop over the rolls 
	      }// are the segments compatibles
	      
	      else{
		compatiblesegments=false;
		std::cout<<"MB4 \t \t I found segments in MB4 and MB3 same wheel and sector but not compatibles Diferent Directions"<<std::endl;
	      }
	    }//if dtid3.station()==3&&dtid3.sector()==DTId.sector()&&dtid3.wheel()==DTId.wheel()
	  }//lood over all the segments looking for one in MB3 
	}//Is the station 4? for this segment
      }
      else{
	std::cout<<"\t \t Strange Segment Is not a 4D Segment neither a 2D in MB4"<<std::endl;
      }
    }//De aca para abajo esta en dtpart.inl
  }
}
