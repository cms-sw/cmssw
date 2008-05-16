edm::ESHandle<DTGeometry> dtGeo;
iSetup.get<MuonGeometryRecord>().get(dtGeo);
   
edm::Handle<DTRecSegment4DCollection> all4DSegments;
iEvent.getByLabel(dt4DSegments, all4DSegments);
    
if(all4DSegments->size()>0){
  
  std::map<DTChamberId,int> scounter;
  DTRecSegment4DCollection::const_iterator segment;  
  
  for (segment = all4DSegments->begin();segment!=all4DSegments->end(); ++segment){
    scounter[segment->chamberId()]++;
  }    
  
  for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ 
    
    DTChamberId DTId = segment->chamberId();
    
    
    if(scounter[DTId] == 1 && DTId.station()==4){
      int dtWheel = DTId.wheel();
      int dtStation = DTId.station();
      int dtSector = DTId.sector();
      
      LocalPoint segmentPosition= segment->localPosition();
      LocalVector segmentDirection=segment->localDirection();
            
      //check if the dimension of the segment is 2
            
      if(segment->dimension()==2){
	
	if(dtStation==4){
	  
	  LocalVector segmentDirectionMB4=segmentDirection;
	  LocalPoint segmentPositionMB4=segmentPosition;
	  
	  bool compatiblesegments=false;
	  float dx=segmentDirectionMB4.x();
	  float dz=segmentDirectionMB4.z();

	  const BoundPlane& DTSurface4 = dtGeo->idToDet(DTId)->surface();
		  

	  DTRecSegment4DCollection::const_iterator segMB3;  
	  
	  for(segMB3=all4DSegments->begin();segMB3!=all4DSegments->end();++segMB3){
	    DTChamberId dtid3 = segMB3->chamberId();
	    
	    if(dtid3.station()==3&&dtid3.wheel()==DTId.wheel()&&scounter[dtid3] == 1&&segMB3->dimension()==4){
	      const GeomDet* gdet3=dtGeo->idToDet(segMB3->geographicalId());
	      const BoundPlane & DTSurface3 = gdet3->surface();
	      
	      float dx3=segMB3->localDirection().x();
	      float dy3=segMB3->localDirection().y();
	      float dz3=segMB3->localDirection().z();
	      
	      LocalVector segDirMB4inMB3Frame=DTSurface3.toLocal(DTSurface4.toGlobal(segmentDirectionMB4));
	      
	      double cosAng=fabs(dx*dx3+dz*dz3/sqrt((dx3*dx3+dz3*dz3)*(dx*dx+dz*dz)));
	      assert(fabs(cosAng)<=1.);

	      if(cosAng>MinCosAng){
		compatiblesegments=true;
		if(dtSector==13){
		 dtSector=4;
	        }
		if(dtSector==14){
		 dtSector=10;
		}
		assert(dtStation==4);
		std::set<RPCDetId> rollsForThisDT = rollstoreDT[DTStationIndex(0,dtWheel,dtSector,dtStation)];

        	assert(rollsForThisDT.size()>=1);

		
		for (std::set<RPCDetId>::iterator iteraRoll
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
		  
		  float X=Xo3+dx3*D/dz3;
		  float Y=Yo3+dy3*D/dz3;
		  float Z=D;
		  
		  const RectangularStripTopology* top_
		    =dynamic_cast<const RectangularStripTopology*>(&(rollasociated->topology())); //Topology roll asociado MB4
		  LocalPoint xmin = top_->localPosition(0.);
		  LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
		  float rsize = fabs( xmax.x()-xmin.x() )*0.5;
		  float stripl = top_->stripLength();
		  
		  GlobalPoint GlobalPointExtrapolated = DTSurfaceMB3.toGlobal(LocalPoint(X,Y,Z));
		  LocalPoint PointExtrapolatedRPCFrame = RPCSurfaceRB4.toLocal(GlobalPointExtrapolated);
		  
		  
		  if(fabs(PointExtrapolatedRPCFrame.z()) < 0.01  &&
		     fabs(PointExtrapolatedRPCFrame.x()) < rsize &&
		     fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){ 
		    
		    const float stripPredicted=
		      rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
		    
		    RPCDetId  rollId = rollasociated->id();
		    

		    //--------- HISTOGRAM STRIP PREDICTED FROM DT  -------------------
		    
	            RPCGeomServ rpcsrv(rollId);
	            std::string nameRoll = rpcsrv.name();
	            _idList.push_back(nameRoll);
		    
		    		    
		    char detUnitLabel[128];
		    sprintf(detUnitLabel ,"%s",nameRoll.c_str());
		    sprintf(layerLabel ,"%s",nameRoll.c_str());

	      
		    std::map<std::string, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(nameRoll);
		    if (meItr == meCollection.end()){
		      meCollection[nameRoll] = bookDetUnitSeg(rollId);
		    }
	      
		    std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];
	      
		    sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
		    meMap[meIdDT]->Fill(stripPredicted);
	      
		    sprintf(meIdDT,"ExpectedOccupancy2DFromDT_%s",detUnitLabel);
		    meMap[meIdDT]->Fill(stripPredicted,Y);

		    //-------------------------------------------------
			    
		    totalcounter[0]++;
		    buff=counter[0];
		    buff[rollId]++;
		    counter[0]=buff;		
		    
		    bool anycoincidence=false;
		    int stripDetected = 0;
		    RPCDigiCollection::Range rpcRangeDigi = rpcDigis->get(rollasociated->id());

		    for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){
		      stripDetected=digiIt->strip();
		      double res = (double)(stripDetected) - (double)(stripPredicted);
		      std::cout<<"\t \t \t \t \t Residual "<<res<<std::endl;
		      //-------filling the histograms--------------------

		      
		      sprintf(meIdRPC,"RPCResidualsFromDT_%s",detUnitLabel);
		      meMap[meIdRPC]->Fill(res);
		
		      sprintf(meIdRPC,"RPCResiduals2DFromDT_%s",detUnitLabel);
		      meMap[meIdRPC]->Fill(res,Y);


		      //-------------------------------------------------------


		      if(fabs(res)<widestripRB4){
			anycoincidence=true;

			//-------filling the histograms-------------------------------

			sprintf(meIdRPC,"RealDetectedOccupancyFromDT_%s",detUnitLabel);
			meMap[meIdRPC]->Fill(stripDetected);

			sprintf(meIdRPC,"RPCDataOccupancyFromDT_%s",detUnitLabel);
			meMap[meIdRPC]->Fill(stripPredicted);

			sprintf(meIdRPC,"RPCDataOccupancy2DFromDT_%s",detUnitLabel);
			meMap[meIdRPC]->Fill(stripPredicted,Y);


			//-------------------------------------------------------
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
		      ofrej<<"MB4 Wh "<<dtWheel
			   <<"\t St "<<dtStation
			   <<"\t Se "<<dtSector
			   <<"\t Roll "<<rollasociated->id()
			   <<"\t Event "
			   <<iEvent.id().event()
			   <<std::endl;
		    }
		  }
		  else{
		  }
		}// loop over the rolls 
	      }// are the segments compatibles
	      
	      else{
		compatiblesegments=false;
	      }
	    }else{//if dtid3.station()==3&&dtid3.sector()==DTId.sector()&&dtid3.wheel()==DTId.wheel()&&segMB3->dim()==4
	    }
	  }//lood over all the segments looking for one in MB3 
	}//Is the station 4? for this segment
	else{
	}
      }
      else{
      }
    }else{
    }//De aca para abajo esta en dtpart.inl
  }
}
else{
}
