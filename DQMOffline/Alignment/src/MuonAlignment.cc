
/*
 *  DQM muon alignment analysis monitoring
 *
 *  \author J. Fernandez - Univ. Oviedo <Javier.Fernandez@cern.ch>
 */

#include "DQMOffline/Alignment/interface/MuonAlignment.h"


MuonAlignment::MuonAlignment(const edm::ParameterSet& pSet) {

    metname = "MuonAlignment";

    LogTrace(metname)<<"[MuonAlignment] Constructor called!"<<std::endl;

    parameters = pSet;

    theMuonCollectionLabel = consumes<reco::TrackCollection>(parameters.getParameter<edm::InputTag>("MuonCollection"));
  
    theRecHits4DTagDT = consumes<DTRecSegment4DCollection>(parameters.getParameter<edm::InputTag>("RecHits4DDTCollectionTag"));
    theRecHits4DTagCSC = consumes<CSCSegmentCollection>(parameters.getParameter<edm::InputTag>("RecHits4DCSCCollectionTag"));
  
    resLocalXRangeStation1 = parameters.getUntrackedParameter<double>("resLocalXRangeStation1");
    resLocalXRangeStation2 = parameters.getUntrackedParameter<double>("resLocalXRangeStation2");
    resLocalXRangeStation3 = parameters.getUntrackedParameter<double>("resLocalXRangeStation3");
    resLocalXRangeStation4 = parameters.getUntrackedParameter<double>("resLocalXRangeStation4");
    resLocalYRangeStation1 = parameters.getUntrackedParameter<double>("resLocalYRangeStation1");
    resLocalYRangeStation2 = parameters.getUntrackedParameter<double>("resLocalYRangeStation2");
    resLocalYRangeStation3 = parameters.getUntrackedParameter<double>("resLocalYRangeStation3");
    resLocalYRangeStation4 = parameters.getUntrackedParameter<double>("resLocalYRangeStation4");
    resPhiRange = parameters.getUntrackedParameter<double>("resPhiRange");
    resThetaRange = parameters.getUntrackedParameter<double>("resThetaRange");

    meanPositionRange = parameters.getUntrackedParameter<double>("meanPositionRange");
    rmsPositionRange = parameters.getUntrackedParameter<double>("rmsPositionRange");
    meanAngleRange = parameters.getUntrackedParameter<double>("meanAngleRange");
    rmsAngleRange = parameters.getUntrackedParameter<double>("rmsAngleRange");

    nbins = parameters.getUntrackedParameter<unsigned int>("nbins");
    min1DTrackRecHitSize = parameters.getUntrackedParameter<unsigned int>("min1DTrackRecHitSize");
    min4DTrackSegmentSize = parameters.getUntrackedParameter<unsigned int>("min4DTrackSegmentSize");

    doDT = parameters.getUntrackedParameter<bool>("doDT");
    doCSC = parameters.getUntrackedParameter<bool>("doCSC");
    doSummary = parameters.getUntrackedParameter<bool>("doSummary");

    numberOfTracks=0;
    numberOfHits=0;
  
    MEFolderName = parameters.getParameter<std::string>("FolderName");  
    topFolder << MEFolderName+"/Alignment/Muon";
}

MuonAlignment::~MuonAlignment() { 
}


void MuonAlignment::beginJob() {


    LogTrace(metname)<<"[MuonAlignment] Parameters initialization";
  
    if(!(doDT || doCSC) ) { 
        edm::LogError("MuonAlignment") <<" Error!! At least one Muon subsystem (DT or CSC) must be monitorized!!" << std::endl;
        edm::LogError("MuonAlignment") <<" Please enable doDT or doCSC to True in your python cfg file!!!" << std::endl;
        exit(1);
    }
  
    dbe = edm::Service<DQMStore>().operator->();

    if (doSummary){
        if (doDT){
            dbe->setCurrentFolder(topFolder.str()+"/DT");
            hLocalPositionDT=dbe->book2D("hLocalPositionDT","Local DT position (cm) absolute MEAN residuals;Sector;;cm", 14,1, 15,40,0,40);
            hLocalAngleDT=dbe->book2D("hLocalAngleDT","Local DT angle (rad) absolute MEAN residuals;Sector;;rad", 14,1, 15,40,0,40); 
            hLocalPositionRmsDT=dbe->book2D("hLocalPositionRmsDT","Local DT position (cm) RMS residuals;Sector;;cm", 14,1, 15,40,0,40);
            hLocalAngleRmsDT=dbe->book2D("hLocalAngleRmsDT","Local DT angle (rad) RMS residuals;Sector;;rad", 14,1, 15,40,0,40); 

            hLocalXMeanDT=dbe->book1D("hLocalXMeanDT","Distribution of absolute MEAN Local X (cm) residuals for DT;<X> (cm);number of chambers",100,0,meanPositionRange);
            hLocalXRmsDT=dbe->book1D("hLocalXRmsDT","Distribution of RMS Local X (cm) residuals for DT;X RMS (cm);number of chambers", 100,0,rmsPositionRange);
            hLocalYMeanDT=dbe->book1D("hLocalYMeanDT","Distribution of absolute MEAN Local Y (cm) residuals for DT;<Y> (cm);number of chambers", 100,0,meanPositionRange);
            hLocalYRmsDT=dbe->book1D("hLocalYRmsDT","Distribution of RMS Local Y (cm) residuals for DT;Y RMS (cm);number of chambers", 100,0,rmsPositionRange);

            hLocalPhiMeanDT=dbe->book1D("hLocalPhiMeanDT","Distribution of MEAN #phi (rad) residuals for DT;<#phi>(rad);number of chambers", 100,-meanAngleRange,meanAngleRange);
            hLocalPhiRmsDT=dbe->book1D("hLocalPhiRmsDT","Distribution of RMS #phi (rad) residuals for DT;#phi RMS (rad);number of chambers", 100,0,rmsAngleRange);
            hLocalThetaMeanDT=dbe->book1D("hLocalThetaMeanDT","Distribution of MEAN #theta (rad) residuals for DT;<#theta>(rad);number of chambers", 100,-meanAngleRange,meanAngleRange);
            hLocalThetaRmsDT=dbe->book1D("hLocalThetaRmsDT","Distribution of RMS #theta (rad) residuals for DT;#theta RMS (rad);number of chambers",100,0,rmsAngleRange);
        }
	
        if (doCSC){
            dbe->setCurrentFolder(topFolder.str()+"/CSC");
            hLocalPositionCSC=dbe->book2D("hLocalPositionCSC","Local CSC position (cm) absolute MEAN residuals;Sector;;cm",36,1,37,40,0,40);
            hLocalAngleCSC=dbe->book2D("hLocalAngleCSC","Local CSC angle (rad) absolute MEAN residuals;Sector;;rad", 36,1,37,40,0,40); 
            hLocalPositionRmsCSC=dbe->book2D("hLocalPositionRmsCSC","Local CSC position (cm) RMS residuals;Sector;;cm", 36,1,37,40,0,40);
            hLocalAngleRmsCSC=dbe->book2D("hLocalAngleRmsCSC","Local CSC angle (rad) RMS residuals;Sector;;rad", 36,1,37,40,0,40); 
	
            hLocalXMeanCSC=dbe->book1D("hLocalXMeanCSC","Distribution of absolute MEAN Local X (cm) residuals for CSC;<X> (cm);number of chambers",100,0,meanPositionRange);
            hLocalXRmsCSC=dbe->book1D("hLocalXRmsCSC","Distribution of RMS Local X (cm) residuals for CSC;X RMS (cm);number of chambers", 100,0,rmsPositionRange);
            hLocalYMeanCSC=dbe->book1D("hLocalYMeanCSC","Distribution of absolute MEAN Local Y (cm) residuals for CSC;<Y> (cm);number of chambers", 100,0,meanPositionRange);
            hLocalYRmsCSC=dbe->book1D("hLocalYRmsCSC","Distribution of RMS Local Y (cm) residuals for CSC;Y RMS (cm);number of chambers", 100,0,rmsPositionRange);

            hLocalPhiMeanCSC=dbe->book1D("hLocalPhiMeanCSC","Distribution of absolute MEAN #phi (rad) residuals for CSC;<#phi>(rad);number of chambers", 100,0,meanAngleRange);
            hLocalPhiRmsCSC=dbe->book1D("hLocalPhiRmsCSC","Distribution of RMS #phi (rad) residuals for CSC;#phi RMS (rad);number of chambers", 100,0,rmsAngleRange);
            hLocalThetaMeanCSC=dbe->book1D("hLocalThetaMeanCSC","Distribution of absolute MEAN #theta (rad) residuals for CSC;<#theta>(rad);number of chambers", 100,0,meanAngleRange);
            hLocalThetaRmsCSC=dbe->book1D("hLocalThetaRmsCSC","Distribution of RMS #theta (rad) residuals for CSC;#theta RMS (rad);number of chambers",100,0,rmsAngleRange);
        }
    }


	// Chamber individual histograms
	// I need to create all of them even if they are empty to allow proper root merging

	// variables for histos ranges	
	double rangeX=0,rangeY=0;
	std::string nameOfHistoLocalX,nameOfHistoLocalY,nameOfHistoLocalPhi,nameOfHistoLocalTheta;

	for (int station = -4; station<5; station++){

        //This piece of code calculates the range of the residuals
        switch(abs(station)) {
            case 1:
            {rangeX = resLocalXRangeStation1; rangeY = resLocalYRangeStation1;}
            break;
            case 2:
            {rangeX = resLocalXRangeStation2; rangeY = resLocalYRangeStation2;}
            break;
            case 3:
            {rangeX = resLocalXRangeStation3; rangeY = resLocalYRangeStation3;}
            break;
            case 4:
            {rangeX = resLocalXRangeStation4; rangeY = resLocalYRangeStation4;}
            break;
            default:
                break;
        }
        if (doDT){
            if(station>0){
					
                for(int wheel=-2;wheel<3;wheel++){
			
                    for (int sector=1;sector<15;sector++){
				
                        if(!((sector==13 || sector ==14) && station!=4)){
					
			    std::stringstream Wheel; Wheel<<wheel;
                            std::stringstream Station; Station<<station;
                            std::stringstream Sector; Sector<<sector;
										
                            nameOfHistoLocalX="ResidualLocalX_W"+Wheel.str()+"MB"+Station.str()+"S"+Sector.str();
                            nameOfHistoLocalPhi= "ResidualLocalPhi_W"+Wheel.str()+"MB"+Station.str()+"S"+Sector.str();
                            nameOfHistoLocalTheta= "ResidualLocalTheta_W"+Wheel.str()+"MB"+Station.str()+"S"+Sector.str();
                            nameOfHistoLocalY= "ResidualLocalY_W"+Wheel.str()+"MB"+Station.str()+"S"+Sector.str();
                                                                               
                            dbe->setCurrentFolder(topFolder.str()+
                                                  "/DT/Wheel"+Wheel.str()+
                                                  "/Station"+Station.str()+
                                                  "/Sector"+Sector.str());

                            //Create ME and push histos into their respective vectors

                            MonitorElement *histoLocalX = dbe->book1D(nameOfHistoLocalX, nameOfHistoLocalX, nbins, -rangeX, rangeX);
                            unitsLocalX.push_back(histoLocalX);
                            MonitorElement *histoLocalPhi = dbe->book1D(nameOfHistoLocalPhi, nameOfHistoLocalPhi, nbins, -resPhiRange, resPhiRange);
                            unitsLocalPhi.push_back(histoLocalPhi);
                            MonitorElement *histoLocalTheta = dbe->book1D(nameOfHistoLocalTheta, nameOfHistoLocalTheta, nbins, -resThetaRange, resThetaRange);
                            unitsLocalTheta.push_back(histoLocalTheta);
                            MonitorElement *histoLocalY = dbe->book1D(nameOfHistoLocalY, nameOfHistoLocalY, nbins, -rangeY, rangeY);
                            unitsLocalY.push_back(histoLocalY);
				    	}
                    }
                }
            } //station>0
        }// doDT
	
        if (doCSC){
            if(station!=0){

                for(int ring=1;ring<5;ring++){

                    for(int chamber=1;chamber<37;chamber++){
				
                        if( !( ((abs(station)==2 || abs(station)==3 || abs(station)==4) && ring==1 && chamber>18) || 
                               ((abs(station)==2 || abs(station)==3 || abs(station)==4) && ring>2)) ){
                            std::stringstream Ring; Ring<<ring;
                            std::stringstream Station; Station<<station;
                            std::stringstream Chamber; Chamber<<chamber;
					                                       
                            nameOfHistoLocalX="ResidualLocalX_ME"+Station.str()+"R"+Ring.str()+"C"+Chamber.str();
                            nameOfHistoLocalPhi= "ResidualLocalPhi_ME"+Station.str()+"R"+Ring.str()+"C"+Chamber.str();
                            nameOfHistoLocalTheta= "ResidualLocalTheta_ME"+Station.str()+"R"+Ring.str()+"C"+Chamber.str();
                            nameOfHistoLocalY= "ResidualLocalY_ME"+Station.str()+"R"+Ring.str()+"C"+Chamber.str();
                                                                               
                            dbe->setCurrentFolder(topFolder.str()+
                                                  "/CSC/Station"+Station.str()+
                                                  "/Ring"+Ring.str()+
                                                  "/Chamber"+Chamber.str());

                            //Create ME and push histos into their respective vectors

                            MonitorElement *histoLocalX = dbe->book1D(nameOfHistoLocalX, nameOfHistoLocalX, nbins, -rangeX, rangeX);
                            unitsLocalX.push_back(histoLocalX);
                            MonitorElement *histoLocalPhi = dbe->book1D(nameOfHistoLocalPhi, nameOfHistoLocalPhi, nbins, -resPhiRange, resPhiRange);
                            unitsLocalPhi.push_back(histoLocalPhi);
                            MonitorElement *histoLocalTheta = dbe->book1D(nameOfHistoLocalTheta, nameOfHistoLocalTheta, nbins, -resThetaRange, resThetaRange);
                            unitsLocalTheta.push_back(histoLocalTheta);
                            MonitorElement *histoLocalY = dbe->book1D(nameOfHistoLocalY, nameOfHistoLocalY, nbins, -rangeY, rangeY);
                            unitsLocalY.push_back(histoLocalY);

                        }
                    }
                }
            } // station!=0
        }// doCSC

	} // loop on stations
}


void MuonAlignment::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

    LogTrace(metname)<<"[MuonAlignment] Analysis of event # ";
  
    edm::ESHandle<MagneticField> theMGField;
    iSetup.get<IdealMagneticFieldRecord>().get(theMGField);

    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
    iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  	edm::ESHandle<Propagator> thePropagatorOpp;
  	iSetup.get<TrackingComponentsRecord>().get( "SmartPropagatorOpposite", thePropagatorOpp );
  
  	edm::ESHandle<Propagator> thePropagatorAlo;
 	iSetup.get<TrackingComponentsRecord>().get( "SmartPropagator", thePropagatorAlo );

//	edm::ESHandle<Propagator> thePropagator;
//  	iSetup.get<TrackingComponentsRecord>().get( "SmartPropagatorAnyOpposite", thePropagator );

    // Get the RecoMuons collection from the event
    edm::Handle<reco::TrackCollection> muons;
    iEvent.getByToken(theMuonCollectionLabel, muons);

    // Get the 4D DTSegments
    edm::Handle<DTRecSegment4DCollection> all4DSegmentsDT;
    iEvent.getByToken(theRecHits4DTagDT, all4DSegmentsDT);
    DTRecSegment4DCollection::const_iterator segmentDT;

    // Get the 4D CSCSegments
    edm::Handle<CSCSegmentCollection> all4DSegmentsCSC;
    iEvent.getByToken(theRecHits4DTagCSC, all4DSegmentsCSC);
    CSCSegmentCollection::const_iterator segmentCSC;
  
    //Vectors used to perform the matching between Segments and hits from Track
    intDVector indexCollectionDT;
    intDVector indexCollectionCSC;

	
//        thePropagator = new SteppingHelixPropagator(&*theMGField, alongMomentum);

    int countTracks   = 0;
    reco::TrackCollection::const_iterator muon;
    for (muon = muons->begin(); muon != muons->end(); ++muon){
//   	if(muon->isGlobalMuon()){
//   	if(muon->isStandAloneMuon()){

        int countPoints   = 0;

//      reco::TrackRef trackTR = muon->innerTrack();
//      reco::TrackRef trackSA = muon->outerTrack();
        //reco::TrackRef trackSA = muon;
    
        if(muon->recHitsSize()>(min1DTrackRecHitSize-1)) {

//      reco::TransientTrack tTrackTR( *trackTR, &*theMGField, theTrackingGeometry );
            reco::TransientTrack tTrackSA(*muon,&*theMGField,theTrackingGeometry); 

            // Adapted code for muonCosmics
   
            Double_t  innerPerpSA  = tTrackSA.innermostMeasurementState().globalPosition().perp();
            Double_t  outerPerpSA  = tTrackSA.outermostMeasurementState().globalPosition().perp();
     
            TrajectoryStateOnSurface innerTSOS=tTrackSA.outermostMeasurementState();
//      PropagationDirection propagationDir=alongMomentum;
            const Propagator * thePropagator;
      
            // Define which kind of reco track is used
            if ( (outerPerpSA-innerPerpSA) > 0 ) {

                trackRefitterType = "LHCLike";
                innerTSOS = tTrackSA.innermostMeasurementState();
                thePropagator = thePropagatorAlo.product();
//	propagationDir = alongMomentum;
	  
            }else {//if ((outerPerpSA-innerPerpSA) < 0 ) {
	
                trackRefitterType = "CosmicLike";
                innerTSOS = tTrackSA.outermostMeasurementState();
                thePropagator = thePropagatorOpp.product(); 
//	propagationDir = oppositeToMomentum;
      
            }	

            RecHitVector  my4DTrack = this->doMatching(*muon, all4DSegmentsDT, all4DSegmentsCSC, &indexCollectionDT, &indexCollectionCSC, theTrackingGeometry);
  
    
//cut in number of segments
            if(my4DTrack.size()>(min4DTrackSegmentSize-1) ){


// start propagation
//    TrajectoryStateOnSurface innerTSOS = track.impactPointState();
//                    TrajectoryStateOnSurface innerTSOS = track.innermostMeasurementState();

                //If the state is valid
                if(innerTSOS.isValid()) {

                    //Loop over Associated segments
                    for(RecHitVector::iterator rechit = my4DTrack.begin(); rechit != my4DTrack.end(); ++rechit) {
	
                        const GeomDet* geomDet = theTrackingGeometry->idToDet((*rechit)->geographicalId());
//Otherwise the propagator could throw an exception
                        const Plane* pDest = dynamic_cast<const Plane*>(&geomDet->surface());
                        const Cylinder* cDest = dynamic_cast<const Cylinder*>(&geomDet->surface());

                        if(pDest != 0 || cDest != 0) {   
 
//			 	    Propagator *updatePropagator=thePropagator->clone();
//				    updatePropagator->setPropagationDirection(propagationDir);
                            TrajectoryStateOnSurface destiny = thePropagator->propagate(*(innerTSOS.freeState()), geomDet->surface());

                            if(!destiny.isValid()|| !destiny.hasError()) continue;

                            const long rawId= (*rechit)->geographicalId().rawId();
                            int position = -1;

                            DetId myDet(rawId);
                            int det = myDet.subdetId();
                            int wheel=0,station=0,sector=0;
                            int endcap=0,ring=0,chamber=0;
                            bool goAhead = (det==1 && doDT) || (det==2 && doCSC);

                            double residualLocalX=0,residualLocalPhi=0,residualLocalY=0,residualLocalTheta=0;

                            // Fill generic histograms
                            //If it's a DT
                            if(det == 1 && doDT) {
                                DTChamberId myChamber(rawId);
                                wheel=myChamber.wheel();
                                station = myChamber.station();
                                sector=myChamber.sector();
	      
                                residualLocalX = (*rechit)->localPosition().x() -destiny.localPosition().x();
		

                                residualLocalPhi = atan2(((RecSegment *)(*rechit))->localDirection().z(), 
                                                         ((RecSegment*)(*rechit))->localDirection().x()) - atan2(destiny.localDirection().z(), destiny.localDirection().x());
                                if(station!=4){
		
                                    residualLocalY = (*rechit)->localPosition().y() - destiny.localPosition().y();
		

                                    residualLocalTheta = atan2(((RecSegment *)(*rechit))->localDirection().z(), 
                                                               ((RecSegment*)(*rechit))->localDirection().y()) - atan2(destiny.localDirection().z(), destiny.localDirection().y());

                                }

                            } 
                            else if (det==2 && doCSC){
                                CSCDetId myChamber(rawId);
                                endcap= myChamber.endcap();
                                station = myChamber.station();
                                if(endcap==2) station = -station;
                                ring = myChamber.ring();
                                chamber=myChamber.chamber();

                                residualLocalX = (*rechit)->localPosition().x() -destiny.localPosition().x();
		
                                residualLocalY = (*rechit)->localPosition().y() - destiny.localPosition().y();
		
                                residualLocalPhi = atan2(((RecSegment *)(*rechit))->localDirection().y(), 
                                                         ((RecSegment*)(*rechit))->localDirection().x()) - atan2(destiny.localDirection().y(), destiny.localDirection().x());

                                residualLocalTheta = atan2(((RecSegment *)(*rechit))->localDirection().y(), 
                                                           ((RecSegment*)(*rechit))->localDirection().z()) - atan2(destiny.localDirection().y(), destiny.localDirection().z());

                            }
                            else{
                                residualLocalX=0,residualLocalPhi=0,residualLocalY=0,residualLocalTheta=0;
                            }
			
                            // Fill individual chamber histograms


                            std::string nameOfHistoLocalX;

	
                            if(det==1 && doDT){ // DT
                                std::stringstream Wheel; Wheel<<wheel;
                                std::stringstream Station; Station<<station;
                                std::stringstream Sector; Sector<<sector;
					
                                nameOfHistoLocalX="ResidualLocalX_W"+Wheel.str()+"MB"+Station.str()+"S"+Sector.str();
 
                            }
                            else if(det==2 && doCSC){ //CSC
                                std::stringstream Ring; Ring<<ring;
                                std::stringstream Station; Station<<station;
                                std::stringstream Chamber; Chamber<<chamber;
					                                       
                                nameOfHistoLocalX="ResidualLocalX_ME"+Station.str()+"R"+Ring.str()+"C"+Chamber.str();
                            }		    
	    
                            if (goAhead){
				    
                                for(unsigned int i=0 ; i<unitsLocalX.size() ; i++)
                                {

                                    if( nameOfHistoLocalX==unitsLocalX[i]->getName()){
                                        position=i; break;}
                                }
					    
  					    
	                            if(trackRefitterType == "CosmicLike") { //problem with angle convention in reverse extrapolation
                                    residualLocalPhi += 3.1416;
                                    residualLocalTheta +=3.1416;
                                }
				
                                unitsLocalX.at(position)->Fill(residualLocalX);
                                unitsLocalPhi.at(position)->Fill(residualLocalPhi);

                                if(det==1 && station!=4) {unitsLocalY.at(position)->Fill(residualLocalY); 
                                    unitsLocalTheta.at(position)->Fill(residualLocalTheta);}
 
                                else if(det==2) {unitsLocalY.at(position)->Fill(residualLocalY);
                                    unitsLocalTheta.at(position)->Fill(residualLocalTheta);}
                                
	   
                                countPoints++;
                                // if at least one point from this track is used, count this track
                                if (countPoints==1) countTracks++;
                            }		
	
                            innerTSOS = destiny;

                            //delete thePropagator;

                        }else {
		  	    edm::LogError("MuonAlignment") <<" Error!! Exception in propagator catched" << std::endl;
                            continue;
                        }

                    } //loop over my4DTrack
                } //TSOS was valid

            } // cut in at least 4 segments

        } //end cut in RecHitsSize>36
        numberOfHits=numberOfHits+countPoints;
//       } //Muon is GlobalMuon
    } //loop over Muons
    numberOfTracks=numberOfTracks+countTracks;

//        delete thePropagator;

//	else edm::LogError("MuonAlignment")<<"Error!! Specified MuonCollection "<< theMuonTrackCollectionLabel <<" is not present in event. ProductNotFound!!"<<std::endl;
}

RecHitVector MuonAlignment::doMatching(const reco::Track &staTrack, edm::Handle<DTRecSegment4DCollection> &all4DSegmentsDT, edm::Handle<CSCSegmentCollection> &all4DSegmentsCSC, intDVector *indexCollectionDT, intDVector *indexCollectionCSC, edm::ESHandle<GlobalTrackingGeometry> &theTrackingGeometry) {

    DTRecSegment4DCollection::const_iterator segmentDT;
    CSCSegmentCollection::const_iterator segmentCSC;
  
    std::vector<int> positionDT;
    std::vector<int> positionCSC;
    RecHitVector my4DTrack;
  
    //Loop over the hits of the track
    for(unsigned int counter = 0; counter != staTrack.recHitsSize()-1; counter++) {
    
        TrackingRecHitRef myRef = staTrack.recHit(counter);
        const TrackingRecHit *rechit = myRef.get();
        const GeomDet* geomDet = theTrackingGeometry->idToDet(rechit->geographicalId());
    
        //It's a DT Hit
        if(geomDet->subDetector() == GeomDetEnumerators::DT) {
      
            //Take the layer associated to this hit
            DTLayerId myLayer(rechit->geographicalId().rawId());
      
            int NumberOfDTSegment = 0;
            //Loop over segments
            for(segmentDT = all4DSegmentsDT->begin(); segmentDT != all4DSegmentsDT->end(); ++segmentDT) {
	
                //By default the chamber associated to this Segment is new
                bool isNewChamber = true;
	
                //Loop over segments already included in the vector of segments in the actual track
                for(std::vector<int>::iterator positionIt = positionDT.begin(); positionIt != positionDT.end(); positionIt++) {
	  
                    //If this segment has been used before isNewChamber = false
                    if(NumberOfDTSegment == *positionIt) isNewChamber = false;
                }
	
                //Loop over vectors of segments associated to previous tracks
                for(std::vector<std::vector<int> >::iterator collect = indexCollectionDT->begin(); collect != indexCollectionDT->end(); ++collect) {
	  
                    //Loop over segments associated to a track
                    for(std::vector<int>::iterator positionIt = (*collect).begin(); positionIt != (*collect).end(); positionIt++) {
	    
                        //If this segment was used in a previos track then isNewChamber = false
                        if(NumberOfDTSegment == *positionIt) isNewChamber = false;
                    }
                }
	
                //If the chamber is new
                if(isNewChamber) {
	  
                    DTChamberId myChamber((*segmentDT).geographicalId().rawId());
                    //If the layer of the hit belongs to the chamber of the 4D Segment
                    if(myLayer.wheel() == myChamber.wheel() && myLayer.station() == myChamber.station() && myLayer.sector() == myChamber.sector()) {
	    
                        //push position of the segment and tracking rechit
                        positionDT.push_back(NumberOfDTSegment);
                        my4DTrack.push_back((TrackingRecHit *) &(*segmentDT));
                    }
                }
                NumberOfDTSegment++;
            }
            //In case is a CSC
        } else if (geomDet->subDetector() == GeomDetEnumerators::CSC) {
      
            //Take the layer associated to this hit
            CSCDetId myLayer(rechit->geographicalId().rawId());
      
            int NumberOfCSCSegment = 0;
            //Loop over 4Dsegments
            for(segmentCSC = all4DSegmentsCSC->begin(); segmentCSC != all4DSegmentsCSC->end(); segmentCSC++) {
	
                //By default the chamber associated to the segment is new
                bool isNewChamber = true;
	
                //Loop over segments in the current track
                for(std::vector<int>::iterator positionIt = positionCSC.begin(); positionIt != positionCSC.end(); positionIt++) {
	  
                    //If this segment has been used then newchamber = false
                    if(NumberOfCSCSegment == *positionIt) isNewChamber = false;
                }
                //Loop over vectors of segments in previous tracks
                for(std::vector<std::vector<int> >::iterator collect = indexCollectionCSC->begin(); collect != indexCollectionCSC->end(); ++collect) {
	  
                    //Loop over segments in a track
                    for(std::vector<int>::iterator positionIt = (*collect).begin(); positionIt != (*collect).end(); positionIt++) {
	    
                        //If the segment was used in a previous track isNewChamber = false
                        if(NumberOfCSCSegment == *positionIt) isNewChamber = false;
                    }
                }
                //If the chamber is new
                if(isNewChamber) {
	  
                    CSCDetId myChamber((*segmentCSC).geographicalId().rawId());
                    //If the chambers are the same
                    if(myLayer.chamberId() == myChamber.chamberId()) {
                        //push
                        positionCSC.push_back(NumberOfCSCSegment);
                        my4DTrack.push_back((TrackingRecHit *) &(*segmentCSC));
                    }
                }
                NumberOfCSCSegment++;
            }
        }
    }
  
    indexCollectionDT->push_back(positionDT);
    indexCollectionCSC->push_back(positionCSC);
  
    if ( trackRefitterType == "CosmicLike") {

        std::reverse(my4DTrack.begin(),my4DTrack.end());

    }
    return my4DTrack;


}



void MuonAlignment::endJob(void) {


    LogTrace(metname)<<"[MuonAlignment] Saving the histos";
    bool outputMEsInRootFile = parameters.getParameter<bool>("OutputMEsInRootFile");
    std::string outputFileName = parameters.getParameter<std::string>("OutputFileName");

    edm::LogInfo("MuonAlignment")  << "Number of Tracks considered for residuals: " << numberOfTracks << std::endl << std::endl;
    edm::LogInfo("MuonAlignment")  << "Number of Hits considered for residuals: " << numberOfHits << std::endl << std::endl;

    if (doSummary){
        char binLabel[15];

	// check if ME still there (and not killed by MEtoEDM for memory saving)
	if( dbe )
	  {
	    // check existence of first histo in the list
	    if (! dbe->get(topFolder.str()+"/DT/hLocalPositionDT")) return;
	  }
	else 
	  return;
	

        for(unsigned int i=0 ; i<unitsLocalX.size() ; i++)
        {

            if(unitsLocalX[i]->getEntries()!=0){

                TString nameHistoLocalX=unitsLocalX[i]->getName();

                TString nameHistoLocalPhi=unitsLocalPhi[i]->getName();

                TString nameHistoLocalTheta=unitsLocalTheta[i]->getName();

                TString nameHistoLocalY=unitsLocalY[i]->getName();


                if (nameHistoLocalX.Contains("MB") ) // HistoLocalX DT
                {
                    int wheel, station, sector;

                    sscanf(nameHistoLocalX, "ResidualLocalX_W%dMB%1dS%d",&wheel,&station,&sector);

                    Int_t nstation=station - 1;
                    Int_t nwheel=wheel+2;

                    Double_t Mean=unitsLocalX[i]->getMean();
                    Double_t Error=unitsLocalX[i]->getMeanError();

                    Int_t ybin=1+nwheel*8+nstation*2;
                    hLocalPositionDT->setBinContent(sector,ybin,fabs(Mean));
                    snprintf(binLabel, sizeof(binLabel), "MB%d/%d_X",wheel, station );
                    hLocalPositionDT->setBinLabel(ybin,binLabel,2);
                    hLocalPositionRmsDT->setBinContent(sector,ybin,Error);
                    hLocalPositionRmsDT->setBinLabel(ybin,binLabel,2);
		
                    hLocalXMeanDT->Fill(fabs(Mean));
                    hLocalXRmsDT->Fill(Error);
                }

                if (nameHistoLocalX.Contains("ME")) // HistoLocalX CSC
                {
                    int station, ring, chamber;

                    sscanf(nameHistoLocalX, "ResidualLocalX_ME%dR%1dC%d",&station,&ring,&chamber);

                    Double_t Mean=unitsLocalX[i]->getMean();
                    Double_t Error=unitsLocalX[i]->getMeanError();

                    Int_t ybin=abs(station)*2+ring;
                    if(abs(station)==1) ybin=ring;
                    if (station>0) ybin=ybin+10;
                    else ybin = 11 -ybin;
                    ybin=2*ybin-1;
                    hLocalPositionCSC->setBinContent(chamber,ybin,fabs(Mean));
                    snprintf(binLabel, sizeof(binLabel), "ME%d/%d_X", station,ring );
                    hLocalPositionCSC->setBinLabel(ybin,binLabel,2);
                    hLocalPositionRmsCSC->setBinContent(chamber,ybin,Error);
                    hLocalPositionRmsCSC->setBinLabel(ybin,binLabel,2);

                    hLocalXMeanCSC->Fill(fabs(Mean));
                    hLocalXRmsCSC->Fill(Error);
                }

                if (nameHistoLocalTheta.Contains("MB")) // HistoLocalTheta DT
                {	

                    int wheel, station, sector;

                    sscanf(nameHistoLocalTheta, "ResidualLocalTheta_W%dMB%1dS%d",&wheel,&station,&sector);

                    if(station != 4){
                        Int_t nstation=station - 1;
                        Int_t nwheel=wheel+2;

                        Double_t Mean=unitsLocalTheta[i]->getMean();
                        Double_t Error=unitsLocalTheta[i]->getMeanError();

                        Int_t ybin=2+nwheel*8+nstation*2;
                        hLocalAngleDT->setBinContent(sector,ybin,fabs(Mean));
                        snprintf(binLabel, sizeof(binLabel), "MB%d/%d_#theta",wheel,station );
                        hLocalAngleDT->setBinLabel(ybin,binLabel,2);
                        hLocalAngleRmsDT->setBinContent(sector,ybin,Error);
                        hLocalAngleRmsDT->setBinLabel(ybin,binLabel,2);
	
                        hLocalThetaMeanDT->Fill(fabs(Mean));
                        hLocalThetaRmsDT->Fill(Error);
                    }
                }

                if (nameHistoLocalPhi.Contains("MB")) // HistoLocalPhi DT
                {	

                    int wheel, station, sector;

                    sscanf(nameHistoLocalPhi, "ResidualLocalPhi_W%dMB%1dS%d",&wheel,&station,&sector);

                    Int_t nstation=station - 1;
                    Int_t nwheel=wheel+2;

                    Double_t Mean=unitsLocalPhi[i]->getMean();
                    Double_t Error=unitsLocalPhi[i]->getMeanError();

                    Int_t ybin=1+nwheel*8+nstation*2;
                    hLocalAngleDT->setBinContent(sector,ybin,fabs(Mean));
                    snprintf(binLabel, sizeof(binLabel), "MB%d/%d_#phi", wheel,station );
                    hLocalAngleDT->setBinLabel(ybin,binLabel,2);
                    hLocalAngleRmsDT->setBinContent(sector,ybin,Error);
                    hLocalAngleRmsDT->setBinLabel(ybin,binLabel,2);

                    hLocalPhiMeanDT->Fill(fabs(Mean));
                    hLocalPhiRmsDT->Fill(Error);
                }

                if (nameHistoLocalPhi.Contains("ME")) // HistoLocalPhi CSC
                {

                    int station, ring, chamber;

                    sscanf(nameHistoLocalPhi, "ResidualLocalPhi_ME%dR%1dC%d",&station,&ring,&chamber);

                    Double_t Mean=unitsLocalPhi[i]->getMean();
                    Double_t Error=unitsLocalPhi[i]->getMeanError();

                    Int_t ybin=abs(station)*2+ring;
                    if(abs(station)==1) ybin=ring;
                    if (station>0) ybin=ybin+10;
                    else ybin = 11 -ybin;
                    ybin=2*ybin-1;
                    hLocalAngleCSC->setBinContent(chamber,ybin,fabs(Mean));
                    snprintf(binLabel, sizeof(binLabel), "ME%d/%d_#phi", station,ring );
                    hLocalAngleCSC->setBinLabel(ybin,binLabel,2);
                    hLocalAngleRmsCSC->setBinContent(chamber,ybin,Error);
                    hLocalAngleRmsCSC->setBinLabel(ybin,binLabel,2);

                    hLocalPhiMeanCSC->Fill(fabs(Mean));
                    hLocalPhiRmsCSC->Fill(Error);
                }

                if (nameHistoLocalTheta.Contains("ME")) // HistoLocalTheta CSC
                {

                    int station, ring, chamber;

                    sscanf(nameHistoLocalTheta, "ResidualLocalTheta_ME%dR%1dC%d",&station,&ring,&chamber);

                    Double_t Mean=unitsLocalTheta[i]->getMean();
                    Double_t Error=unitsLocalTheta[i]->getMeanError();

                    Int_t ybin=abs(station)*2+ring;
                    if(abs(station)==1) ybin=ring;
                    if (station>0) ybin=ybin+10;
                    else ybin = 11 -ybin;
                    ybin=2*ybin;
                    hLocalAngleCSC->setBinContent(chamber,ybin,fabs(Mean));
                    snprintf(binLabel, sizeof(binLabel), "ME%d/%d_#theta", station,ring );
                    hLocalAngleCSC->setBinLabel(ybin,binLabel,2);
                    hLocalAngleRmsCSC->setBinContent(chamber,ybin,Error);
                    hLocalAngleRmsCSC->setBinLabel(ybin,binLabel,2);

                    hLocalThetaMeanCSC->Fill(fabs(Mean));
                    hLocalThetaRmsCSC->Fill(Error);

                }

                if (nameHistoLocalY.Contains("MB")) // HistoLocalY DT
                {

                    int wheel, station, sector;

                    sscanf(nameHistoLocalY, "ResidualLocalY_W%dMB%1dS%d",&wheel,&station,&sector);

                    if(station!=4){
                        Int_t nstation=station - 1;
                        Int_t nwheel=wheel+2;

                        Double_t Mean=unitsLocalY[i]->getMean();
                        Double_t Error=unitsLocalY[i]->getMeanError();

                        Int_t ybin=2+nwheel*8+nstation*2;
                        hLocalPositionDT->setBinContent(sector,ybin,fabs(Mean));
                        snprintf(binLabel, sizeof(binLabel), "MB%d/%d_Y", wheel,station );
                        hLocalPositionDT->setBinLabel(ybin,binLabel,2);
                        hLocalPositionRmsDT->setBinContent(sector,ybin,Error);
                        hLocalPositionRmsDT->setBinLabel(ybin,binLabel,2);

                        hLocalYMeanDT->Fill(fabs(Mean));
                        hLocalYRmsDT->Fill(Error);
                    }
                }

                if (nameHistoLocalY.Contains("ME")) // HistoLocalY CSC
                {

                    int station, ring, chamber;

                    sscanf(nameHistoLocalY, "ResidualLocalY_ME%dR%1dC%d",&station,&ring,&chamber);

                    Double_t Mean=unitsLocalY[i]->getMean();
                    Double_t Error=unitsLocalY[i]->getMeanError();

                    Int_t ybin=abs(station)*2+ring;
                    if(abs(station)==1) ybin=ring;
                    if (station>0) ybin=ybin+10;
                    else ybin = 11 -ybin;
                    ybin=2*ybin;
                    hLocalPositionCSC->setBinContent(chamber,ybin,fabs(Mean));
                    snprintf(binLabel, sizeof(binLabel), "ME%d/%d_Y", station,ring );
                    hLocalPositionCSC->setBinLabel(ybin,binLabel,2);
                    hLocalPositionRmsCSC->setBinContent(chamber,ybin,Error);
                    hLocalPositionRmsCSC->setBinLabel(ybin,binLabel,2);

                    hLocalYMeanCSC->Fill(fabs(Mean));
                    hLocalYRmsCSC->Fill(Error);
                }
            } // check in # entries
        } // loop on vector of histos
    } //doSummary
    
    if(outputMEsInRootFile){
//    dbe->showDirStructure();
        dbe->save(outputFileName);
    }


}

