#include <RecoMuon/L3TrackFinder/interface/MuonRoadTrajectoryBuilder.h>

/** \file RecoMuon/
 *
 *  $Date: 2011/12/22 21:12:08 $
 *  $Revision: 1.5 $
 *  \author Adam Evertt, Jean-Roch Vlimant
 */



#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <DataFormats/GeometrySurface/interface/TkRotation.h>
#include <DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h>
#include <DataFormats/GeometryVector/interface/LocalPoint.h>
#include <DataFormats/GeometryVector/interface/LocalVector.h>
#include <DataFormats/GeometrySurface/interface/BoundPlane.h>
#include <DataFormats/GeometrySurface/interface/OpenBounds.h>
#include <DataFormats/GeometrySurface/interface/SimpleDiskBounds.h>

//#include <DataFormats/GeometrySurface/interface/LocalError.h> //does not exist anymore...
#include <DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h>
#include <DataFormats/DetId/interface/DetId.h>

#include <TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h>
#include <TrackingTools/MeasurementDet/interface/MeasurementDet.h>
#include <TrackingTools/KalmanUpdators/interface/KFUpdator.h>

//#include <RecoMuon/TrackingTools/interface/VertexRecHit.h>
#include <TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h>

#include <RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h>
#include <RecoTracker/Record/interface/CkfComponentsRecord.h>

#include <DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h>
#include <DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h>

//alternative constructor
MuonRoadTrajectoryBuilder::MuonRoadTrajectoryBuilder(const edm::ParameterSet & iConfig,
						 const MeasurementTracker * mt,
						 const MagneticField * f,
						 const Propagator * p):
  theRoadEstimator(0),theHitEstimator(0),theUpdator(0),theSmoother(0)
{

  theMeasurementTracker = mt;
  theField = f;
  thePropagator = p;

  theCategory = "Muon|RecoMuon|MuonRoadTrajectoryBuilder";

  //get parameters from ParameterSet

  //propagator name to get from muon service
  //  thePropagatorName = iConfig.getParameter<std::string>("propagatorName");

  //chi2 estimator
  double chi2R=iConfig.getParameter<double>("maxChi2Road");
  theRoadEstimator = new Chi2MeasurementEstimator(chi2R,sqrt(chi2R));
  double chi2H=iConfig.getParameter<double>("maxChi2Hit");
  theHitEstimator = new Chi2MeasurementEstimator(chi2H,sqrt(chi2H));

  //trajectory updator
  theUpdator= new KFUpdator();

  //limit the total number of possible trajectories taken into account for a single seed
  theMaxTrajectories = iConfig.getParameter<unsigned int>("maxTrajectories");

  //limit the type of module considered to gather rechits
  theDynamicMaxNumberOfHitPerModule = iConfig.getParameter<bool>("dynamicMaxNumberOfHitPerModule");
  theNumberOfHitPerModuleDefault = iConfig.getParameter<unsigned int>("numberOfHitPerModule");
  theMaxTrajectoriesThreshold = iConfig.getParameter<std::vector<unsigned int> >("maxTrajectoriesThreshold");
  theNumberOfHitPerModuleThreshold = iConfig.getParameter<std::vector<unsigned int> >("numberOfHitPerModuleThreshold");

  //could be configurable, but not
  theBranchonfirstlayer=true;
  theCarriedIPatfirstlayer=false;
  theCarriedIPatfirstlayerModule =true;

  //output track candidate selection
  theMinNumberOfHitOnCandidate = iConfig.getParameter<unsigned int>("minNumberOfHitOnCandidate");
  
  //single or multiple trajectories per muon
  theOutputAllTraj = iConfig.getParameter<bool>("outputAllTraj");

  if (!theSmoother)
    theSmoother = new KFTrajectorySmoother(thePropagator,theUpdator,theRoadEstimator);
}

//destructor
MuonRoadTrajectoryBuilder::~MuonRoadTrajectoryBuilder()
{
  edm::LogInfo(theCategory)<<"cleaning the object";
  if (theRoadEstimator) delete theRoadEstimator;
  if (theHitEstimator) delete theHitEstimator;
  if (theUpdator) delete theUpdator;
  if (theSmoother) delete theSmoother;
}

void MuonRoadTrajectoryBuilder::setEvent(const edm::Event & iEvent) const {
  theMeasurementTracker->update(iEvent);
}




bool trajectoryOrder(const MuonRoadTrajectoryBuilder::trajectory & traj1, const MuonRoadTrajectoryBuilder::trajectory & traj2)
{ //order so that first element of the list have
  //most hits, if equal smallest chi2
  unsigned int s1=traj1.measurements.size();
  unsigned int s2=traj2.measurements.size();
  if (s1>s2) return true; //decreasing order of measurements.size()
  else {
      if (s1==s2) return (traj1.chi2<traj2.chi2); //increase order of chi2
      else return false; }}


void MuonRoadTrajectoryBuilder::trajectories(const TrajectorySeed & seed,  TrajectoryContainer &ret) const
{
  LogDebug(theCategory)<<"makeTrajectories START";
  //process the seed through the tracker
  makeTrajectories(seed,ret);
}

//reconstruct trajectory for the trajectory seed
std::vector<Trajectory> MuonRoadTrajectoryBuilder::trajectories(const TrajectorySeed & seed) const 
{
  LogDebug(theCategory)<<"makeTrajectories START";
  
  //default output
  std::vector<Trajectory> result;
  
  //process the seed through the tracker
  makeTrajectories(seed,result);
  
  //output the result of regional tracking
  return result;
}



void MuonRoadTrajectoryBuilder::makeTrajectories(const TrajectorySeed & seed, std::vector<Trajectory> & result,int version) const
{  if (version==0) { makeTrajectories_0(seed,result);}
  else if (version==1) { makeTrajectories_1(seed,result);}}


//home grown tools
//old implementation
void MuonRoadTrajectoryBuilder::makeTrajectories_0(const TrajectorySeed & seed, std::vector<Trajectory> & result) const 
{
  Trajectory basicTrajectory(seed,alongMomentum);
  //add the muon system measurement to the basicTrajectory
  //...no yet implemented...

  //get the initial state
  PTrajectoryStateOnDet   PTstart=seed.startingState();
  DetId detId(PTstart.detId());
  const BoundPlane * surface =(&theMeasurementTracker->geomTracker()->idToDet(detId)->surface());
  //start from this point
  TrajectoryStateOnSurface TSOS = trajectoryStateTransform::transientState(PTstart,surface,theField);
  if (!TSOS.isValid())/*/abort*/{ edm::LogError(theCategory)<<"TSOS from PTSOD is not valid.";return ;}
  
  LogDebug(theCategory) <<"(detId.rawId()) "<<(detId.rawId())<<"(detId.subdetId()) "<<(detId.subdetId())
			<<"(TSOS.globalPosition()) "<<(TSOS.globalPosition())
			<<"(TSOS.globalMomentum()) "<<(TSOS.globalMomentum());
 
  //initialization
  //----------------------
  //  WARNING. this is mutable 
  //  you should never remove this line
  theFirstlayer=true;
  theNumberOfHitPerModule = theNumberOfHitPerModuleDefault;
  //----------------------
  int Nhits=0;
  GlobalPoint position;
  double z=0,r=0;
  //flipping pair of list of trajectories
  TrajectoryCollectionFPair Trajectories; 
  //initialize the head() of pair of list of trajectories
  Trajectories.head().push_back(trajectory());
  trajectory & initial = *Trajectories.head().rbegin();
  initial.TSOS=TSOS;  
  initial.traj=basicTrajectory;      
  


  //get the DetLayer in the tracker
  std::vector<BarrelDetLayer*> tiblc = theMeasurementTracker->geometricSearchTracker()->tibLayers();
  std::vector<BarrelDetLayer*> toblc = theMeasurementTracker->geometricSearchTracker()->tobLayers();
  std::vector<ForwardDetLayer*> tidlc[2];
  tidlc[0]=theMeasurementTracker->geometricSearchTracker()->negTidLayers();
  tidlc[1]=theMeasurementTracker->geometricSearchTracker()->posTidLayers();
  std::vector<ForwardDetLayer*> teclc[2];
  teclc[0]=theMeasurementTracker->geometricSearchTracker()->negTecLayers();
  teclc[1]=theMeasurementTracker->geometricSearchTracker()->posTecLayers();
  const int Ntib=tiblc.size();
  const int Ntob=toblc.size();
  const int Ntid=tidlc[0].size();


  const int Ntec=teclc[0].size();

  LogDebug(theCategory)<<"(Ntib) "<<(Ntib)<<"(Ntob) "<<(Ntob)<<"(Ntid) "<<(Ntid)<<"(Ntec) "<<(Ntec);

  const SimpleDiskBounds * sdbounds=0;

  position = TSOS.globalPosition();
  z = position.z();
  r = position.perp();

  //select which part we are in
  enum PART { fault , PXB, PXF, TIB , TID , TOB , TEC};
  PART whichpart = fault;
  switch(detId.subdetId()){
  case 1: {whichpart=PXB;break;}
  case 2: {whichpart=PXF;break;}

  case 3: {whichpart=TIB;break;}
  case 4: {whichpart=TID;break;}
  case 5: {whichpart=TOB;break;}
  case 6: {whichpart=TEC;break;}}

  bool inapart=true;
  bool firstRound =true;
  int indexinpart=0;
  //loop while in a valid part of the tracker
  while(inapart){

      //check on the trajectory list and stuff
      if (!checkStep(Trajectories.head())) break;
      //................

    switch (whichpart){

    case fault: /*abort*/ {edm::LogError(theCategory)<<"something's wrong with the seed";return;}
    case PXB: /*abort*/ {edm::LogError(theCategory)<<"PXB no yet implemented";return;}
    case PXF: /*abort*/ {edm::LogError(theCategory)<<"PXF no yet implemented";return;}
	
      //-------------- into TIB----------------
    case TIB:{
      if (indexinpart==Ntib){/*you have reach the last layer of the TIB.let's go to the TOB*/whichpart = TOB; indexinpart=-1;break; }

      LogDebug(theCategory)<<"within TIB "<<indexinpart+1;
	
      //propagate to corresponding surface
      if (!firstRound) TSOS = thePropagator->propagate(*TSOS.freeState(),tiblc[indexinpart]->surface());
      if (!TSOS.isValid()) {;break;} //go to the next one 
	    
      z=TSOS.globalPosition().z();

      //have we reached a boundary
      if (fabs(z) > fabs(tidlc[(z>0)][0]->surface().position().z())) 
	{/*z bigger than the TID min z: go to TID*/
	  LogDebug(theCategory)<<"|z| ("<<z<<") bigger than the TID min z("<<tidlc[(z>0)][0]->surface().position().z()<<"): go to TID";
	  whichpart = TID; indexinpart=-1; break;}
      else  {/*gather hits in the corresponding TIB layer*/ Nhits+=GatherHits(TSOS,tiblc[indexinpart],Trajectories);}
      break;}

      //-------------- into TID----------------
    case TID: {
      if (indexinpart==Ntid){/*you have reach the last layer of the TID. let's go to the TEC */ whichpart = TEC; indexinpart=-1; break;}

      LogDebug(theCategory)<<"within TID "<<indexinpart+1;

      //propagate to corresponding surface
      if (!firstRound) TSOS = thePropagator->propagate(*TSOS.freeState(),tidlc[(z>0)][indexinpart]->surface());
      if (!TSOS.isValid()){break;}//go to the next one 
	  
      position =  TSOS.globalPosition();
      z = position.z();
      r = position.perp();

      sdbounds = dynamic_cast<const SimpleDiskBounds *>(&tidlc[(z>0)][indexinpart]->surface().bounds());
      if (!sdbounds)/*abort*/{edm::LogError(theCategory)<<" detlayer bounds are not SimpleDiskBounds in tid geometry";return;}
	
      //have we reached a boundary
      if (r < sdbounds->innerRadius())
	{/*radius smaller than the TID disk inner radius: next disk please*/
	  LogDebug(theCategory)<<"radius ("<<r<<") smaller than the TID disk inner radius ("<<sdbounds->innerRadius()<<"): next disk please";
	  break;}
      else if (r >sdbounds->outerRadius())
	{/*radius bigger than the TID disk outer radius: go to TOB*/
	  LogDebug(theCategory)<<"radius ("<<r<<") bigger than the TID disk outer radius("<<sdbounds->outerRadius()<<"): go to TOB";
	  whichpart = TOB; indexinpart=-1;break;}
      else  {/*gather hits in the corresponding TIB layer*/
	LogDebug(theCategory)<<"collecting hits";
	Nhits+=GatherHits(TSOS,tidlc[(z>0)][indexinpart],Trajectories);}
      break;}

      //-------------- into TOB----------------
    case TOB: {
      if (indexinpart==Ntob){/*you have reach the last layer of the TOB. this is an end*/ inapart=false;break;}

      LogDebug(theCategory)<<"within TOB "<<indexinpart+1;
	
      //propagate to corresponding surface
      if (!firstRound) TSOS = thePropagator->propagate(*TSOS.freeState(),toblc[indexinpart]->surface());
      if (!TSOS.isValid())  {break;} //go to the next one 
	    
      z =  TSOS.globalPosition().z();
	
      //have we reached a boundary
      if (fabs(z) > fabs(teclc[(z>0)][0]->surface().position().z()))
	{/*z bigger than the TOB layer max z: go to TEC*/
	  LogDebug(theCategory)<<"|z| ("<<z<<") bigger than the TOB layer max z ("<< teclc[(z>0)][0]->surface().position().z()<<"): go to TEC";
	  whichpart = TEC; indexinpart=-1;break;}
      else {/*gather hits in the corresponding TOB layer*/Nhits+=GatherHits(TSOS,toblc[indexinpart],Trajectories);}
      break;}

      //-------------- into TEC----------------
    case TEC:	{
      if (indexinpart==Ntec){/*you have reach the last layer of the TEC. let's end here*/inapart=false;break;}
	  
      LogDebug(theCategory)<<"within TEC "<<indexinpart+1;
	  
      //propagate to corresponding TEC disk
      if (!firstRound) TSOS = thePropagator->propagate(*TSOS.freeState(),teclc[(z>0)][indexinpart]->surface());
      if (!TSOS.isValid()) {break;} //go to the next one
	  
      position = TSOS.globalPosition();
      z = position.z();
      r = position.perp();
	  
      sdbounds = dynamic_cast<const SimpleDiskBounds *>(&teclc[(z>0)][indexinpart]->surface().bounds());
      if (!sdbounds)/*abort*/ {edm::LogError(theCategory)<<" detlayer bounds are not SimpleDiskBounds in tec geometry";return;}
	  
      //have we reached a boundary ?
      if (r < sdbounds->innerRadius())
	{/*radius smaller than the TEC disk inner radius: next disk please*/
	  LogDebug(theCategory)<<"radius ("<<r<<") smaller than the TEC disk inner radius ("<<sdbounds->innerRadius()<<"): next disk please";
	  break;}
      else if (r > sdbounds->outerRadius())
	{/*radius bigger than the TEC disk outer radius: I can stop here*/
	  LogDebug(theCategory)<<"radius ("<<r<<") bigger than the TEC disk outer radius ("<<sdbounds->outerRadius()<<"): I can stop here";
	  inapart=false;break;}
      else {/*gather hits in the corresponding TEC layer*/Nhits+=GatherHits(TSOS,teclc[(z>0)][indexinpart],Trajectories);}
	  
      break;}

    }//switch
    indexinpart++;
    firstRound=false;
  }//while inapart


  LogDebug(theCategory)<<"propagating through layers is done";
  
  //--------------------------------------- SWIMMING DONE --------------------------------------
  //the list of trajectory has been build. let's find out which one is the best to use.
  
  edm::LogInfo(theCategory)<<"found: "
				      <<Nhits<<" hits in the road for this seed \n"
				      << Trajectories.head().size()<<" possible trajectory(ies)";
  
  if (Trajectories.head().empty()) /*abort*/{edm::LogError(theCategory)<<" no possible trajectory found"; return;}
  
  //order the list to put the best in front
  Trajectories.head().sort(trajectoryOrder);
  
  //------------------------------------OUTPUT FILL---------------------------------------------
  if (theOutputAllTraj) {
    //output all the possible trajectories found, if they have enough hits on them
    //the best is still the first
    for (TrajectoryCollection::iterator tit = Trajectories.head().begin(); tit!=Trajectories.head().end();tit++)
      {if (tit->measurements.size()>= theMinNumberOfHitOnCandidate){result.push_back(smooth(tit->traj));}}
  }
  else{
    //output only the best one
    trajectory & best = (*Trajectories.head().begin());
    if (best.measurements.size()< theMinNumberOfHitOnCandidate)/*abort*/{edm::LogError(theCategory)<<"best trajectory does not have enough ("<<best.measurements.size()<<") hits on it (<"<<theMinNumberOfHitOnCandidate<<")"; return;}
    //output only the best trajectory
    result.push_back(smooth(best.traj));}
  
  edm::LogInfo(theCategory)<<result.size()<<" trajectory(ies) output";
}//makeTrajectories_0


bool trajectorymeasurementOrder(const TrajectoryMeasurement & meas_1 ,const TrajectoryMeasurement & meas_2 )
 {
   //   GlobalVector d= (meas_2.recHit()->globalPosition () - meas_1.recHit()->globalPosition ()).unit();
   GlobalVector d= (meas_2.predictedState().globalPosition() - meas_1.predictedState().globalPosition()).unit();

   GlobalVector u1= meas_1.predictedState().globalDirection().unit(); //was using only that before
   GlobalVector u2= meas_2.predictedState().globalDirection().unit();
   GlobalVector u =(u1+u2)/2.; //the average momentum to be sure that the relation is symetrical.
   return   d.dot(u) > 0;
 }
bool trajectorymeasurementInverseOrder(const TrajectoryMeasurement & meas_1 ,const TrajectoryMeasurement & meas_2 )
 {return trajectorymeasurementOrder(meas_2,meas_1);}



 void MuonRoadTrajectoryBuilder::cleanTrajectory(Trajectory & traj) const {
   //remove the overlapping recHits since the smoother will chock on it
   Trajectory::DataContainer meas =traj.measurements();
   
   const DetLayer* lastDetLayer =0;
   Trajectory::DataContainer::iterator mit = meas.begin();
   while (mit!=meas.end()){
     {
       const DetLayer * detLayer = theMeasurementTracker->geometricSearchTracker()->detLayer(mit->recHit()->det()->geographicalId());
       LogDebug(theCategory)<<"(mit->recHit()->det()->geographicalId().rawId()) "<<(mit->recHit()->det()->geographicalId().rawId())
			    <<"(detLayer) "<<(detLayer)<<"(lastDetLayer) "<<(lastDetLayer);

       if (detLayer==lastDetLayer)
	 {
	   mit=meas.erase(mit); //serve as mit++ too
	 }
       else {mit++;}
       lastDetLayer=detLayer;
     }
     Trajectory newTraj(traj.seed(),traj.direction());
     for (Trajectory::DataContainer::iterator mit=meas.begin();mit!=meas.end();++mit)
       {newTraj.push(*mit);}
     traj=newTraj;
   }
 }


 void sortTrajectoryMeasurements(Trajectory & traj){
  Trajectory::DataContainer meas =traj.measurements();
  
  //  edm::LogInfo(theCategory)<<"sorting ("<< meas.size() <<") measurements.";
  //sort the measurements
  if (traj.direction() == oppositeToMomentum)
    {std::sort(meas.begin(),meas.end(),trajectorymeasurementInverseOrder);}
  else
    {std::sort(meas.begin(),meas.end(),trajectorymeasurementOrder);}

  //create a new one
  Trajectory newTraj(traj.seed(),traj.direction());
  for (Trajectory::DataContainer::iterator mit=meas.begin();mit!=meas.end();++mit)
    {newTraj.push(*mit);}

  //exchange now.
  traj = newTraj;
 }

Trajectory MuonRoadTrajectoryBuilder::smooth(Trajectory & traj) const {
  
  //need to order the list of measurements on the trajectory first
  sortTrajectoryMeasurements(traj);

  std::vector<Trajectory> ret=theSmoother->trajectories(traj);

  if (ret.empty()){
    edm::LogError(theCategory)<<"smoother returns an empty vector of trajectories: try cleaning first and resmooth";
    cleanTrajectory(traj);
    ret=theSmoother->trajectories(traj);
  }
  
  if (ret.empty()){edm::LogError(theCategory)<<"smoother returns an empty vector of trajectories.";
    return traj;}
  else{return (ret.front());}
}

//function called for each layer during propagation
int  MuonRoadTrajectoryBuilder::GatherHits( const TrajectoryStateOnSurface & step,const DetLayer * thislayer , TrajectoryCollectionFPair & Trajectories) const 
{
  TrajectoryStateOnSurface restep;
  bool atleastoneadded=false;
  int Nhits=0;

  //find compatible modules
  std::vector<DetLayer::DetWithState>  compatible =thislayer->compatibleDets( step, *thePropagator , *theRoadEstimator);
  
  //loop over compatible modules
  for (std::vector< DetLayer::DetWithState > ::iterator dws = compatible.begin();dws != compatible.end();dws++)
    {
      const DetId presentdetid = dws->first->geographicalId();//get the det Id
      restep = dws->second;


	LogDebug(theCategory)<<((dws->first->components ().size()!=0) ? /*stereo layer*/"double sided layer":/*single sided*/"single sided layer")
			     <<"(presentdetid.rawId()) "<<(presentdetid.rawId());

      //get the rechits on this module
      TransientTrackingRecHit::ConstRecHitContainer  thoseHits = theMeasurementTracker->idToDet(presentdetid)->recHits(restep); 
      int additionalHits =0;

      LogDebug(theCategory)<<"(thoseHits.size()) "<<(thoseHits.size());

      if (thoseHits.size()>theNumberOfHitPerModule)
	{ edm::LogInfo(theCategory)<<"more than "<<theNumberOfHitPerModule 
					   <<" compatible hits ("<<thoseHits.size()<<")on module "<<presentdetid.rawId()
					   <<", skip it";continue; }

      //loop over the rechit on the module
      for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator iTThit = thoseHits.begin();iTThit != thoseHits.end(); iTThit++)
	{
	  if (!(*iTThit)->isValid())/*skip*/{edm::LogInfo(theCategory)<<"rec hit not valid on module "<<presentdetid.rawId() <<". I skip it"; continue;}

	  LogDebug(theCategory)<<((dynamic_cast<const SiStripMatchedRecHit2D *>((*iTThit)->hit())!=0) ? /*matched rechit*/ "matched rechit" : " r-phi rechit");

	  //get the surface of the module Id
	  const BoundPlane & surface = (*iTThit)->det()->surface();

	  //estimate the consistency
	  MeasurementEstimator::HitReturnType est_road = theRoadEstimator->estimate(restep,**iTThit);

	  LogDebug(theCategory)<<"(restep.globalPosition().perp()) "<<(restep.globalPosition().perp())<<"(restep.globalPosition().mag()) "<<(restep.globalPosition().mag())
			     <<"road step parameters at module: \n"<<restep
			       << "(est_road.first) "<<(est_road.first)<<"(est_road.second) "<<(est_road.second);

	  //check consistency
	  if (est_road.first)
	    { //hit and propagation are consistent : gather the hit
	      Nhits++;
	      additionalHits++;
	      atleastoneadded=true;
	      
	      LogDebug(theCategory)<<"hit is consistent with road"<<"(presentdetid.rawId()) "<<(presentdetid.rawId())
				 <<"loop over previous trajectories\n"
				   <<"(Trajectories.tail().size()) "<<(Trajectories.tail().size())<<"(Trajectories.head().size()) "<<(Trajectories.head().size());
		
	      //update the list of trajectory that we have with the consistent rechit.
	      //loop over the existing list of trajectories and add the hit if necessary
	      for ( TrajectoryCollection::iterator traj =Trajectories.head().begin();traj != Trajectories.head().end(); traj++)
		{
		  //what is the previous state for this trajectory
		  const TrajectoryStateOnSurface & previousState = traj->TSOS;
		  if (!previousState.isValid())
		    {edm::LogError(theCategory)<<"previous free state is invalid at trajectory update, this is WRONG"; continue;}
		  const FreeTrajectoryState & previousFreestate = *previousState.freeState();
		  
		  //propagate it to the current surface
		  TrajectoryStateOnSurface predictedState = thePropagator->propagate(previousFreestate,surface);
		  if (!predictedState.isValid())/*skip*/{edm::LogError(theCategory)
			<<"predicted state is not valid at trajectory update, rechit surface cannot be reached by the previous updated state.";continue;}

		  MeasurementEstimator::HitReturnType est ;
		  est= theHitEstimator->estimate(predictedState,**iTThit);

		  LogDebug(theCategory)<<"(est.first) "<<(est.first)<<"(est.second) "<<(est.second);

		  //is the current hit consistent with the trajectory
		  if (est.first )
		    {
		      //update the trajectory state with the rechit
		      const TrajectoryStateOnSurface & updatedState = theUpdator->update(predictedState,**iTThit);
		      if (!updatedState.isValid())/*skip*/{edm::LogError(theCategory)<<"updated state is not valid, this is really wrong";continue;}

		      LogDebug(theCategory)<<"updating a previous state with a rechit";

		      //add a combined trajectory to the new list of trajectory, starting from the existing trajecotry
		      Trajectories.tail().push_back(*traj); //from existing
		      trajectory & combined = (*Trajectories.tail().rbegin());
		      combined.duplicate =false; //this is important
		      //increment the chi2
		      //combined.chi2 += est.second;
		      combined.chi2 += est_road.second;//better to add the road chi2 too unbias the chi2 sum towards first hits
		      //some history about the trajectory
		      combined.lastmissed=false;
		      combined.missedinarow=0;
		      //add a new hits to the measurements
		      combined.measurements.push_back(TrajectoryMeasurement(updatedState,*iTThit));
		      TrajectoryMeasurement & trajMeasurement = (*combined.measurements.rbegin());
		      //assigne updated state
		      combined.TSOS = updatedState;
		      //add trajectory measurement 
		      combined.traj.push(trajMeasurement,est.second);

		      LogDebug(theCategory)<<"(combined.traj.foundHits()) "<<(combined.traj.foundHits())
					 <<"muon measurement on previous module: \n"<<traj->TSOS
					 <<"muon measurement before update: \n"<<predictedState
					 <<"muon measurement after update: \n"<<updatedState;

		    }//hit consistent with trajectory
		  else
		    {
		      LogDebug(theCategory)<<"hit failed chi2 test for trajectories update\n"<<"(traj->duplicate) "<<(traj->duplicate);

		      if (!traj->duplicate){
			Trajectories.tail().push_back(*traj);
			trajectory & copied = (*Trajectories.tail().rbegin());
			copied.missed++;
			copied.lastmissed=true;
			copied.missedinarow++;
			traj->duplicate =true; //set this trajectory to already have been copied into next step
		       }//not a duplicate trajectory
		    }//hit not consistent with trajectory
		}//existing trajectories loop
	    }//hit is consistent with muon road
	}//rechits on module loop
      
      
      //discard previous list of trajectories
      //if something as been done of course
      if (!Trajectories.tail().empty()) 
	{
	  //if this is not the first "layer" of detector, set updated trajectories as new seed trajectories
	  //will branch on every single rechit uncountered for first "layer"
	  if (!theFirstlayer || theBranchonfirstlayer)
	    {
	      LogDebug(theCategory)<<"swapping trajectory list index";
	      
	      //always carry on the <IP> alone state in the next list of trajectories to avoid bias from the first rechits
	      if (theFirstlayer && theCarriedIPatfirstlayerModule)
		{
		  LogDebug(theCategory)<<"push front <IP> to next list of trajectories";

		  Trajectories.tail().push_front(*Trajectories.head().begin()); //[0] is <IP> always
		  trajectory & pushed = *Trajectories.tail().begin();
		  pushed.missed+=additionalHits;
		  pushed.lastmissed = ( additionalHits!=0);
		  pushed.missedinarow++;
		}
	      //FIXME, there is a candidate leak at this point
	      //if the IP state is carried on at first layer, without update, then it will be duplicated. this is very unlikely though

	      Trajectories.head().clear();

	      //swap the lists
	      Trajectories.flip();
	    }
	}//discard previous list of trajectories
    }//module loop

  

  //do some cleaning of the list of trajectories
  if(theFirstlayer && atleastoneadded )
    {
      theFirstlayer =false; //we are not in the first layer if something has been added
      if (!theBranchonfirstlayer)
	{
	  LogDebug(theCategory)<<"swapping trajectory list index (end of first layer)";

	  //and for consistency, you have to swap index here, because it could ahve not been done in the module loop above
	  //always carry on the <IP> alone state in the next list of trajectories to avoid bias from the first rechits
	  if (theCarriedIPatfirstlayer)
	    {Trajectories.tail().push_front(*Trajectories.head().begin());} //[0] is <IP> always at this stage
	  //FIXME, there is a candidate leak at this point
	  //if the IP state is carried on at first layer, without update, then it will be duplicated. this is very unlikely though
	  
	  Trajectories.head().clear();
	  //swap the switch
	  Trajectories.flip();
	}
      else
	{
	  //actually remove the <IP> from first postion of the next source of trajectories
	  //since the swaping has already been done. pop up from theTrajectorysource, not !theTrajectorysource
	  //only if it has been done at the first layer module though
	  if (!theCarriedIPatfirstlayer && theCarriedIPatfirstlayerModule){

	    LogDebug(theCategory)<<"pop up <IP> from trajectories";

	    Trajectories.head().pop_front(); }
	}


      //check an remove trajectories that are subset of the other in next source
      //after the first layer only
      if (Trajectories.head().size()>=2)
	{checkDuplicate(Trajectories.head());}

    }  //do some cleaning of the list of trajectories
  return Nhits;

}


bool MuonRoadTrajectoryBuilder::checkStep(TrajectoryCollection & collection) const 
{
  //dynamic cut on the max number of rechit allowed on a single module
  if (theDynamicMaxNumberOfHitPerModule) {
    for (unsigned int vit = 0; vit!= theMaxTrajectoriesThreshold.size() ; vit++){
      if (collection.size() >theMaxTrajectoriesThreshold[vit]){
	theNumberOfHitPerModule= theNumberOfHitPerModuleThreshold[vit];}
      else
	break;}}
  
  //reduce the number of possible trajectories if too many
  if ( collection.size() > theMaxTrajectories) {
    //order with most hits or best chi2
    collection.sort(trajectoryOrder);
    unsigned int prevSize=collection.size();
    collection.resize(theMaxTrajectories);
    edm::LogInfo(theCategory)
      <<" too many possible trajectories ("<<prevSize
      <<"), reduce the possibilities to "<<theMaxTrajectories<<" bests.";
  }
  return true;
}

void MuonRoadTrajectoryBuilder::checkDuplicate(TrajectoryCollection & collection) const
{
  LogDebug(theCategory)<<"checking for duplicates here. list size: "<<collection.size();

  //loop over the trajectory list
  TrajectoryCollection::iterator traj1 = collection.begin();
  while(traj1 != collection.end())
    {
      LogDebug(theCategory)<<"";

      //reloop over the trajectory list from traj1
      TrajectoryCollection::iterator traj2 = traj1;
      traj2++;//advance one more
      bool traj1_removed=false;
      while( traj2 != collection.end())
	{
	  if (traj2 == traj1 ) continue; //skip itself of course

	  //need to start from the back of the list of measurment
	  std::list <TrajectoryMeasurement >::reverse_iterator h1 = traj1->measurements.rbegin();
	  std::list <TrajectoryMeasurement >::reverse_iterator h2 = traj2->measurements.rbegin();
		  
	  bool break_different = false;
	  while (h1 != traj1->measurements.rend() && h2!=traj2->measurements.rend())
	    {
	      TransientTrackingRecHit::ConstRecHitPointer hit1 = h1->recHit();
	      TransientTrackingRecHit::ConstRecHitPointer hit2 = h2->recHit();

	      LogDebug(theCategory)<<"(hit1->geographicalId().rawId()) "<<(hit1->geographicalId().rawId())<<"(hit1->globalPosition()) "<<(hit1->globalPosition())
				   <<"(hit2->geographicalId().rawId()) "<<(hit2->geographicalId().rawId())<<"(hit2->globalPosition()) "<<(hit2->globalPosition());

	      if (hit1 == hit2){/*those are common hits, everything's alright so far*/ h1++;h2++; continue;}
	      else{break_different =true;
			
		LogDebug(theCategory)<<"list of hits are different";
			
		break;}
	    }
	  if (!break_different) 
	    //meaning one of the list has been exhausted
	    //one list if the subset of the other
	    {
	      LogDebug(theCategory)<<"list of hits are identical/subset. remove one of them.";
	      //there is a common subset to the two list of rechits.
	      //remove the one with the fewer hits (iterator that reached the end first)
	      //in case they are exactly identical (both iterator reached the end at the same time), traj2 will be removed by default
	      if (h1 != traj1->measurements.rend())
		{
		  LogDebug(theCategory)<<"I removed traj2";
		  //traj2 has been exhausted first. remove it and place the iterator on next item
		  traj2=collection.erase(traj2);
		}
	      else
		{
		  LogDebug(theCategory)<<"I removed traj1. and decrement traj2";
		  //traj1 has been exhausted first. remove it
		  traj1=collection.erase(traj1);
		  //and set the iterator traj1 so that next++ will set it to the correct place in the list  
		  traj1_removed=true;
		  break; // break the traj2 loop, advance to next traj1 item
		}
	    }
	  else
	    {traj2++;   }
	}//loop on traj2
      if (!traj1_removed)
	{//increment only if you have remove the item at traj1
	  traj1++;}
    }//loop on traj1
}


//CTF tool implementation
//first find the collection of rechits, then do something about it
void MuonRoadTrajectoryBuilder::makeTrajectories_1(const TrajectorySeed & seed, std::vector<Trajectory> & result) const 
{
  Trajectory basicTrajectory(seed,alongMomentum);
  //add the muon system measurement to the basicTrajectory
  //...

//   //build the trajectories
//   std::vector<Trajectory> unsmoothed = theCkfbuilder->trajectories(seed);
  
//   //smoothed them
//   if (theOutputAllTraj) {
//     for (std::vector<Trajectory>::iterator tit = unsmoothed.begin(); tit!=unsmoothed.end();tit++)
//       {result.push_back(smooth(*tit));}
//   }
//   else{
//     //output only the first one
//     result.push_back(smooth(unsmoothed.front()));} 
    
}

