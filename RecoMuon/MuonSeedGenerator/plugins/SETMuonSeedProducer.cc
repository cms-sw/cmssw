/** \class SETMuonSeedProducer
    I. Bloch, E. James, S. Stoynev
 */

#include "RecoMuon/MuonSeedGenerator/plugins/SETMuonSeedProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"            
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"             
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"                

#include "TMath.h"

// there is an existing sorter somewhere in the CMSSW code (I think) - delete that
struct sorter{
  bool operator() (MuonTransientTrackingRecHit::MuonRecHitPointer hit_1,
                   MuonTransientTrackingRecHit::MuonRecHitPointer hit_2){
    //double GlobalPoint ss = hit_1->globalPosition
    //
    // globalPosition().mag()?
    double radius2_1 =
      pow(hit_1->globalPosition().x(),2) +
      pow(hit_1->globalPosition().y(),2) +
      pow(hit_1->globalPosition().z(),2);

    double radius2_2 =
      pow(hit_2->globalPosition().x(),2) +
      pow(hit_2->globalPosition().y(),2) +
      pow(hit_2->globalPosition().z(),2);
    return (radius2_1<radius2_2);
  }
} sortSegRadius;// smaller first

using namespace edm;
using namespace std;

SETMuonSeedProducer::SETMuonSeedProducer(const ParameterSet& parameterSet){

  const string metname = "Muon|RecoMuon|SETMuonSeedSeed";  
  //std::cout<<" The SET SEED"<<std::endl;

  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");
  theService        = new MuonServiceProxy(serviceParameters);

  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("SETTrajBuilderParameters");

  LogTrace(metname) << "constructor called" << endl;
  
  // load pT seed parameters
  thePtExtractor = new MuonSeedPtExtractor::MuonSeedPtExtractor(trajectoryBuilderParameters);

  apply_prePruning = trajectoryBuilderParameters.getParameter<bool>("Apply_prePruning");

  useSegmentsInTrajectory = trajectoryBuilderParameters.getParameter<bool>("UseSegmentsInTrajectory");

  // The inward-outward fitter (starts from seed state)
  ParameterSet filterPSet = trajectoryBuilderParameters.getParameter<ParameterSet>("FilterParameters");
  filterPSet.addUntrackedParameter("UseSegmentsInTrajectory", useSegmentsInTrajectory);
  theFilter = new SETFilter(filterPSet,theService);

  useRPCs = filterPSet.getParameter<bool>("EnableRPCMeasurement");

  DTRecSegmentLabel  = filterPSet.getParameter<edm::InputTag>("DTRecSegmentLabel");
  CSCRecSegmentLabel  = filterPSet.getParameter<edm::InputTag>("CSCRecSegmentLabel");
  RPCRecSegmentLabel  = filterPSet.getParameter<edm::InputTag>("RPCRecSegmentLabel");

  //----

  produces<TrajectorySeedCollection>();

} 

SETMuonSeedProducer::~SETMuonSeedProducer(){

  LogTrace("Muon|RecoMuon|SETMuonSeedProducer") 
    << "SETMuonSeedProducer destructor called" << endl;
  
  if(thePtExtractor) delete thePtExtractor;
  if(theFilter) delete theFilter;
}

void SETMuonSeedProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup){
  //std::cout<<" start producing..."<<std::endl;  
  const string metname = "Muon|RecoMuon|SETMuonSeedSeed";  

  MuonPatternRecoDumper debug;

  //Get the CSC Geometry :
  theService->update(eventSetup);

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  
  Handle<View<TrajectorySeed> > seeds; 

  setEvent(event);

  std::vector < std::pair <TrajectoryStateOnSurface , 
    TransientTrackingRecHit::ConstRecHitContainer > > setMeasurementContainer = trajectories(event);

  for(unsigned int iTraj = 0;iTraj<setMeasurementContainer.size();++iTraj){
    edm::OwnVector<TrackingRecHit> recHitsContainer;
    for(uint iHit = 0;iHit <setMeasurementContainer.at(iTraj).second.size();++iHit){
      recHitsContainer.push_back(setMeasurementContainer.at(iTraj).second.at(iHit)->hit()->clone());
    }
    TrajectoryStateOnSurface firstTSOS = setMeasurementContainer.at(iTraj).first; 
     PropagationDirection dir = oppositeToMomentum;
     if(useSegmentsInTrajectory){
       dir = alongMomentum;// why forward (for rechits) later?
     }
     TrajectoryStateTransform tsTransform;
       PTrajectoryStateOnDet *seedTSOS =
      tsTransform.persistentState( firstTSOS, setMeasurementContainer.at(iTraj).second.at(0)->geographicalId().rawId());
    TrajectorySeed seed(*seedTSOS,recHitsContainer,dir);    
    output->push_back(seed);
    TrajectorySeed::range range = seed.recHits();
    //std::cout<<" firstTSOS = "<<debug.dumpTSOS(firstTSOS)<<std::endl;
    //std::cout<<" iTraj = "<<iTraj<<" hits = "<<range.second-range.first<<std::endl;
    //std::cout<<" nhits = "<<setMeasurementContainer.at(iTraj).second.size()<<std::endl;
    for(unsigned int iRH=0;iRH<setMeasurementContainer.at(iTraj).second.size();++iRH){
      //std::cout<<" RH = "<<iRH+1<<" globPos = "<<setMeasurementContainer.at(iTraj).second.at(iRH)->globalPosition()<<std::endl;
    }
  }
  event.put(output);
}
std::vector < std::pair <TrajectoryStateOnSurface, 
			 TransientTrackingRecHit::ConstRecHitContainer > >  
SETMuonSeedProducer::trajectories(const edm::Event& event){

  const std::string metname = "Muon|RecoMuon|SETMuonSeedProducer";
  MuonPatternRecoDumper debug;

  // the measurements container. 
  std::vector < std::pair <TrajectoryStateOnSurface , 
    TransientTrackingRecHit::ConstRecHitContainer > > setMeasurementsContainer;

  std::vector < TrajectoryMeasurement > trajectoryMeasurementsFW;

  bool fwFitFailed = true;

  edm::ESHandle<GlobalTrackingGeometry> geomESH = theService->trackingGeometry();

  //---- Build collection of all segments
  MuonRecHitContainer muonRecHits;
  MuonRecHitContainer muonRecHits_DT2D_hasPhi;
  MuonRecHitContainer muonRecHits_DT2D_hasZed;
  MuonRecHitContainer muonRecHits_RPC;

  // ********************************************;
  // Get the DT-Segment collection from the Event
  // ********************************************;

  edm::Handle<DTRecSegment4DCollection> dtRecHits;
  event.getByLabel(DTRecSegmentLabel, dtRecHits);
  for (DTRecSegment4DCollection::const_iterator rechit = dtRecHits->begin(); rechit!=dtRecHits->end();++rechit) {
    if( (rechit->hasZed() && rechit->hasPhi()) ) {
    muonRecHits.push_back(MuonTransientTrackingRecHit::specificBuild(theService->trackingGeometry()->idToDet((*rechit).geographicalId()),&*rechit));
    }
    else if(rechit->hasZed()) {
    muonRecHits_DT2D_hasZed.push_back(MuonTransientTrackingRecHit::specificBuild(theService->trackingGeometry()->idToDet((*rechit).geographicalId()),&*rechit));
    }
    else if(rechit->hasPhi()) { // safeguard
    muonRecHits_DT2D_hasPhi.push_back(MuonTransientTrackingRecHit::specificBuild(theService->trackingGeometry()->idToDet((*rechit).geographicalId()),&*rechit));
    }
    else {
      std::cout<<"Warning in "<<metname<<": DT segment which claims to have neither phi nor Z."<<std::endl;
    }
  }
  //std::cout<<"DT done"<<std::endl;

  // ********************************************;
  // Get the CSC-Segment collection from the event
  // ********************************************;

  edm::Handle<CSCSegmentCollection> cscSegments;
  event.getByLabel(CSCRecSegmentLabel, cscSegments);
  for(CSCSegmentCollection::const_iterator rechit=cscSegments->begin(); rechit != cscSegments->end(); ++rechit) {
    muonRecHits.push_back(MuonTransientTrackingRecHit::specificBuild(theService->trackingGeometry()->idToDet((*rechit).geographicalId()),&*rechit));
  }
  //std::cout<<"CSC done"<<std::endl;

  // ********************************************;
  // Get the RPC-Hit collection from the event
  // ********************************************;

  edm::Handle<RPCRecHitCollection> rpcRecHits;
  event.getByLabel(RPCRecSegmentLabel, rpcRecHits);
  for(RPCRecHitCollection::const_iterator rechit=rpcRecHits->begin(); rechit != rpcRecHits->end(); ++rechit) {
    if(useRPCs){
      muonRecHits_RPC.push_back(MuonTransientTrackingRecHit::specificBuild(theService->trackingGeometry()->idToDet((*rechit).geographicalId()),&*rechit));
    }
  }
  //std::cout<<"RPC done"<<std::endl;

  //---- Find "pre-clusters" from all segments; these contain potential muon candidates

  std::vector< MuonRecHitContainer > MuonRecHitContainer_clusters;
  MuonRecHitContainer_clusters = clusterHits( muonRecHits,
                                              muonRecHits_DT2D_hasPhi,
                                              muonRecHits_DT2D_hasZed,
                                              muonRecHits_RPC);
  //std::cout<<"We have formed "<<MuonRecHitContainer_clusters.size()<<" clusters"<<std::endl;
  // for each cluster,
  for(unsigned int cluster = 0; cluster < MuonRecHitContainer_clusters.size(); ++cluster) {
    //std::cout<<" This is cluster number : "<<cluster<<std::endl;
     std::vector <seedSet> seedSets_inCluster; 

   //--- Sort segments (first will be ones with smaller distance to IP)
    sort(MuonRecHitContainer_clusters[cluster].begin(),MuonRecHitContainer_clusters[cluster].end(),sortSegRadius);

    //---- group hits in detector layers (if in same layer); the idea is that
    //---- some hits could not belong to a track simultaneously - these will be in a 
    //---- group; two hits from one and the same group will not go to the same track 
    std::vector< MuonRecHitContainer > MuonRecHitContainer_perLayer;
    if(MuonRecHitContainer_clusters[cluster].size()){
      int iHit =0;
      MuonRecHitContainer hitsInThisLayer;
      hitsInThisLayer.push_back(MuonRecHitContainer_clusters[cluster][iHit]);
      DetId  detId = MuonRecHitContainer_clusters[cluster][iHit]->hit()->geographicalId();
      const GeomDet* geomDet = theService->trackingGeometry()->idToDet( detId );
      while(iHit<int(MuonRecHitContainer_clusters[cluster].size())-1){
	DetId  detId_2 = MuonRecHitContainer_clusters[cluster][iHit+1]->hit()->geographicalId();
	const GlobalPoint gp_nextHit = MuonRecHitContainer_clusters[cluster][iHit+1]->globalPosition();

	// this is the distance of the "second" hit to the "first" detector (containing the "first hit")
	float distanceToDetector = fabs(geomDet->surface().localZ(gp_nextHit));

	//---- hits from DT and CSC  could be very close in angle but incosistent with 
	//---- belonging to a common track (and these are different surfaces);
	//---- also DT hits from a station (in a pre-cluster) should be always in a group together;  
	//---- take this into account and put such hits in a group together

	bool specialCase = ( detId.subdetId()   == MuonSubdetId::DT && 
                             detId_2.subdetId() == MuonSubdetId::DT    );



	if(specialCase){
          if(detId.subdetId() != MuonSubdetId::DT || detId_2.subdetId() != MuonSubdetId::DT) {
            std::cout<<"IBL ALARM 0000"<<std::endl;
          }
	  DTChamberId dtCh(detId);
	  DTChamberId dtCh_2(detId_2);
	  specialCase =  (dtCh.station() == dtCh_2.station());
	}
	if(distanceToDetector<0.001 || true==specialCase){ // hardcoded value - remove!
	  hitsInThisLayer.push_back(MuonRecHitContainer_clusters[cluster][iHit+1]);
	  
	}
	else{
	  specialCase = false;
	  if(( (MuonRecHitContainer_clusters[cluster][iHit]->isDT() &&
		MuonRecHitContainer_clusters[cluster][iHit+1]->isCSC()) ||
	       (MuonRecHitContainer_clusters[cluster][iHit]->isCSC() &&
		MuonRecHitContainer_clusters[cluster][iHit+1]->isDT())) &&
	     //---- what is the minimal distance between a DT and a CSC hit belonging
	     //---- to a common track? (well, with "reasonable" errors; put 10 cm for now) 
	     fabs(MuonRecHitContainer_clusters[cluster][iHit+1]->globalPosition().mag() -
		  MuonRecHitContainer_clusters[cluster][iHit]->globalPosition().mag())<10.){
	    hitsInThisLayer.push_back(MuonRecHitContainer_clusters[cluster][iHit+1]);
            // change to Stoyan - now we also update the detID here... give it a try. IBL 080905
	    detId = MuonRecHitContainer_clusters[cluster][iHit+1]->hit()->geographicalId();
	    geomDet = theService->trackingGeometry()->idToDet( detId );
	  }
	  else if(!specialCase){
	    //---- put the group of hits in the vector (containing the groups of hits) 
	    //---- and continue with next layer (group)	    
	    MuonRecHitContainer_perLayer.push_back(hitsInThisLayer);
	    hitsInThisLayer.clear();
	    hitsInThisLayer.push_back(MuonRecHitContainer_clusters[cluster][iHit+1]);
	    detId = MuonRecHitContainer_clusters[cluster][iHit+1]->hit()->geographicalId();
	    geomDet = theService->trackingGeometry()->idToDet( detId );
	  }
	}
	++iHit;
      }
      MuonRecHitContainer_perLayer.push_back(hitsInThisLayer);
    }
    //---- build all possible combinations (valid sets)
    std::vector <MuonRecHitContainer> allValidSets = findAllValidSets(MuonRecHitContainer_perLayer);
    if(apply_prePruning){
      //---- remove "wild" segments from the combination
      validSetsPrePruning(allValidSets);
    }
    
    //---- build the appropriate output: seedSets_inCluster
    //---- if too many (?) valid sets in a cluster - skip it 
    if(allValidSets.size()<500){// hardcoded - remove it
      seedSets_inCluster = fillSeedSets(allValidSets);
    }
    //// find the best valid combinations using simple (no material effects) chi2-fit 
    //std::cout<<"Found "<<seedSets_inCluster.size()<<" valid sets in the current cluster."<<std::endl;
    if(seedSets_inCluster.size()){
      //TrajectorySeed seed;
      //PropagationDirection dir = alongMomentum;
      //Trajectory trajectoryNew(seed, dir);
      //fwFitFailed = !(filter()->refit(seedSets_inCluster, trajectoryNew));

      //---- this is the forward fitter (segments)
      trajectoryMeasurementsFW.clear();
      fwFitFailed = !(filter()->fwfit_SET(seedSets_inCluster, trajectoryMeasurementsFW));
      //std::cout<<"after refit : fwFitFailed = "<<fwFitFailed<<std::endl;
      //trajectoryFW = trajectoryNew;

    // has the fit failed? continue to the next cluster instead of returning the empty trajectoryContainer and stop the loop IBL 080903
    if(fwFitFailed || !trajectoryMeasurementsFW.at(trajectoryMeasurementsFW.size()-1).forwardPredictedState().isValid()) continue;

    //TrajectoryStateOnSurface tsosAfterRefit = trajectoryMeasurementsFW.at(trajectoryMeasurementsFW.size()-1).forwardPredictedState();

      // are there measurements (or detLayers) used at all?
      if( filter()->layers().size() )
        LogTrace(metname) << debug.dumpLayer( filter()->lastDetLayer());
      else {
        continue;
      }

      //---- ask for some "reasonable" conditions to build a STA muon; 
      //---- (totalChambers >= 2, dtChambers + cscChambers >0)
      if (filter()->goodState()) {
	TransientTrackingRecHit::ConstRecHitContainer hitContainer;
	TrajectoryStateOnSurface firstTSOS;
	bool conversionPassed = false;
	if(!useSegmentsInTrajectory){
	// transforms set of segment measurements to a set of rechit measurements
	  conversionPassed = filter()->transform(trajectoryMeasurementsFW, hitContainer, firstTSOS);
	}
	else{
	// transforms set of segment measurements to a set of segment measurements
          conversionPassed = filter()->transformLight(trajectoryMeasurementsFW, hitContainer, firstTSOS);
	}
	if ( conversionPassed && trajectoryMeasurementsFW.size() && hitContainer.size()) {
	  setMeasurementsContainer.push_back(make_pair(firstTSOS, hitContainer));
	}
	else{
          continue;
	}
      }else{
        continue;
      }
    }
  }
  return  setMeasurementsContainer;
}

std::vector< MuonRecHitContainer > 
SETMuonSeedProducer::clusterHits( MuonRecHitContainer muonRecHits,  
					      MuonRecHitContainer muonRecHits_DT2D_hasPhi, 
					      MuonRecHitContainer muonRecHits_DT2D_hasZed, 
					      MuonRecHitContainer muonRecHits_RPC) {
  //---- From all the hits (i.e. segments; sometimes "rechits" is also used with the same meaning;
  //---- this convention has meaning in the global reconstruction though could be misleading 
  //---- from a local reconstruction point of view; "local rechits" are used in the backward fit only) 
  //---- make clusters of hits; a trajectory could contain hits from one cluster only   

  // the clustering procedure is very similar to the one used in the segment reconstruction 

  bool useDT2D_hasPhi = true;
  bool useDT2D_hasZed = true;
  double dXclusBoxMax         = 0.60; // phi - can be as large as 15 - 20 degrees for 6 GeV muons
  double dYclusBoxMax = 0.;

  // this is the main selection criteria; the value of 0.02 rad seems wide enough to 
  // contain any hit from a passing muon and still narrow enough to remove good part of
  // possible "junk" hits
  // (Comment: it may be better to allow maximum difference between any two hits in a trajectory
  // to be 0.02 or 0.04 or ...; currently the requirement below is imposed on two consecutive hits)  

  dYclusBoxMax              = 0.02; // theta // hardoded - remove it!
  
  std::vector< MuonRecHitContainer > segments_clusters; // this is a collection of groups preclustered segments

  // X and Y are distance variables - we use eta and phi here

  float dXclus = 0.0;
  float dXclus_box = 0.0;
  float dYclus_box = 0.0;

  MuonRecHitContainer temp;

  std::vector< MuonRecHitContainer > seeds;

  std::vector<float> running_meanX;
  std::vector<float> running_meanY;

  std::vector<float> seed_minX;
  std::vector<float> seed_maxX;
  std::vector<float> seed_minY;
  std::vector<float> seed_maxY;

  //std::cout<<"*************************************************************"<<std::endl;
  //std::cout<<"Called clusterHits in Chamber "<< theChamber->specs()->chamberTypeName()<<std::endl;
  //std::cout<<"*************************************************************"<<std::endl;

  // split rechits into subvectors and return vector of vectors:
  // Loop over rechits 
  // Create one seed per hit
  for (MuonRecHitContainer::const_iterator it = muonRecHits.begin(); it != muonRecHits.end(); ++it ) {

    // try to avoid using 2D DT segments. We will add them later to the 
    // clusters they are most likely to belong to. Might need to add them 
    // to more than just one cluster, if we find them to be consistent with 
    // more than one. This would lead to an implicit sharing of hits between 
    // SA muon candidates. 

    temp.clear();

    temp.push_back((*it));

    seeds.push_back(temp);

    // First added hit in seed defines the mean to which the next hit is compared
    // for this seed.

    running_meanX.push_back( (*it)->globalPosition().phi() );
    running_meanY.push_back( (*it)->globalPosition().theta() );

    // set min/max X and Y for box containing the hits in the precluster:
    seed_minX.push_back( (*it)->globalPosition().phi() );
    seed_maxX.push_back( (*it)->globalPosition().phi() );
    seed_minY.push_back( (*it)->globalPosition().theta() );
    seed_maxY.push_back( (*it)->globalPosition().theta() );
  }

  // merge clusters that are too close
  // measure distance between final "running mean"
  for(uint NNN = 0; NNN < seeds.size(); ++NNN) {

    for(uint MMM = NNN+1; MMM < seeds.size(); ++MMM) {
      if(running_meanX[MMM] == 999999. || running_meanX[NNN] == 999999. ) {
	//        LogDebug("CSC") << "CSCSegmentST::clusterHits: Warning: Skipping used seeds, this should happen - inform developers!\n";
        //      std::cout<<"We should never see this line now!!!"<<std::endl;
        continue; //skip seeds that have been used 
      }

      // Some complications for using phi as a clustering variable due to wrap-around (-pi = pi)
      // Define temporary mean, min, and max variables for the cluster which could be merged (NNN)
      double temp_meanX = running_meanX[NNN];
      double temp_minX = seed_minX[NNN];
      double temp_maxX = seed_maxX[NNN];

      // check if the difference between the two phi values is greater than pi
      // if so, need to shift temporary values by 2*pi to make a valid comparison
      dXclus = running_meanX[NNN] - running_meanX[MMM];
      if (dXclus > TMath::Pi()) {
        temp_meanX = temp_meanX - 2.*TMath::Pi();
        temp_minX = temp_minX - 2.*TMath::Pi();
        temp_maxX = temp_maxX - 2.*TMath::Pi();
      }
      if (dXclus < -TMath::Pi()) {
        temp_meanX = temp_meanX + 2.*TMath::Pi();
        temp_minX = temp_minX + 2.*TMath::Pi();
        temp_maxX = temp_maxX + 2.*TMath::Pi();
      }

      //       // calculate cut criteria for simple running mean distance cut:
      //       // not sure that these values are really used anywhere

      // calculate minmal distance between precluster boxes containing the hits:
      // use the temp variables from above for phi of the NNN cluster 
      if ( temp_meanX > running_meanX[MMM] )         dXclus_box = temp_minX - seed_maxX[MMM];
      else                                           dXclus_box = seed_minX[MMM] - temp_maxX;
      if ( running_meanY[NNN] > running_meanY[MMM] ) dYclus_box = seed_minY[NNN] - seed_maxY[MMM];
      else                                           dYclus_box = seed_minY[MMM] - seed_maxY[NNN];


      if( dXclus_box < dXclusBoxMax && dYclus_box < dYclusBoxMax ) {
        // merge clusters!
        // merge by adding seed NNN to seed MMM and erasing seed NNN

        // calculate running mean for the merged seed:
        // use the temp variables from above for phi of the NNN cluster 
        running_meanX[MMM] = (temp_meanX*seeds[NNN].size() + running_meanX[MMM]*seeds[MMM].size()) / (seeds[NNN].size()+seeds[MMM].size());
        running_meanY[MMM] = (running_meanY[NNN]*seeds[NNN].size() + running_meanY[MMM]*seeds[MMM].size()) / (seeds[NNN].size()+seeds[MMM].size());

        // update min/max X and Y for box containing the hits in the merged cluster:
        // use the temp variables from above for phi of the NNN cluster 
        if ( temp_minX <= seed_minX[MMM] ) seed_minX[MMM] = temp_minX;
        if ( temp_maxX >  seed_maxX[MMM] ) seed_maxX[MMM] = temp_maxX;
        if ( seed_minY[NNN] <= seed_minY[MMM] ) seed_minY[MMM] = seed_minY[NNN];
        if ( seed_maxY[NNN] >  seed_maxY[MMM] ) seed_maxY[MMM] = seed_maxY[NNN];

        // now check to see if the running mean has moved outside of the allowed -pi to pi region
        // if so, then adjust shift all values up or down by 2 * pi
        if (running_meanX[MMM] > TMath::Pi()) {
          running_meanX[MMM] = running_meanX[MMM] - 2.*TMath::Pi();
          seed_minX[MMM] = seed_minX[MMM] - 2.*TMath::Pi();
          seed_maxX[MMM] = seed_maxX[MMM] - 2.*TMath::Pi();
        }
        if (running_meanX[MMM] < -TMath::Pi()) {
          running_meanX[MMM] = running_meanX[MMM] + 2.*TMath::Pi();
          seed_minX[MMM] = seed_minX[MMM] + 2.*TMath::Pi();
          seed_maxX[MMM] = seed_maxX[MMM] + 2.*TMath::Pi();
        }

        // add seed NNN to MMM (lower to larger number)
        seeds[MMM].insert(seeds[MMM].end(),seeds[NNN].begin(),seeds[NNN].end());

        // mark seed NNN as used (at the moment just set running mean to 999999.)
        running_meanX[NNN] = 999999.;
        running_meanY[NNN] = 999999.;
        // we have merged a seed (NNN) to the highter seed (MMM) - need to contimue to 
        // next seed (NNN+1)
        break;
      }

    }
  }
  bool tooCloseClusters = false;
  if(seeds.size()>1){
    std::vector <double> seedTheta(seeds.size());
    for(uint iSeed = 0;iSeed<seeds.size();++iSeed){
      seedTheta[iSeed] = seeds[iSeed][0]->globalPosition().theta(); 
      if(iSeed){
	double dTheta = fabs(seedTheta[iSeed] - seedTheta[iSeed-1]);
	if (dTheta < 0.5){ //? should be something more clever
	  tooCloseClusters = true;
	  break;
	}
      }
    }
    
  }

  // have formed clusters from all hits except for 2D DT segments. Now add the 2D segments to the 
  // compatible clusters. For this we compare the mean cluster postition with the 
  // 2D segment position. We should use the valid coordinate only and use the bad coordinate 
  // as a cross check.
  for(uint NNN = 0; NNN < seeds.size(); ++NNN) {
    if(running_meanX[NNN] == 999999.) continue; //skip seeds that have been marked as used up in merging

    // We have a valid cluster - loop over all 2D segments.
    if(useDT2D_hasZed) {
      for (MuonRecHitContainer::const_iterator it2 = muonRecHits_DT2D_hasZed.begin(); it2 != muonRecHits_DT2D_hasZed.end(); ++it2 ) {
	// check that global theta of 2-D segment lies within cluster box plus or minus allowed slop
	if (((*it2)->globalPosition().theta() < seed_maxY[NNN] + dYclusBoxMax) && ((*it2)->globalPosition().theta() > seed_minY[NNN] - dYclusBoxMax)) {
	  // check that global phi of 2-D segment (assumed to be center of chamber since no phi hit info)
	  // matches with cluster box plus or minus allowed slop given that the true phi value could be 
	  // anywhere within a given chamber (+/- 5 degrees ~ 0.09 radians from center) 
	  if(
	     !( 
	       (
		((*it2)->globalPosition().phi() + 0.09) < (seed_minX[NNN] - dXclusBoxMax) 
		&& 
		((*it2)->globalPosition().phi() - 0.09) < (seed_minX[NNN] - dXclusBoxMax)
		)
	       ||
	       (
		((*it2)->globalPosition().phi() + 0.09) > (seed_maxX[NNN] + dXclusBoxMax) 
		&& 
		((*it2)->globalPosition().phi() - 0.09) > (seed_maxX[NNN] + dXclusBoxMax)
		)
	       )
	     ) { // we have checked that the 2Dsegment is within tight theta boundaries and loose phi boundaries of the current cluster -> add it
	    seeds[NNN].push_back((*it2));
	    
	  }
	}                 
      }
      
    }
    
    // put DT hasphi loop here
    if (useDT2D_hasPhi) {
      
      for (MuonRecHitContainer::const_iterator it2 = muonRecHits_DT2D_hasPhi.begin(); it2 != muonRecHits_DT2D_hasPhi.end(); ++it2 ) {
	if (((*it2)->globalPosition().phi() < seed_maxX[NNN] + dXclusBoxMax) && ((*it2)->globalPosition().phi() > seed_minX[NNN] - dXclusBoxMax)) {
	  if(
	     !( 
	       (
		((*it2)->globalPosition().theta() + 0.3) < (seed_minY[NNN] - dYclusBoxMax) 
		&& 
		((*it2)->globalPosition().theta() - 0.3) < (seed_minY[NNN] - dYclusBoxMax)
		)
	       ||
	       (
		((*it2)->globalPosition().theta() + 0.3) > (seed_maxY[NNN] + dYclusBoxMax) 
		&& 
		((*it2)->globalPosition().theta() - 0.3) > (seed_maxY[NNN] + dYclusBoxMax)
		)
	       )
	     ) { // we have checked that the 2Dsegment is within tight phi boundaries and loose theta boundaries of the current cluster -> add it
	    seeds[NNN].push_back((*it2)); // warning - neeed eta/theta switch here
	    
	  }
	}
      }
    } // DT2D_hastPhi loop
    
    // put RPC loop here
    int secondCh = 0;
    DetId detId_prev;
    if(seeds[NNN].size()>1){// actually we should check how many chambers with measurements are present
      for(uint iRH = 0 ;iRH<seeds[NNN].size() ;++iRH){
        if( iRH && detId_prev != seeds[NNN][iRH]->hit()->geographicalId()){
	  ++secondCh;
	  break;
	}
	detId_prev = seeds[NNN][iRH]->hit()->geographicalId();
      }
    }

    if (useRPCs && !secondCh && !tooCloseClusters) {
      for (MuonRecHitContainer::const_iterator it2 = muonRecHits_RPC.begin(); it2 != muonRecHits_RPC.end(); ++it2 ) {
	if (((*it2)->globalPosition().phi() < seed_maxX[NNN] + dXclusBoxMax) && ((*it2)->globalPosition().phi() > seed_minX[NNN] - dXclusBoxMax)) {
	  if(
	     !( 
	       (
		((*it2)->globalPosition().theta() + 0.3) < (seed_minY[NNN] - dYclusBoxMax) 
		&& 
		((*it2)->globalPosition().theta() - 0.3) < (seed_minY[NNN] - dYclusBoxMax)
		)
	       ||
	       (
		((*it2)->globalPosition().theta() + 0.3) > (seed_maxY[NNN] + dYclusBoxMax) 
		&& 
		((*it2)->globalPosition().theta() - 0.3) > (seed_maxY[NNN] + dYclusBoxMax)
		)
	       )
	     ) { // we have checked that the 2Dsegment is within tight phi boundaries and loose theta boundaries of the current cluster -> add it
	    seeds[NNN].push_back((*it2)); // warning - neeed eta/theta switch here
	    
	  }
	}
      }
    } // RPC loop
  }
  
  // hand over the final seeds to the output
  // would be more elegant if we could do the above step with 
  // erasing the merged ones, rather than the 
  for(uint NNN = 0; NNN < seeds.size(); ++NNN) {
    if(running_meanX[NNN] == 999999.) continue; //skip seeds that have been marked as used up in merging
    //std::cout<<"Next Cluster..."<<std::endl;
    segments_clusters.push_back(seeds[NNN]);
  }

  //***************************************************************

      return segments_clusters;
}

std::vector <MuonRecHitContainer> SETMuonSeedProducer::
findAllValidSets(std::vector< MuonRecHitContainer > MuonRecHitContainer_perLayer){
  std::vector <MuonRecHitContainer> allValidSets;
  // build all possible combinations (i.e valid sets; the algorithm name is after this feature - 
  // SET algorithm)
  // 
  // ugly... use recursive function?! 
  // or implement Ingo's suggestion (a la ST)
  if(1==MuonRecHitContainer_perLayer.size()){
    return allValidSets;
  }
  MuonRecHitContainer validSet;
  unsigned int iPos0 = 0;
  std::vector <unsigned int> iLayer(12);// could there be more than 11 layers?
  std::vector <unsigned int> size(12);
  if(iPos0<MuonRecHitContainer_perLayer.size()){
    size.at(iPos0) =  MuonRecHitContainer_perLayer.at(iPos0).size();
    for(iLayer[iPos0] = 0; iLayer[iPos0]<size[iPos0];++iLayer[iPos0]){
      validSet.clear();
      validSet.push_back(MuonRecHitContainer_perLayer[iPos0][iLayer[iPos0]]);
      unsigned int iPos1 = 1;
      if(iPos1<MuonRecHitContainer_perLayer.size()){
        size.at(iPos1) =  MuonRecHitContainer_perLayer.at(iPos1).size();
        for(iLayer[iPos1] = 0; iLayer[iPos1]<size[iPos1];++iLayer[iPos1]){
          validSet.resize(iPos1);
          validSet.push_back(MuonRecHitContainer_perLayer[iPos1][iLayer[iPos1]]);
          unsigned int iPos2 = 2;
          if(iPos2<MuonRecHitContainer_perLayer.size()){
            size.at(iPos2) =  MuonRecHitContainer_perLayer.at(iPos2).size();
            for(iLayer[iPos2] = 0; iLayer[iPos2]<size[iPos2];++iLayer[iPos2]){
              validSet.resize(iPos2);
              validSet.push_back(MuonRecHitContainer_perLayer[iPos2][iLayer[iPos2]]);
              unsigned int iPos3 = 3;
              if(iPos3<MuonRecHitContainer_perLayer.size()){
                size.at(iPos3) =  MuonRecHitContainer_perLayer.at(iPos3).size();
                for(iLayer[iPos3] = 0; iLayer[iPos3]<size[iPos3];++iLayer[iPos3]){
                  validSet.resize(iPos3);
                  validSet.push_back(MuonRecHitContainer_perLayer[iPos3][iLayer[iPos3]]);
                  unsigned int iPos4 = 4;
                  if(iPos4<MuonRecHitContainer_perLayer.size()){
                    size.at(iPos4) =  MuonRecHitContainer_perLayer.at(iPos4).size();
                    for(iLayer[iPos4] = 0; iLayer[iPos4]<size[iPos4];++iLayer[iPos4]){
                      validSet.resize(iPos4);
                      validSet.push_back(MuonRecHitContainer_perLayer[iPos4][iLayer[iPos4]]);
                      unsigned int iPos5 = 5;
                      if(iPos5<MuonRecHitContainer_perLayer.size()){
                        size.at(iPos5) =  MuonRecHitContainer_perLayer.at(iPos5).size();
                        for(iLayer[iPos5] = 0; iLayer[iPos5]<size[iPos5];++iLayer[iPos5]){
                          validSet.resize(iPos5);
                          validSet.push_back(MuonRecHitContainer_perLayer[iPos5][iLayer[iPos5]]);
                          unsigned int iPos6 = 6;
                          if(iPos6<MuonRecHitContainer_perLayer.size()){
                            size.at(iPos6) =  MuonRecHitContainer_perLayer.at(iPos6).size();
                            for(iLayer[iPos6] = 0; iLayer[iPos6]<size[iPos6];++iLayer[iPos6]){
                              validSet.resize(iPos6);
                              validSet.push_back(MuonRecHitContainer_perLayer[iPos6][iLayer[iPos6]]);
                              unsigned int iPos7 = 7;
                              if(iPos7<MuonRecHitContainer_perLayer.size()){
                                size.at(iPos7) =  MuonRecHitContainer_perLayer.at(iPos7).size();
                                for(iLayer[iPos7] = 0; iLayer[iPos7]<size[iPos7];++iLayer[iPos7]){
                                  validSet.resize(iPos7);
                                  validSet.push_back(MuonRecHitContainer_perLayer[iPos7][iLayer[iPos7]]);
                                  unsigned int iPos8 = 8;
                                  if(iPos8<MuonRecHitContainer_perLayer.size()){
                                    size.at(iPos8) =  MuonRecHitContainer_perLayer.at(iPos8).size();
                                    for(iLayer[iPos8] = 0; iLayer[iPos8]<size[iPos8];++iLayer[iPos8]){
                                      validSet.resize(iPos8);
                                      validSet.push_back(MuonRecHitContainer_perLayer[iPos8][iLayer[iPos8]]);
                                      unsigned int iPos9 = 9;
                                      if(iPos9<MuonRecHitContainer_perLayer.size()){
                                        size.at(iPos9) =  MuonRecHitContainer_perLayer.at(iPos9).size();
                                        for(iLayer[iPos9] = 0; iLayer[iPos9]<size[iPos9];++iLayer[iPos9]){
                                          validSet.resize(iPos9);
                                          validSet.push_back(MuonRecHitContainer_perLayer[iPos9][iLayer[iPos9]]);
                                          unsigned int iPos10 = 10;
                                          if(iPos10<MuonRecHitContainer_perLayer.size()){
                                            size.at(iPos10) =  MuonRecHitContainer_perLayer.at(iPos10).size();
                                            for(iLayer[iPos10] = 0; iLayer[iPos10]<size[iPos10];++iLayer[iPos10]){
                                              validSet.resize(iPos10);
                                              validSet.push_back(MuonRecHitContainer_perLayer[iPos10][iLayer[iPos10]]);
                                              unsigned int iPos11 = 11;// more?
                                              if(iPos11<MuonRecHitContainer_perLayer.size()){
                                                size.at(iPos11) =  MuonRecHitContainer_perLayer.at(iPos11).size();
                                                for(iLayer[iPos11] = 0; iLayer[iPos11]<size[iPos11];++iLayer[iPos11]){
                                                }
                                              }
                                              else{
                                                allValidSets.push_back(validSet);
						
                                              }
                                            }
                                          }
                                          else{
                                            allValidSets.push_back(validSet);
                                          }
                                        }
                                      }
                                      else{
                                        allValidSets.push_back(validSet);
                                      }
                                    }
                                  }
                                  else{
                                    allValidSets.push_back(validSet);
                                  }
                                }
                              }
                              else{
                                allValidSets.push_back(validSet);
                              }
                            }
                          }
                          else{
                            allValidSets.push_back(validSet);
                          }
                        }
                      }
                      else{
                        allValidSets.push_back(validSet);
                      }
                    }
                  }
                  else{
                    allValidSets.push_back(validSet);
                  }
                }
              }
              else{
                allValidSets.push_back(validSet);
              }
            }
          }
          else{
            allValidSets.push_back(validSet);
          }
        }
      }
      else{
        allValidSets.push_back(validSet);
      }
    }
  }
  else{
    allValidSets.push_back(validSet);
  }
  return allValidSets;
}

void SETMuonSeedProducer::
validSetsPrePruning(std::vector <MuonRecHitContainer>  & allValidSets){
  //---- this actually is a pre-pruning; it does not include any fit information; 
  //---- it is intended to remove only very "wild" segments from a set;
  //---- no "good" segment is to be lost (otherwise - widen the parameters)

  // any information could be used to make a decision for pruning
  // maybe dPhi (delta Phi) is enough
  std::vector <double> dPhi;
  //std::vector <double> dTheta;
  //std::vector <double> dR;  
  std::vector <int> pruneHit;
  double dPhi_tmp;
  //double dTheta_tmp;
  //double dR_tmp;
  bool wildCandidate;
  int pruneHit_tmp;

  //---- loop over all the valid sets
  for(unsigned int iSet = 0;iSet<allValidSets.size();++iSet){
    if(allValidSets[iSet].size()>3){ // to decide we need at least 4 measurements
      dPhi.clear();
      for(unsigned int iHit = 1;iHit<allValidSets[iSet].size();++iHit){
        dPhi_tmp = allValidSets[iSet][iHit]->globalPosition().phi() -
	           allValidSets[iSet][iHit-1]->globalPosition().phi();
        dPhi.push_back(dPhi_tmp);		   
      }
      pruneHit.clear();
      //---- loop over all the hits in a set
      
      for(unsigned int iHit = 0;iHit<allValidSets[iSet].size();++iHit){
	double dPHI_MIN = 0.02;//?? hardcoded - remove it
	if(iHit){
	  // if we have to remove the very first hit (iHit == 0) then 
	  // we'll probably be in trouble already  
	  wildCandidate = false;
	  // actually 2D is bad only if not r-phi... Should I refine it? 
	  // a hit is a candidate for pruning only if dPhi > dPHI_MIN;
	  // pruning decision is based on combination of hits characteristics
	  if(4==allValidSets[iSet][iHit-1]->dimension() && 4 == allValidSets[iSet][iHit]->dimension() &&
	     fabs(allValidSets[iSet][iHit]->globalPosition().phi() -
		  allValidSets[iSet][iHit-1]->globalPosition().phi())>dPHI_MIN ){
	    wildCandidate = true;
	  }
	  pruneHit_tmp = -1;
	  if(wildCandidate){
	    // OK - this couple doesn't look good (and is from 4D segments); proceed...
	    if(1==iHit){// the first  and the last hits are special case
	      if(4==allValidSets[iSet][iHit+1]->dimension() && 4 == allValidSets[iSet][iHit+2]->dimension()){//4D?
		// is the picture better if we remove the second hit?
		dPhi_tmp = allValidSets[iSet][iHit+1]->globalPosition().phi() - 
		  allValidSets[iSet][iHit-1]->globalPosition().phi();
		// is the deviation what we expect (sign, not magnitude)?
		std::pair <int, int> sign = checkAngleDeviation(dPhi_tmp, dPhi[2]);
		if( 1==sign.first && 1==sign.second){
		  pruneHit_tmp = iHit;// mark the hit 1 for removing
		}
	      }
	    }
	    else if(iHit>1 && iHit<allValidSets[iSet].size()-1){ 
	      if(4 == allValidSets[iSet][0]->dimension() && // we rely on the first (most important) couple
		 4 == allValidSets[iSet][1]->dimension() && 
		 pruneHit.back()!=int(iHit-1) && pruneHit.back()!=1){// check if hits are already marked 
		// decide which of the two hits should be removed (if any; preferably the outer one i.e.
		// iHit rather than iHit-1); here - check what we get by removing iHit
		dPhi_tmp = allValidSets[iSet][iHit+1]->globalPosition().phi() - 
		  allValidSets[iSet][iHit-1]->globalPosition().phi();
		// first couple is most important anyway so again compare to it
		std::pair <int, int> sign = checkAngleDeviation(dPhi[0],dPhi_tmp);
		if( 1==sign.first && 1==sign.second){
		  pruneHit_tmp = iHit; // mark the hit iHit for removing
		}
		else{ // iHit is not to be removed; proceed...
		  // what if we remove (iHit - 1) instead of iHit?
		  dPhi_tmp = allValidSets[iSet][iHit+1]->globalPosition().phi() - 
		    allValidSets[iSet][iHit]->globalPosition().phi();
		  std::pair <int, int> sign = checkAngleDeviation(dPhi[0],dPhi_tmp);
		  if( 1==sign.first && 1==sign.second){
		    pruneHit_tmp = iHit-1;// mark the hit (iHit -1) for removing
		  }
		}
	      }
	    }
	    else{
	      // the last hit: if picture is not good - remove it 
	      if(pruneHit.size()>1 && pruneHit[pruneHit.size()-1]<0 && pruneHit[pruneHit.size()-2]<0){
		std::pair <int, int> sign = checkAngleDeviation(dPhi[dPhi.size()-2], dPhi[dPhi.size()-1]);
		if( -1==sign.first && -1==sign.second){// here logic is a bit twisted
		  pruneHit_tmp = iHit; // mark the last hit for removing
		}
	      }
	    }
	  }
	  pruneHit.push_back(pruneHit_tmp);
	}
      }
      // }
      // actual pruning
      unsigned int size = allValidSets[iSet].size();
      for(unsigned int iHit = 1;iHit<size;++iHit){
	int count = 0;
	if(pruneHit[iHit-1]>0){
	  allValidSets[iSet].erase(allValidSets[iSet].begin()+pruneHit[iHit-1]-count);
	  ++count; 
	}
      }
    }
  }
}

std::pair <int, int> SETMuonSeedProducer::// or <bool, bool>
checkAngleDeviation(double dPhi_1, double dPhi_2){
  // Two conditions:
  // a) deviations should be only to one side (above some absolute value cut to avoid
  //    material effects; this should be refined)
  // b) deviatiation in preceding steps should be bigger due to higher magnetic field
  //    (again - a minimal value cut should be in place; this also should account for 
  //     the small (Z) distances in overlaping CSC chambers)

  double mult = dPhi_1 * dPhi_2;
  int signVal = 1;
  if(fabs(dPhi_1)<fabs(dPhi_2)){
    signVal = -1;
  }
  int signMult = -1;
  if(mult>0) signMult = 1;
  std::pair <int, int> sign;
  sign = make_pair (signVal, signMult);

  return sign;
}

std::vector <seedSet> SETMuonSeedProducer::
fillSeedSets(std::vector <MuonRecHitContainer> & allValidSets){
  //---- we have the valid sets constructed; transform the information in an
  //---- apropriate form; meanwhile - estimate the momentum for a given set

  // RPCs should not be used (no parametrization)
  std::vector <seedSet> seedSets_inCluster;
  // calculate and fill the inputs needed
  // loop over all valid sets
  for(unsigned int iSet = 0;iSet<allValidSets.size();++iSet){
    int firstMeasurement = -1;
    int lastMeasurement = -1;    
    // don't use 2D measurements for momentum estimation (except there are no others)
    //if( 4==allValidSets[iSet].front()->dimension() &&
    //(allValidSets[iSet].front()->isCSC() || allValidSets[iSet].front()->isDT())){
    //firstMeasurement = 0;
    //}
    //else{
    // which is the "first" hit (4D)?
    for(unsigned int iMeas = 0;iMeas<allValidSets[iSet].size();++iMeas){
      if(4==allValidSets[iSet][iMeas]->dimension() &&
	 (allValidSets[iSet][iMeas]->isCSC() || allValidSets[iSet][iMeas]->isDT())){
	firstMeasurement = iMeas;
	break;
      }
    } 
      //}

    std::vector<double> momentum_estimate;
    double pT = 0.;
    MuonTransientTrackingRecHit::ConstMuonRecHitPointer  firstHit;
    MuonTransientTrackingRecHit::ConstMuonRecHitPointer  secondHit;
    // which is the second hit?
    for(int loop = 0; loop<2; ++loop){// it is actually not used; to be removed    
      // this is the last measurement
      if(!loop){// this is what is used currently
	for(int iMeas = allValidSets[iSet].size()-1;iMeas>-1;--iMeas){
	  if(4==allValidSets[iSet][iMeas]->dimension() && 
	     (allValidSets[iSet][iMeas]->isCSC() || allValidSets[iSet][iMeas]->isDT())){
	    lastMeasurement = iMeas;
	    break;
	  }
	} 
      }
      else{
	// this is the second measurement
	for(unsigned int iMeas = 1;iMeas<allValidSets[iSet].size();++iMeas){
	  if(4==allValidSets[iSet][iMeas]->dimension() &&
	     (allValidSets[iSet][iMeas]->isCSC() || allValidSets[iSet][iMeas]->isDT())){
	    lastMeasurement = iMeas;
	    break;
	  }
	} 
      }

      // only 2D measurements (it should have been already abandoned)
      if(-1==lastMeasurement && -1==firstMeasurement){
        firstMeasurement = 0; 
        lastMeasurement = allValidSets[iSet].size()-1;
      }
      
      firstHit = allValidSets[iSet][firstMeasurement];
      secondHit = allValidSets[iSet][lastMeasurement];
      if(firstHit->isRPC() && secondHit->isRPC()){ // remove all RPCs from here?
	momentum_estimate.push_back(300.);
	momentum_estimate.push_back(300.);
      }
      else{
	if(firstHit->isRPC()){
	  firstHit = secondHit;
	}
	else if(secondHit->isRPC()){
	  secondHit  = firstHit;
	}
	//---- estimate pT given two hits
	momentum_estimate = pt_extractor()->pT_extract(firstHit, secondHit);
      }
      pT = fabs(momentum_estimate[0]);
      if(1 || pT>40.){ //it is skipped; we have to look at least into number of hits in the chamber actually...
	// and then decide which segment to use
	// use the last measurement, otherwise use the second; this is to be investigated
	break;
      }
    }

    const float pT_min = 1.99;// many hardcoded - remove them!     
    if(pT>3000.){
      pT=3000.;
    }
    else if(pT<pT_min ){
      if(pT>0){
	pT=pT_min ;
      }
      else if(pT>(-1)*pT_min ){
	pT=(-1)*pT_min ;
      }
      else if (pT<-3000.){
	pT= -3000;
      }
    }
    //std::cout<<" THE pT from the parametrization: "<<momentum_estimate[0]<<std::endl;
    // estimate the charge of the track candidate from the delta phi of two segments:
    //int charge      = dPhi > 0 ? 1 : -1; // what we want is: dphi < 0 => charge = -1
    int charge =  momentum_estimate[0]> 0 ? 1 : -1;
    
    // we have the pT - get the 3D momentum estimate as well

    // this is already final info:
    double xHit     = allValidSets[iSet][firstMeasurement]->globalPosition().x();
    double yHit     = allValidSets[iSet][firstMeasurement]->globalPosition().y();
    double rHit     = TMath::Sqrt(pow(xHit,2) + pow(yHit,2));

    double thetaInner = allValidSets[iSet][firstMeasurement]->globalPosition().theta();
    // if some of the segments is missing r-phi measurement then we should
    // use only the 4D phi estimate (also use 4D eta estimate only)
    // the direction is not so important (it will be corrected) 
   
    double rTrack   = (pT /(0.3*3.8))*100.; //times 100 for conversion to cm!

    double par      = -1.*(2./charge)*(TMath::ASin(rHit/(2*rTrack)));
    double sinPar     = TMath::Sin( par );
    double cosPar     = TMath::Cos( par );

    // calculate phi at coordinate origin (0,0,0).
    double sinPhiH  = 1./(2.*charge*rTrack)*(xHit + ((sinPar)/(cosPar-1.))*yHit);
    double cosPhiH  = -1./(2.*charge*rTrack)*(((sinPar)/(1.-cosPar))*xHit + yHit);

    // finally set the return vector

    // try out the reco info:
    Hep3Vector momEstimate(pT*cosPhiH, pT*sinPhiH, pT/TMath::Tan(thetaInner)); // should used into to theta directly here (rather than tan(atan2(...)))
    //Hep3Vector momEstimate(6.97961,      5.89732,     -50.0855);
    if (momEstimate.mag()<10.){
      int sign = (pT<0.) ? -1: 1;
      pT = sign * (fabs(pT)+1);
      Hep3Vector momEstimate2(pT*cosPhiH, pT*sinPhiH, pT/TMath::Tan(thetaInner));
      momEstimate = momEstimate2;
      if (momEstimate.mag()<10.){
	pT = sign * (fabs(pT)+1);
	Hep3Vector momEstimate3(pT*cosPhiH, pT*sinPhiH, pT/TMath::Tan(thetaInner));
	momEstimate = momEstimate3;
	if (momEstimate.mag()<10.){
	  pT = sign * (fabs(pT)+1);
	  Hep3Vector momEstimate4(pT*cosPhiH, pT*sinPhiH, pT/TMath::Tan(thetaInner));
	  momEstimate = momEstimate4;
	}
      }
    }
    int chargeEstimate = charge;
    //std::cout<<"Final estimated p  "<<momEstimate.mag() <<std::endl;
    //std::cout<<"Final estimated pT "<<momEstimate.perp()<<std::endl;
    //std::cout<<"Final estimated q  "<<chargeEstimate<<std::endl;

    // push back info to vector of structs:

    MuonRecHitContainer MuonRecHitContainer_theSet_prep;
    // currently hardcoded - will be in proper loop of course:

    seedSet seedSets_inCluster_prep;
    seedSets_inCluster_prep.theSet   = allValidSets[iSet];
    seedSets_inCluster_prep.momentum = momEstimate;
    seedSets_inCluster_prep.charge   = chargeEstimate;
    seedSets_inCluster.push_back(seedSets_inCluster_prep);
    // END estimateMomentum 
  }
  return seedSets_inCluster;
}

void SETMuonSeedProducer::setEvent(const edm::Event& event){
  theFilter->setEvent(event);
}
