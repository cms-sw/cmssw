/** \class SETPatternRecognition
    I. Bloch, E. James, S. Stoynev
 */

#include "RecoMuon/MuonSeedGenerator/src/SETPatternRecognition.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"            
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"             
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"                
#include "TMath.h"


using namespace edm;
using namespace std;

SETPatternRecognition::SETPatternRecognition(const ParameterSet& parameterSet)
: MuonSeedVPatternRecognition(parameterSet.getParameter<ParameterSet>("SETTrajBuilderParameters").getParameter<ParameterSet>("FilterParameters"))
{
  const string metname = "Muon|RecoMuon|SETPatternRecognition";  
  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("SETTrajBuilderParameters");
  // The inward-outward fitter (starts from seed state)
  ParameterSet filterPSet = trajectoryBuilderParameters.getParameter<ParameterSet>("FilterParameters");
  maxActiveChambers = filterPSet.getParameter<int>("maxActiveChambers"); 
  useRPCs = filterPSet.getParameter<bool>("EnableRPCMeasurement");
  DTRecSegmentLabel  = filterPSet.getParameter<edm::InputTag>("DTRecSegmentLabel");
  CSCRecSegmentLabel  = filterPSet.getParameter<edm::InputTag>("CSCRecSegmentLabel");
  RPCRecSegmentLabel  = filterPSet.getParameter<edm::InputTag>("RPCRecSegmentLabel");

  outsideChamberErrorScale = filterPSet.getParameter<double>("OutsideChamberErrorScale");
  minLocalSegmentAngle = filterPSet.getParameter<double>("MinLocalSegmentAngle");
  //----

} 

//---- it is a "cluster recognition" actually; the pattern recognition is in SETFilter 
void SETPatternRecognition::produce(const edm::Event& event, const edm::EventSetup& eventSetup,
                         std::vector<MuonRecHitContainer> & segments_clusters)
{
  const string metname = "Muon|RecoMuon|SETMuonSeedSeed";  

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
  std::vector<DTChamberId> chambers_DT;
  std::vector<DTChamberId>::const_iterator chIt_DT;
  for (DTRecSegment4DCollection::const_iterator rechit = dtRecHits->begin(); rechit!=dtRecHits->end();++rechit) {
    bool insert = true;
    for(chIt_DT=chambers_DT.begin(); chIt_DT != chambers_DT.end(); ++chIt_DT){
      if (
	  ((*rechit).chamberId().wheel()) == ((*chIt_DT).wheel()) &&
	  ((*rechit).chamberId().station() == (*chIt_DT).station()) &&
	  ((*rechit).chamberId().sector() == (*chIt_DT).sector())){
	insert = false;
      }
    }
    if (insert){
      chambers_DT.push_back((*rechit).chamberId());
    }
    if(segmentCleaning((*rechit).geographicalId(), 
		       rechit->localPosition(), rechit->localPositionError(),
		       rechit->localDirection(), rechit->localDirectionError(),
		       rechit->chi2(), rechit->degreesOfFreedom())){
      continue;
    }
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
      //std::cout<<"Warning in "<<metname<<": DT segment which claims to have neither phi nor Z."<<std::endl;
    }
  }
  //std::cout<<"DT done"<<std::endl;

  // ********************************************;
  // Get the CSC-Segment collection from the event
  // ********************************************;

  edm::Handle<CSCSegmentCollection> cscSegments;
  event.getByLabel(CSCRecSegmentLabel, cscSegments);
  std::vector<CSCDetId> chambers_CSC;
  std::vector<CSCDetId>::const_iterator chIt_CSC;
  for(CSCSegmentCollection::const_iterator rechit=cscSegments->begin(); rechit != cscSegments->end(); ++rechit) {
    bool insert = true;
    for(chIt_CSC=chambers_CSC.begin(); chIt_CSC != chambers_CSC.end(); ++chIt_CSC){
      if (((*rechit).cscDetId().chamber() == (*chIt_CSC).chamber()) &&
	  ((*rechit).cscDetId().station() == (*chIt_CSC).station()) &&
	  ((*rechit).cscDetId().ring() == (*chIt_CSC).ring()) &&
	  ((*rechit).cscDetId().endcap() == (*chIt_CSC).endcap())){
	insert = false;
      }
    }
    if (insert){
      chambers_CSC.push_back((*rechit).cscDetId().chamberId());
    }
    if(segmentCleaning((*rechit).geographicalId(), 
		       rechit->localPosition(), rechit->localPositionError(),
		       rechit->localDirection(), rechit->localDirectionError(),
		       rechit->chi2(), rechit->degreesOfFreedom())){
      continue;
    }
    muonRecHits.push_back(MuonTransientTrackingRecHit::specificBuild(theService->trackingGeometry()->idToDet((*rechit).geographicalId()),&*rechit));
  }
  //std::cout<<"CSC done"<<std::endl;

  // ********************************************;
  // Get the RPC-Hit collection from the event
  // ********************************************;

  edm::Handle<RPCRecHitCollection> rpcRecHits;
  event.getByLabel(RPCRecSegmentLabel, rpcRecHits);
  if(useRPCs){
    for(RPCRecHitCollection::const_iterator rechit=rpcRecHits->begin(); rechit != rpcRecHits->end(); ++rechit) {
      // RPCs are special
      const LocalVector  localDirection(0.,0.,1.);
      const LocalError localDirectionError (0.,0.,0.); 
      const double chi2 = 1.;
      const int ndf = 1;
      if(segmentCleaning((*rechit).geographicalId(), 
			 rechit->localPosition(), rechit->localPositionError(),
			 localDirection, localDirectionError,
			 chi2, ndf)){
	continue;
      }
      muonRecHits_RPC.push_back(MuonTransientTrackingRecHit::specificBuild(theService->trackingGeometry()->idToDet((*rechit).geographicalId()),&*rechit));
    }
  }
  //std::cout<<"RPC done"<<std::endl;
  //
  if(int(chambers_DT.size() + chambers_CSC.size()) > maxActiveChambers){
    // std::cout <<" Too many active chambers : nDT = "<<chambers_DT.size()<<
    // " nCSC = "<<chambers_CSC.size()<<"  Skip them all."<<std::endl;
    edm::LogWarning("tooManyActiveChambers")<<" Too many active chambers : nDT = "<<chambers_DT.size()
		     <<" nCSC = "<<chambers_CSC.size()<<"  Skip them all.";
    muonRecHits.clear();                                
    muonRecHits_DT2D_hasPhi.clear();    
    muonRecHits_DT2D_hasZed.clear();
    muonRecHits_RPC.clear();
  }
  //---- Find "pre-clusters" from all segments; these contain potential muon candidates

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
  for(unsigned int NNN = 0; NNN < seeds.size(); ++NNN) {

    for(unsigned int MMM = NNN+1; MMM < seeds.size(); ++MMM) {
      if(running_meanX[MMM] == 999999. || running_meanX[NNN] == 999999. ) {
	//        LogDebug("CSC") << "CSCSegmentST::clusterHits: Warning: Skipping used seeds, this should happen - inform developers!\n";
	//std::cout<<"We should never see this line now!!!"<<std::endl;
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
    for(unsigned int iSeed = 0;iSeed<seeds.size();++iSeed){
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
  for(unsigned int NNN = 0; NNN < seeds.size(); ++NNN) {
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
      for(unsigned int iRH = 0 ;iRH<seeds[NNN].size() ;++iRH){
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
  for(unsigned int NNN = 0; NNN < seeds.size(); ++NNN) {
    if(running_meanX[NNN] == 999999.) continue; //skip seeds that have been marked as used up in merging
    //std::cout<<"Next Cluster..."<<std::endl;
    segments_clusters.push_back(seeds[NNN]);
  }
}


bool SETPatternRecognition::segmentCleaning(const DetId & detId, 
					    const LocalPoint& localPosition, const LocalError& localError,
					    const LocalVector& localDirection, const LocalError& localDirectionError,
					    const double& chi2, const int& ndf){
  // drop segments which are "bad"
  bool dropTheSegment = true;
  const GeomDet* geomDet = theService->trackingGeometry()->idToDet( detId );
  // only segments whithin the boundaries of the chamber
  bool insideCh = geomDet->surface().bounds().inside(localPosition, localError,outsideChamberErrorScale);

  // Don't use segments (nearly) parallel to the chamberi;
  // the direction vector is normalized (R=1)  
  bool parallelSegment = fabs(localDirection.z())>minLocalSegmentAngle? false: true;

  if(insideCh && !parallelSegment){
    dropTheSegment = false;
  }
  // use chi2 too? (DT, CSCs, RPCs; 2D, 4D;...)


  return dropTheSegment;
}
