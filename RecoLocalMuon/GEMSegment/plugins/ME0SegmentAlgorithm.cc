/**
 * \file ME0SegmentAlgorithm.cc
 *  based on CSCSegAlgoST.cc
 * 
 *  \authors: Marcello Maggi, Jason Lee
 */
 
#include "ME0SegmentAlgorithm.h"
#include "MuonSegFit.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

/* Constructor
 *
 */
ME0SegmentAlgorithm::ME0SegmentAlgorithm(const edm::ParameterSet& ps) : ME0SegmentAlgorithmBase(ps), myName("ME0SegmentAlgorithm")
{
  debug                     = ps.getUntrackedParameter<bool>("ME0Debug");
  minHitsPerSegment         = ps.getParameter<unsigned int>("minHitsPerSegment");
  preClustering             = ps.getParameter<bool>("preClustering");
  dXclusBoxMax              = ps.getParameter<double>("dXclusBoxMax");
  dYclusBoxMax              = ps.getParameter<double>("dYclusBoxMax");
  preClustering_useChaining = ps.getParameter<bool>("preClusteringUseChaining");
  dPhiChainBoxMax           = ps.getParameter<double>("dPhiChainBoxMax");
  dEtaChainBoxMax           = ps.getParameter<double>("dEtaChainBoxMax");
  dTimeChainBoxMax          = ps.getParameter<double>("dTimeChainBoxMax");
  maxRecHitsInCluster       = ps.getParameter<int>("maxRecHitsInCluster");

  edm::LogVerbatim("ME0SegmentAlgorithm") << "[ME0SegmentAlgorithm::ctor] Parameters to build segments :: "
				   << "preClustering = "<<preClustering<<" preClustering_useChaining = "<<preClustering_useChaining
				   <<" dPhiChainBoxMax = "<<dPhiChainBoxMax<<" dEtaChainBoxMax = "<<dEtaChainBoxMax<<" dTimeChainBoxMax = "<<dTimeChainBoxMax
				   <<" minHitsPerSegment = "<<minHitsPerSegment<<" maxRecHitsInCluster = "<<maxRecHitsInCluster;
}

/* Destructor
 *
 */
ME0SegmentAlgorithm::~ME0SegmentAlgorithm() {
}


std::vector<ME0Segment> ME0SegmentAlgorithm::run(const ME0Ensemble& ensemble, const EnsembleHitContainer& rechits) {

  #ifdef EDM_ML_DEBUG // have lines below only compiled when in debug mode
  ME0DetId chId((ensemble.first)->id());  
  edm::LogVerbatim("ME0SegAlgoMM") << "[ME0SegmentAlgorithm::run] build segments in chamber " << chId << " which contains "<<rechits.size()<<" rechits";
  for (auto rh=rechits.begin(); rh!=rechits.end(); ++rh){
    auto me0id = (*rh)->me0Id();
    auto rhLP = (*rh)->localPosition();
    edm::LogVerbatim("ME0SegmentAlgorithm") << "[RecHit :: Loc x = "<<std::showpos<<std::setw(9)<<rhLP.x()<<" Loc y = "<<std::showpos<<std::setw(9)<<rhLP.y()<<" Time = "<<std::showpos<<(*rh)->tof()<<" -- "<<me0id.rawId()<<" = "<<me0id<<" ]";
  }
  #endif

  // pre-cluster rechits and loop over all sub clusters separately
  std::vector<ME0Segment>          segments_temp;
  std::vector<ME0Segment>          segments;
  ProtoSegments rechits_clusters; // this is a collection of groups of rechits
  
  if(preClustering) {
    // run a pre-clusterer on the given rechits to split obviously separated segment seeds:
    if(preClustering_useChaining){
      // it uses X,Y,Z information; there are no configurable parameters used;
      // the X, Y, Z "cuts" are just (much) wider than reasonable high pt segments
      rechits_clusters = this->chainHits(ensemble, rechits );
    }
    else{
      // it uses X,Y information + configurable parameters
      rechits_clusters = this->clusterHits(rechits );
    }
    // loop over the found clusters:
    for(auto sub_rechits = rechits_clusters.begin(); sub_rechits !=  rechits_clusters.end(); ++sub_rechits ) {
      // clear the buffer for the subset of segments:
      segments_temp.clear();
      // build the subset of segments:
      this->buildSegments(ensemble, (*sub_rechits), segments_temp );
      // add the found subset of segments to the collection of all segments in this chamber:
      segments.insert( segments.end(), segments_temp.begin(), segments_temp.end() );
    }
    return segments;
  }
  else {
    this->buildSegments(ensemble, rechits, segments);
    return segments;
  }
}


// ********************************************************************;
ME0SegmentAlgorithm::ProtoSegments 
ME0SegmentAlgorithm::clusterHits(const EnsembleHitContainer& rechits) {

  ProtoSegments rechits_clusters; // this is a collection of groups of rechits

  float dXclus_box = 0.0;
  float dYclus_box = 0.0;

  ProtoSegments seeds; seeds.reserve(rechits.size());

  std::vector<float> running_meanX; running_meanX.reserve(rechits.size());
  std::vector<float> running_meanY; running_meanY.reserve(rechits.size());

  std::vector<float> seed_minX; seed_minX.reserve(rechits.size());
  std::vector<float> seed_maxX; seed_maxX.reserve(rechits.size());
  std::vector<float> seed_minY; seed_minY.reserve(rechits.size());
  std::vector<float> seed_maxY; seed_maxY.reserve(rechits.size());

  // split rechits into subvectors and return vector of vectors:
  // Loop over rechits 
  // Create one seed per hit
  for(unsigned int i = 0; i < rechits.size(); ++i) {
    seeds.push_back(EnsembleHitContainer(1,rechits[i]));

    // First added hit in seed defines the mean to which the next hit is compared
    // for this seed.

    running_meanX.push_back( rechits[i]->localPosition().x() );
    running_meanY.push_back( rechits[i]->localPosition().y() );
	
    // set min/max X and Y for box containing the hits in the precluster:
    seed_minX.push_back( rechits[i]->localPosition().x() );
    seed_maxX.push_back( rechits[i]->localPosition().x() );
    seed_minY.push_back( rechits[i]->localPosition().y() );
    seed_maxY.push_back( rechits[i]->localPosition().y() );
  }
    
  // merge clusters that are too close
  // measure distance between final "running mean"
  for(size_t NNN = 0; NNN < seeds.size(); ++NNN) {
    for(size_t MMM = NNN+1; MMM < seeds.size(); ++MMM) {
      if(running_meanX[MMM] == running_max || running_meanX[NNN] == running_max ) {
	LogDebug("ME0SegmentAlgorithm") << "[ME0SegmentAlgorithm::clusterHits]: ALARM! Skipping used seeds, this should not happen - inform developers!";
	continue; //skip seeds that have been used 
      }
	  
      // calculate cut criteria for simple running mean distance cut:
      //dXclus = fabs(running_meanX[NNN] - running_meanX[MMM]);
      //dYclus = fabs(running_meanY[NNN] - running_meanY[MMM]);
      // calculate minmal distance between precluster boxes containing the hits:
      if ( running_meanX[NNN] > running_meanX[MMM] ) dXclus_box = seed_minX[NNN] - seed_maxX[MMM];
      else                                           dXclus_box = seed_minX[MMM] - seed_maxX[NNN];
      if ( running_meanY[NNN] > running_meanY[MMM] ) dYclus_box = seed_minY[NNN] - seed_maxY[MMM];
      else                                           dYclus_box = seed_minY[MMM] - seed_maxY[NNN];
	  
	  
      if( dXclus_box < dXclusBoxMax && dYclus_box < dYclusBoxMax ) {
	// merge clusters!
	// merge by adding seed NNN to seed MMM and erasing seed NNN
	    
	// calculate running mean for the merged seed:
	if(seeds[NNN].size()+seeds[MMM].size() != 0) {
	  running_meanX[MMM] = (running_meanX[NNN]*seeds[NNN].size() + running_meanX[MMM]*seeds[MMM].size()) / (seeds[NNN].size()+seeds[MMM].size());
	  running_meanY[MMM] = (running_meanY[NNN]*seeds[NNN].size() + running_meanY[MMM]*seeds[MMM].size()) / (seeds[NNN].size()+seeds[MMM].size());
	}

	// update min/max X and Y for box containing the hits in the merged cluster:
	if ( seed_minX[NNN] <  seed_minX[MMM] ) seed_minX[MMM] = seed_minX[NNN];
	if ( seed_maxX[NNN] >  seed_maxX[MMM] ) seed_maxX[MMM] = seed_maxX[NNN];
	if ( seed_minY[NNN] <  seed_minY[MMM] ) seed_minY[MMM] = seed_minY[NNN];
	if ( seed_maxY[NNN] >  seed_maxY[MMM] ) seed_maxY[MMM] = seed_maxY[NNN];
	    
	// add seed NNN to MMM (lower to larger number)
	seeds[MMM].insert(seeds[MMM].end(),seeds[NNN].begin(),seeds[NNN].end());
	    
	// mark seed NNN as used (at the moment just set running mean to 999999.)
	running_meanX[NNN] = running_max;
	running_meanY[NNN] = running_max;
	// we have merged a seed (NNN) to the highter seed (MMM) - need to contimue to 
	// next seed (NNN+1)
	break;
      }
    }
  }

  // hand over the final seeds to the output
  // would be more elegant if we could do the above step with 
  // erasing the merged ones, rather than the 
  for(size_t NNN = 0; NNN < seeds.size(); ++NNN) {
    if(running_meanX[NNN] == running_max) continue; //skip seeds that have been marked as used up in merging
    rechits_clusters.push_back(seeds[NNN]);
  }

  return rechits_clusters; 
}


ME0SegmentAlgorithm::ProtoSegments 
ME0SegmentAlgorithm::chainHits(const ME0Ensemble& ensemble, const EnsembleHitContainer& rechits) {

  ProtoSegments rechits_chains; 
  ProtoSegments seeds;
  seeds.reserve(rechits.size());
  std::vector<bool> usedCluster(rechits.size(),false);
  
  // split rechits into subvectors and return vector of vectors:
  // Loop over rechits
  // Create one seed per hit
  for ( unsigned int i=0; i<rechits.size(); ++i){
    if(std::abs(rechits[i]->tof()) > dTimeChainBoxMax) continue;
    seeds.push_back(EnsembleHitContainer(1,rechits[i]));
  }
  
  // merge chains that are too close ("touch" each other)
  for(size_t NNN = 0; NNN < seeds.size(); ++NNN) {
    for(size_t MMM = NNN+1; MMM < seeds.size(); ++MMM) {
      if(usedCluster[MMM] || usedCluster[NNN]){
        continue;
      }
      // all is in the way we define "good";
      // try not to "cluster" the hits but to "chain" them;
      // it does the clustering but also does a better job
      // for inclined tracks (not clustering them together;
      // crossed tracks would be still clustered together) 
      // 22.12.09: In fact it is not much more different 
      // than the "clustering", we just introduce another
      // variable in the game - Z. And it makes sense 
      // to re-introduce Y (or actually wire group mumber)
      // in a similar way as for the strip number - see
      // the code below.
      bool goodToMerge  = isGoodToMerge(ensemble, seeds[NNN], seeds[MMM]);
      if(goodToMerge){
        // merge chains!
        // merge by adding seed NNN to seed MMM and erasing seed NNN

        // add seed NNN to MMM (lower to larger number)
        seeds[MMM].insert(seeds[MMM].end(),seeds[NNN].begin(),seeds[NNN].end());

        // mark seed NNN as used
        usedCluster[NNN] = true;
        // we have merged a seed (NNN) to the highter seed (MMM) - need to contimue to
        // next seed (NNN+1)
        break;
      }

    }
  }

  // hand over the final seeds to the output
  // would be more elegant if we could do the above step with
  // erasing the merged ones, rather than the

  for(size_t NNN = 0; NNN < seeds.size(); ++NNN) {
    if(usedCluster[NNN]) continue; //skip seeds that have been marked as used up in merging
    rechits_chains.push_back(seeds[NNN]);
  }

  //***************************************************************

  return rechits_chains;
}

bool ME0SegmentAlgorithm::isGoodToMerge(const ME0Ensemble& ensemble, const EnsembleHitContainer& newChain, const EnsembleHitContainer& oldChain) {
  for(size_t iRH_new = 0;iRH_new<newChain.size();++iRH_new){
    GlobalPoint pos_new = ensemble.first->toGlobal(newChain[iRH_new]->localPosition());
          
    for(size_t iRH_old = 0;iRH_old<oldChain.size();++iRH_old){
      GlobalPoint pos_old = ensemble.first->toGlobal(oldChain[iRH_old]->localPosition());
      // to be chained, two hits need to be in neighbouring layers...
      // or better allow few missing layers (upto 3 to avoid inefficiencies);
      // however we'll not make an angle correction because it
      // worsen the situation in some of the "regular" cases 
      // (not making the correction means that the conditions for
      // forming a cluster are different if we have missing layers -
      // this could affect events at the boundaries ) 

      // to be chained, two hits need also to be "close" in phi and eta
      if (std::abs(reco::deltaPhi( float(pos_new.phi()), float(pos_old.phi()) )) >= dPhiChainBoxMax) continue;
      if (std::abs(pos_new.eta()-pos_old.eta()) >= dEtaChainBoxMax) continue;
      // and the difference in layer index should be < (nlayers-1)
      if (std::abs(newChain[iRH_new]->me0Id().layer() - oldChain[iRH_old]->me0Id().layer()) >= (ensemble.first->id().nlayers()-1)) continue;
      // and they should have a time difference compatible with the hypothesis 
      // that the rechits originate from the same particle, but were detected in different layers
      if (std::abs(newChain[iRH_new]->tof() - oldChain[iRH_old]->tof()) >= dTimeChainBoxMax) continue;

      return true;
    }
  }
  return false;
}

void ME0SegmentAlgorithm::buildSegments(const ME0Ensemble& ensemble, const EnsembleHitContainer& rechits, std::vector<ME0Segment>& me0segs) {
  if (rechits.size() < minHitsPerSegment) return;
  
#ifdef EDM_ML_DEBUG // have lines below only compiled when in debug mode 
  edm::LogVerbatim("ME0SegmentAlgorithm") << "[ME0SegmentAlgorithm::buildSegments] will now try to fit a ME0Segment from collection of "<<rechits.size()<<" ME0 RecHits";
  for (auto rh=rechits.begin(); rh!=rechits.end(); ++rh){
    auto me0id = (*rh)->me0Id();
    auto rhLP = (*rh)->localPosition();
    edm::LogVerbatim("ME0SegmentAlgorithm") << "[RecHit :: Loc x = "<<std::showpos<<std::setw(9)<<rhLP.x()<<" Loc y = "<<std::showpos<<std::setw(9)<<rhLP.y()<<" Time = "<<std::showpos<<(*rh)->tof()<<" -- "<<me0id.rawId()<<" = "<<me0id<<" ]";
  }
#endif

  MuonSegFit::MuonRecHitContainer muonRecHits;
  proto_segment.clear();
  
  // select hits from the ensemble and sort it 
  const ME0Chamber * chamber   = ensemble.first;
  for (auto rh=rechits.begin(); rh!=rechits.end();rh++){
    proto_segment.push_back(*rh);
    // for segFit - using local point in chamber frame
    const ME0EtaPartition * thePartition   = (ensemble.second.find((*rh)->me0Id()))->second;
    GlobalPoint gp = thePartition->toGlobal((*rh)->localPosition());
    const LocalPoint lp = chamber->toLocal(gp);    
    ME0RecHit *newRH = (*rh)->clone();
    newRH->setPosition(lp);

    MuonSegFit::MuonRecHitPtr trkRecHit(newRH);
    muonRecHits.push_back(trkRecHit);
  }

  // The actual fit on all hits of the vector of the selected Tracking RecHits:
  sfit_ = std::make_unique<MuonSegFit>(muonRecHits);
  bool goodfit = sfit_->fit();
  edm::LogVerbatim("ME0SegmentAlgorithm") << "[ME0SegmentAlgorithm::buildSegments] ME0Segment fit done";

  // quit function if fit was not OK  
  if(!goodfit){
    for (auto rh:muonRecHits) rh.reset();
    return;
  }
  
  // obtain all information necessary to make the segment:
  LocalPoint protoIntercept      = sfit_->intercept();
  LocalVector protoDirection     = sfit_->localdir();
  AlgebraicSymMatrix protoErrors = sfit_->covarianceMatrix(); 
  double protoChi2               = sfit_->chi2();
  // Calculate the central value and uncertainty of the segment time
  float averageTime=0.;
  for (auto rh=rechits.begin(); rh!=rechits.end(); ++rh){
    averageTime += (*rh)->tof();                                          
  }
  if(rechits.size() != 0) averageTime=averageTime/(rechits.size());
  float timeUncrt=0.;
  for (auto rh=rechits.begin(); rh!=rechits.end(); ++rh){
    timeUncrt += pow((*rh)->tof()-averageTime,2);
  }
  if(rechits.size() > 1) timeUncrt=timeUncrt/(rechits.size()-1);
  timeUncrt = sqrt(timeUncrt);

  // save all information inside GEMCSCSegment
  edm::LogVerbatim("ME0SegmentAlgorithm") << "[ME0SegmentAlgorithm::buildSegments] will now try to make ME0Segment from collection of "<<rechits.size()<<" ME0 RecHits";
  ME0Segment tmp(proto_segment, protoIntercept, protoDirection, protoErrors, protoChi2, averageTime, timeUncrt);

  edm::LogVerbatim("ME0SegmentAlgorithm") << "[ME0SegmentAlgorithm::buildSegments] ME0Segment made";
  edm::LogVerbatim("ME0SegmentAlgorithm") << "[ME0SegmentAlgorithm::buildSegments] "<<tmp;
  
  for (auto rh:muonRecHits) rh.reset();
  me0segs.push_back(tmp);
  return;
}
