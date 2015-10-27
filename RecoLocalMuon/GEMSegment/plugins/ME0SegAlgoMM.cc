/**
 * \file ME0SegAlgoMM.cc
 *  based on CSCSegAlgoST.cc
 * 
 *  \authors: Marcello Maggi
 */
 
#include "ME0SegAlgoMM.h"
#include "ME0SegFit.h"

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
ME0SegAlgoMM::ME0SegAlgoMM(const edm::ParameterSet& ps) : ME0SegmentAlgorithm(ps), myName("ME0SegAlgoMM")
{
  debug                     = ps.getUntrackedParameter<bool>("ME0Debug");
  minHitsPerSegment         = ps.getParameter<unsigned int>("minHitsPerSegment");
  preClustering             = ps.getParameter<bool>("preClustering");
  dXclusBoxMax              = ps.getParameter<double>("dXclusBoxMax");
  dYclusBoxMax              = ps.getParameter<double>("dYclusBoxMax");
  preClustering_useChaining = ps.getParameter<bool>("preClusteringUseChaining");
  dPhiChainBoxMax           = ps.getParameter<double>("dPhiChainBoxMax");
  dEtaChainBoxMax           = ps.getParameter<double>("dEtaChainBoxMax");
  maxRecHitsInCluster       = ps.getParameter<int>("maxRecHitsInCluster");
}

/* Destructor
 *
 */
ME0SegAlgoMM::~ME0SegAlgoMM() {
}


std::vector<ME0Segment> ME0SegAlgoMM::run(const ME0Ensemble& ensemble, const EnsembleHitContainer& rechits) {

  theEnsemble = ensemble;
  
  ME0DetId enId((theEnsemble.first)->id());
  edm::LogVerbatim("ME0SegAlgoMM") << "[ME0SegAlgoMM::run] build segments in chamber " << enId;
  
  // pre-cluster rechits and loop over all sub clusters separately
  std::vector<ME0Segment>          segments_temp;
  std::vector<ME0Segment>          segments;
  ProtoSegments rechits_clusters; // this is a collection of groups of rechits
  
  if(preClustering) {
    // run a pre-clusterer on the given rechits to split obviously separated segment seeds:
    if(preClustering_useChaining){
      // it uses X,Y,Z information; there are no configurable parameters used;
      // the X, Y, Z "cuts" are just (much) wider than reasonable high pt segments
      rechits_clusters = this->chainHits( rechits );
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
      segments_temp = this->buildSegments( (*sub_rechits) );
      // add the found subset of segments to the collection of all segments in this chamber:
      segments.insert( segments.end(), segments_temp.begin(), segments_temp.end() );
    }
  

    return segments;
  }
  else {
    segments = this->buildSegments(rechits);
    return segments;
  }
}


// ********************************************************************;
ME0SegAlgoMM::ProtoSegments 
ME0SegAlgoMM::clusterHits(const EnsembleHitContainer& rechits) {

  ProtoSegments rechits_clusters; // this is a collection of groups of rechits

  float dXclus_box = 0.0;
  float dYclus_box = 0.0;

  EnsembleHitContainer temp;
  ProtoSegments seeds;

  std::vector<float> running_meanX;
  std::vector<float> running_meanY;

  std::vector<float> seed_minX;
  std::vector<float> seed_maxX;
  std::vector<float> seed_minY;
  std::vector<float> seed_maxY;

  // split rechits into subvectors and return vector of vectors:
  // Loop over rechits 
  // Create one seed per hit
  for(unsigned int i = 0; i < rechits.size(); ++i) {
    temp.clear();
    temp.push_back(rechits[i]);
    seeds.push_back(temp);

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
	LogDebug("ME0SegAlgoMM") << "[ME0SegAlgoMM::clusterHits]: ALARM! Skipping used seeds, this should not happen - inform developers!";
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


ME0SegAlgoMM::ProtoSegments 
ME0SegAlgoMM::chainHits(const EnsembleHitContainer& rechits) {

  ProtoSegments rechits_chains; 
  EnsembleHitContainer temp;
  ProtoSegments seeds;

  std::vector <bool> usedCluster;

  // split rechits into subvectors and return vector of vectors:
  // Loop over rechits
  // Create one seed per hit
  for(unsigned int i = 0; i < rechits.size(); ++i) {
    temp.clear();
    temp.push_back(rechits[i]);
    seeds.push_back(temp);
    usedCluster.push_back(false);
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
      bool goodToMerge  = isGoodToMerge(seeds[NNN], seeds[MMM]);
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

bool ME0SegAlgoMM::isGoodToMerge(const EnsembleHitContainer& newChain, const EnsembleHitContainer& oldChain) {

  std::vector<float> phi_new, eta_new, phi_old, eta_old;
  std::vector<int> layer_new, layer_old;

  for(size_t iRH_new = 0;iRH_new<newChain.size();++iRH_new){
    GlobalPoint pos_new = theEnsemble.first->toGlobal(newChain[iRH_new]->localPosition());
    layer_new.push_back(newChain[iRH_new]->me0Id().layer());
    phi_new.push_back(pos_new.phi());
    eta_new.push_back(pos_new.eta());
  }  
  for(size_t iRH_old = 0;iRH_old<oldChain.size();++iRH_old){
    GlobalPoint pos_old = theEnsemble.first->toGlobal(oldChain[iRH_old]->localPosition());
    layer_old.push_back(oldChain[iRH_old]->me0Id().layer());
    phi_old.push_back(pos_old.phi());
    eta_old.push_back(pos_old.eta());
  }

  for(size_t jRH_new = 0; jRH_new<phi_new.size(); ++jRH_new){
    for(size_t jRH_old = 0; jRH_old<phi_old.size(); ++jRH_old){

      // to be chained, two hits need to be in neighbouring layers...
      // or better allow few missing layers (upto 3 to avoid inefficiencies);
      // however we'll not make an angle correction because it
      // worsen the situation in some of the "regular" cases 
      // (not making the correction means that the conditions for
      // forming a cluster are different if we have missing layers -
      // this could affect events at the boundaries ) 

      // to be chained, two hits need also to be "close" in phi and eta
      bool phiRequirementOK = std::abs(reco::deltaPhi(phi_new[jRH_new],phi_old[jRH_old])) < dPhiChainBoxMax;
      bool etaRequirementOK = fabs(eta_new[jRH_new]-eta_old[jRH_old]) < dEtaChainBoxMax;
      // and the difference in layer index should be < (nlayers-1)
      bool layerRequirementOK = abs(layer_new[jRH_new]-layer_old[jRH_old]) < (theEnsemble.first->id().nlayers()-1);

      if(layerRequirementOK && phiRequirementOK && etaRequirementOK){
        return true;
      }
    }
  } 
  return false;
}





std::vector<ME0Segment> ME0SegAlgoMM::buildSegments(const EnsembleHitContainer& rechits) {
  std::vector<ME0Segment> me0segs;

  proto_segment.clear();
  // select hits from the ensemble and sort it 
  for (auto rh=rechits.begin(); rh!=rechits.end();rh++){
    proto_segment.push_back(*rh);
  }
  if (proto_segment.size() < minHitsPerSegment){
    return me0segs;
  }

  // The actual fit on all hits of the vector of the selected Tracking RecHits:
  sfit_ = std::unique_ptr<ME0SegFit>(new ME0SegFit(theEnsemble.second, rechits));
  sfit_->fit();
  edm::LogVerbatim("ME0SegAlgoMM") << "[ME0SegAlgoMM::buildSegments] ME0Segment fit done";

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
  edm::LogVerbatim("ME0SegAlgoMM") << "[ME0SegAlgoMM::buildSegments] will now try to make ME0Segment from collection of "<<rechits.size()<<" ME0 RecHits";
  ME0Segment tmp(proto_segment, protoIntercept, protoDirection, protoErrors, protoChi2, averageTime, timeUncrt);

  edm::LogVerbatim("ME0SegAlgoMM") << "[ME0SegAlgoMM::buildSegments] ME0Segment made";
  edm::LogVerbatim("ME0SegAlgoMM") << "[ME0SegAlgoMM::buildSegments] "<<tmp;

  me0segs.push_back(tmp);
  return me0segs;
}


