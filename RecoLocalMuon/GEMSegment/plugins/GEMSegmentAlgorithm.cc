/**
 * \file GEMSegmentAlgorithm.cc
 *  based on CSCSegAlgoST.cc
 * 
 *  \authors: Piet Verwilligen, Jason Lee
 */
 
#include "GEMSegmentAlgorithm.h"
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
GEMSegmentAlgorithm::GEMSegmentAlgorithm(const edm::ParameterSet& ps) : GEMSegmentAlgorithmBase(ps), myName("GEMSegmentAlgorithm")
{
  debug                     = ps.getUntrackedParameter<bool>("GEMDebug");
  minHitsPerSegment         = ps.getParameter<unsigned int>("minHitsPerSegment");
  preClustering             = ps.getParameter<bool>("preClustering");
  dXclusBoxMax              = ps.getParameter<double>("dXclusBoxMax");
  dYclusBoxMax              = ps.getParameter<double>("dYclusBoxMax");
  preClustering_useChaining = ps.getParameter<bool>("preClusteringUseChaining");
  dPhiChainBoxMax           = ps.getParameter<double>("dPhiChainBoxMax");
  dEtaChainBoxMax           = ps.getParameter<double>("dEtaChainBoxMax");
  maxRecHitsInCluster       = ps.getParameter<int>("maxRecHitsInCluster");
  clusterOnlySameBXRecHits  = ps.getParameter<bool>("clusterOnlySameBXRecHits");

  // maybe to be used in the future ???
  // Pruning                = ps.getParameter<bool>("Pruning");
  // BrutePruning           = ps.getParameter<bool>("BrutePruning");
  // maxRecHitsInCluster is the maximal number of hits in a precluster that is being processed
  // This cut is intended to remove messy events. Currently nothing is returned if there are
  // more that maxRecHitsInCluster hits. It could be useful to return an estimate of the 
  // cluster position, which is available.
  // maxRecHitsInCluster    = ps.getParameter<int>("maxRecHitsInCluster");
  // onlyBestSegment        = ps.getParameter<bool>("onlyBestSegment");

  // CSC uses pruning to remove clearly bad hits, using as much info from the rechits as possible: charge, position, timing, ...
  // In fits with bad chi^2 they look for the worst hit (hit with abnormally large residual) 
  // if worst hit was found, refit without worst hit and select if considerably better than original fit.
}

/* Destructor
 *
 */
GEMSegmentAlgorithm::~GEMSegmentAlgorithm() {
}


std::vector<GEMSegment> GEMSegmentAlgorithm::run(const GEMEnsemble& ensemble, const EnsembleHitContainer& rechits) {

  // pre-cluster rechits and loop over all sub clusters separately
  std::vector<GEMSegment>          segments_temp;
  std::vector<GEMSegment>          segments;
  ProtoSegments rechits_clusters; // this is a collection of groups of rechits
  
  if(preClustering) {
    // run a pre-clusterer on the given rechits to split obviously separated segment seeds:
    if(preClustering_useChaining){
      // it uses X,Y,Z information; there are no configurable parameters used;
      // the X, Y, Z "cuts" are just (much) wider than reasonable high pt segments
      edm::LogVerbatim("GEMSegmentAlgorithm") << "[GEMSegmentAlgorithm::run] preClustering :: use Chaining";
      rechits_clusters = this->chainHits(ensemble, rechits );
    }
    else{
      // it uses X,Y information + configurable parameters
      edm::LogVerbatim("GEMSegmentAlgorithm") << "[GEMSegmentAlgorithm::run] Clustering";
      rechits_clusters = this->clusterHits(ensemble, rechits );
    }
    // loop over the found clusters:
      edm::LogVerbatim("GEMSegmentAlgorithm") << "[GEMSegmentAlgorithm::run] Loop over clusters and build segments";
    for(auto sub_rechits = rechits_clusters.begin(); sub_rechits !=  rechits_clusters.end(); ++sub_rechits ) {
      // clear the buffer for the subset of segments:
      segments_temp.clear();
      // build the subset of segments:
      this->buildSegments(ensemble, (*sub_rechits), segments_temp);
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
GEMSegmentAlgorithm::ProtoSegments 
GEMSegmentAlgorithm::clusterHits(const GEMEnsemble& ensemble, const EnsembleHitContainer& rechits) {

  // think how to implement BX requirement here

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

    GEMDetId rhID                  = rechits[i]->gemId();
    const GEMEtaPartition * rhEP   = (ensemble.second.find(rhID.rawId()))->second;
    if(!rhEP) throw cms::Exception("GEMEtaPartition not found") << "Corresponding GEMEtaPartition to GEMDetId: "<<rhID<<" not found in the GEMEnsemble";
    const GEMSuperChamber * rhCH   = ensemble.first;
    LocalPoint rhLP_inEtaPartFrame = rechits[i]->localPosition();
    GlobalPoint rhGP_inCMSFrame    = rhEP->toGlobal(rhLP_inEtaPartFrame);
    LocalPoint rhLP_inChamberFrame = rhCH->toLocal(rhGP_inCMSFrame);

    running_meanX.push_back(rhLP_inChamberFrame.x());
    running_meanY.push_back(rhLP_inChamberFrame.y() );
	
    // set min/max X and Y for box containing the hits in the precluster:
    seed_minX.push_back( rhLP_inChamberFrame.x() );
    seed_maxX.push_back( rhLP_inChamberFrame.x() );
    seed_minY.push_back( rhLP_inChamberFrame.y() );
    seed_maxY.push_back( rhLP_inChamberFrame.y() );

  }
    
  // merge clusters that are too close
  // measure distance between final "running mean"
  for(size_t NNN = 0; NNN < seeds.size(); ++NNN) {
    for(size_t MMM = NNN+1; MMM < seeds.size(); ++MMM) {
      if(running_meanX[MMM] == running_max || running_meanX[NNN] == running_max ) {
	LogDebug("GEMSegmentAlgorithm") << "[GEMSegmentAlgorithm::clusterHits]: ALARM! Skipping used seeds, this should not happen - inform developers!";
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


GEMSegmentAlgorithm::ProtoSegments 
GEMSegmentAlgorithm::chainHits(const GEMEnsemble& ensemble, const EnsembleHitContainer& rechits) {

  ProtoSegments rechits_chains; 
  ProtoSegments seeds;  
  seeds.reserve(rechits.size());
  std::vector<bool> usedCluster(rechits.size(),false);
  
  // split rechits into subvectors and return vector of vectors:
  // Loop over rechits
  // Create one seed per hit
  for ( unsigned int i=0; i<rechits.size(); ++i)
    seeds.push_back(EnsembleHitContainer(1,rechits[i]));
  
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

bool GEMSegmentAlgorithm::isGoodToMerge(const GEMEnsemble& ensemble, const EnsembleHitContainer& newChain, const EnsembleHitContainer& oldChain) {

  bool phiRequirementOK   = false; // once it is true in the loop, it is ok to merge
  bool etaRequirementOK   = false; // once it is true in the loop, it is ok to merge
  bool bxRequirementOK    = false; // once it is true in the loop, it is ok to merge
  
  for(size_t iRH_new = 0;iRH_new<newChain.size();++iRH_new){
    int layer_new = (newChain[iRH_new]->gemId().station() - 1)*2 + newChain[iRH_new]->gemId().layer();

    const GEMEtaPartition * rhEP   = (ensemble.second.find(newChain[iRH_new]->gemId().rawId()))->second;
    GlobalPoint pos_new = rhEP->toGlobal(newChain[iRH_new]->localPosition());
    
    for(size_t iRH_old = 0;iRH_old<oldChain.size();++iRH_old){
      int layer_old = (oldChain[iRH_old]->gemId().station() - 1)*2 + oldChain[iRH_old]->gemId().layer();
      // Layers - hits on the same layer should not be allowed ==> if abs(layer_new - layer_old) > 0 is ok. if = 0 is false
      if ( layer_new == layer_old ) return false;

      const GEMEtaPartition * oldrhEP   = (ensemble.second.find(oldChain[iRH_old]->gemId().rawId()))->second;
      GlobalPoint pos_old = oldrhEP->toGlobal(oldChain[iRH_old]->localPosition());
      
      // Eta & Phi- to be chained, two hits need also to be "close" in phi and eta      
      if(phiRequirementOK==false) phiRequirementOK = std::abs(reco::deltaPhi( float(pos_new.phi()), float(pos_old.phi()) )) < dPhiChainBoxMax;
      if(etaRequirementOK==false) etaRequirementOK = std::abs(pos_new.eta()-pos_old.eta()) < dEtaChainBoxMax;
      // and they should have a time difference compatible with the hypothesis 
      // that the rechits originate from the same particle, but were detected in different layers
      if(bxRequirementOK==false) {      
	if (!clusterOnlySameBXRecHits){
	  bxRequirementOK = true;
	}
	else {
	  if (newChain[iRH_new]->BunchX() == oldChain[iRH_old]->BunchX()) bxRequirementOK = true;
	}
      }
      
      if( phiRequirementOK && etaRequirementOK && bxRequirementOK)
        return true;
    }
  }
  return false;
}

void GEMSegmentAlgorithm::buildSegments(const GEMEnsemble& ensemble, const EnsembleHitContainer& rechits, std::vector<GEMSegment>& gemsegs) {
  if (rechits.size() < minHitsPerSegment) return;

  MuonSegFit::MuonRecHitContainer muonRecHits;
  proto_segment.clear();
  
  // select hits from the ensemble and sort it
  const GEMSuperChamber * suCh   = ensemble.first;
  for (auto rh=rechits.begin(); rh!=rechits.end();rh++){
    proto_segment.push_back(*rh);
    
    // for segFit - using local point in chamber frame
    const GEMEtaPartition * thePartition   = (ensemble.second.find((*rh)->gemId()))->second;
    GlobalPoint gp = thePartition->toGlobal((*rh)->localPosition());
    const LocalPoint lp = suCh->toLocal(gp);
    
    GEMRecHit *newRH = (*rh)->clone();
    newRH->setPosition(lp);
    MuonSegFit::MuonRecHitPtr trkRecHit(newRH);
    muonRecHits.push_back(trkRecHit);
  }
  
  #ifdef EDM_ML_DEBUG // have lines below only compiled when in debug mode 
  edm::LogVerbatim("GEMSegmentAlgorithm") << "[GEMSegmentAlgorithm::buildSegments] will now try to fit a GEMSegment from collection of "<<rechits.size()<<" GEM RecHits";
  for (auto rh=rechits.begin(); rh!=rechits.end(); ++rh){
    auto gemid = (*rh)->gemId();
    auto rhLP = (*rh)->localPosition();
    edm::LogVerbatim("GEMSegmentAlgorithm") << "[RecHit :: Loc x = "<<std::showpos<<std::setw(9)<<rhLP.x()<<" Loc y = "<<std::showpos<<std::setw(9)<<rhLP.y()
				     <<" BX = "<<std::showpos<<(*rh)->BunchX()<<" -- "<<gemid.rawId()<<" = "<<gemid<<" ]";
  }
  #endif

  // The actual fit on all hits of the vector of the selected Tracking RecHits:
  sfit_ = std::make_unique<MuonSegFit>(muonRecHits);
  bool goodfit = sfit_->fit();
  edm::LogVerbatim("GEMSegmentAlgorithm") << "[GEMSegmentAlgorithm::buildSegments] GEMSegment fit done :: fit is good = "<<goodfit;

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

  // Calculate the bunch crossing of the GEM Segment
  float bx = 0.0;
  for (auto rh=rechits.begin(); rh!=rechits.end(); ++rh){
    bx += (*rh)->BunchX();
  }
  if(rechits.size() != 0) bx=bx*1.0/(rechits.size());

  // Calculate the central value and uncertainty of the segment time
  // if we want to study impact of 2-3ns time resolution on GEM Segment 
  // (if there will be TDCs in readout and not just BX determination)
  // then implement tof() method for rechits and use this here
  /*`
  float averageTime=0.;
  for (auto rh=rechits.begin(); rh!=rechits.end(); ++rh){
    GEMEtaPartition thePartition = ensemble.second.find((*rh)->gemId));
    GlobalPoint pos = (thePartition->toGlobal((*rh)->localPosition());
    float tof = pos.mag() * 0.01 / 0.2997925 + 25.0*(*rh)->BunchX(); 
    averageTime += pos;                                          
  }
  if(rechits.size() != 0) averageTime=averageTime/(rechits.size());
  float timeUncrt=0.;
  for (auto rh=rechits.begin(); rh!=rechits.end(); ++rh){
    GEMEtaPartition thePartition = ensemble.second.find((*rh)->gemId));
    GlobalPoint pos = (thePartition->toGlobal((*rh)->localPosition());
    float tof = pos.mag() * 0.01 / 0.2997925 + 25.0*(*rh)->BunchX();
    timeUncrt += pow(tof-averageTime,2);
  }
  if(rechits.size() != 0) timeUncrt=timeUncrt/(rechits.size());
  timeUncrt = sqrt(timeUncrt);
  */

  // save all information inside GEMCSCSegment
  edm::LogVerbatim("GEMSegmentAlgorithm") << "[GEMSegmentAlgorithm::buildSegments] will now wrap fit info in GEMSegment dataformat";
  GEMSegment tmp(proto_segment, protoIntercept, protoDirection, protoErrors, protoChi2, bx);
  // GEMSegment tmp(proto_segment, protoIntercept, protoDirection, protoErrors, protoChi2, averageTime, timeUncrt);

  edm::LogVerbatim("GEMSegmentAlgorithm") << "[GEMSegmentAlgorithm::buildSegments] GEMSegment made in "<<tmp.gemDetId();
  edm::LogVerbatim("GEMSegmentAlgorithm") << "[GEMSegmentAlgorithm::buildSegments] "<<tmp;

  for (auto rh:muonRecHits) rh.reset();
  gemsegs.push_back(tmp);
}


