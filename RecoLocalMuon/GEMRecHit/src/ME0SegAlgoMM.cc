/**
 * \file ME0SegAlgoMM.cc
 *
 *  \authors: Marcello Maggi
 */
 
#include "ME0SegAlgoMM.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

/* Constructor
 *
 */
ME0SegAlgoMM::ME0SegAlgoMM(const edm::ParameterSet& ps) : ME0SegmentAlgorithm(ps), myName("ME0SegAlgoMM") {
	 
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


std::vector<ME0Segment> ME0SegAlgoMM::run(ME0Ensamble ensamble, const EnsambleHitContainer& rechits) {

  theEnsamble = ensamble;
  //  ME0DetId enId(ensambleId);
  // LogTrace("ME0SegAlgoMM") << "[ME0SegAlgoMM::run] build segments in chamber " << enId;
  
  // pre-cluster rechits and loop over all sub clusters seperately
  std::vector<ME0Segment>          segments_temp;
  std::vector<ME0Segment>          segments;
  ProtoSegments rechits_clusters; // this is a collection of groups of rechits
  
  if(preClustering) {
    // run a pre-clusterer on the given rechits to split obviously separated segment seeds:
    if(preClustering_useChaining){
      // it uses X,Y,Z information; there are no configurable parameters used;
      // the X, Y, Z "cuts" are just (much) wider than the LCT readout ones
      // (which are actually not step functions); this new code could accomodate
      // the clusterHits one below but we leave it for security and backward 
      // comparison reasons 
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
ME0SegAlgoMM::clusterHits(const EnsambleHitContainer & rechits) {

  ProtoSegments rechits_clusters; // this is a collection of groups of rechits
  //   const float dXclus_box_cut       = 4.; // seems to work reasonably 070116
  //   const float dYclus_box_cut       = 8.; // seems to work reasonably 070116

  float dXclus_box = 0.0;
  float dYclus_box = 0.0;

  EnsambleHitContainer temp;
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
      if(running_meanX[MMM] == 999999. || running_meanX[NNN] == 999999. ) {
	LogDebug("ME0Segment|ME0") << "ME0SegAlgoMM::clusterHits: Warning: Skipping used seeds, this should happen - inform developers!";
	//	std::cout<<"We should never see this line now!!!"<<std::endl;
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
	running_meanX[MMM] = (running_meanX[NNN]*seeds[NNN].size() + running_meanX[MMM]*seeds[MMM].size()) / (seeds[NNN].size()+seeds[MMM].size());
	running_meanY[MMM] = (running_meanY[NNN]*seeds[NNN].size() + running_meanY[MMM]*seeds[MMM].size()) / (seeds[NNN].size()+seeds[MMM].size());
	    
	// update min/max X and Y for box containing the hits in the merged cluster:
	if ( seed_minX[NNN] <= seed_minX[MMM] ) seed_minX[MMM] = seed_minX[NNN];
	if ( seed_maxX[NNN] >  seed_maxX[MMM] ) seed_maxX[MMM] = seed_maxX[NNN];
	if ( seed_minY[NNN] <= seed_minY[MMM] ) seed_minY[MMM] = seed_minY[NNN];
	if ( seed_maxY[NNN] >  seed_maxY[MMM] ) seed_maxY[MMM] = seed_maxY[NNN];
	    
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

  // hand over the final seeds to the output
  // would be more elegant if we could do the above step with 
  // erasing the merged ones, rather than the 
  for(size_t NNN = 0; NNN < seeds.size(); ++NNN) {
    if(running_meanX[NNN] == 999999.) continue; //skip seeds that have been marked as used up in merging
    rechits_clusters.push_back(seeds[NNN]);
  }

  return rechits_clusters; 
}


ME0SegAlgoMM::ProtoSegments 
ME0SegAlgoMM::chainHits(const EnsambleHitContainer & rechits) {

  ProtoSegments rechits_chains; 
  EnsambleHitContainer temp;
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

bool ME0SegAlgoMM::isGoodToMerge(EnsambleHitContainer & newChain, EnsambleHitContainer & oldChain) {
   for(size_t iRH_new = 0;iRH_new<newChain.size();++iRH_new){
    int layer_new = newChain[iRH_new]->me0Id().layer();     
    float phi_new = theEnsamble.first->toGlobal(newChain[iRH_new]->localPosition()).phi();
    float eta_new = theEnsamble.first->toGlobal(newChain[iRH_new]->localPosition()).eta();
    for(size_t iRH_old = 0;iRH_old<oldChain.size();++iRH_old){      
      int layer_old = oldChain[iRH_old]->me0Id().layer();
      float phi_old = theEnsamble.first->toGlobal(oldChain[iRH_old]->localPosition()).phi();
      float eta_old = theEnsamble.first->toGlobal(oldChain[iRH_old]->localPosition()).eta();
      // to be chained, two hits need to be in neighbouring layers...
      // or better allow few missing layers (upto 3 to avoid inefficiencies);
      // however we'll not make an angle correction because it
      // worsen the situation in some of the "regular" cases 
      // (not making the correction means that the conditions for
      // forming a cluster are different if we have missing layers -
      // this could affect events at the boundaries ) 
      //to be chained, two hits need also to be "close" in phi and eta
      bool layerRequirementOK = abs(layer_new-layer_old)<5;
      bool phiRequirementOK = fabs(phi_old-phi_new) < dPhiChainBoxMax;
      bool etaRequirementOK = fabs(eta_old-eta_new) < dEtaChainBoxMax;
      
      if(layerRequirementOK && phiRequirementOK && etaRequirementOK){
        return true;
      }
    }
  }
  return false;
}





std::vector<ME0Segment> ME0SegAlgoMM::buildSegments(const EnsambleHitContainer& rechits) {
  std::vector<ME0Segment> me0segs;

  proto_segment.clear();
  // select hits from the ensemble and sort it 
  for (auto rh=rechits.begin(); rh!=rechits.end();rh++){
    proto_segment.push_back(*rh);
  }
  if (proto_segment.size() < minHitsPerSegment){
    return me0segs;
  }
  // The actual fit on all hit of the protosegments;
  this->doSlopesAndChi2();
  this->fillLocalDirection();
  AlgebraicSymMatrix protoErrors = this->calculateError();
  this->flipErrors( protoErrors ); 
  ME0Segment tmp(proto_segment,protoIntercept, protoDirection, protoErrors,protoChi2);
  me0segs.push_back(tmp);
  return me0segs;
}

//Method doSlopesAndChi2
// fitSlopes() and  fillChiSquared() are always called one after the other 
// In fact the code is duplicated in the two functions (as we need 2 loops) - 
// it is much better to fix that at some point 
void ME0SegAlgoMM::doSlopesAndChi2(){
  this->fitSlopes();
  this->fillChiSquared();
}
/* Method fitSlopes
 *
 * Perform a Least Square Fit on a segment as per SK algo
 *
 */
void ME0SegAlgoMM::fitSlopes() {

  CLHEP::HepMatrix M(4,4,0);
  CLHEP::HepVector B(4,0);
  // In absence of a geometrical construction of the ME0Ensamble take layer 1  
  const ME0EtaPartition* ens = theEnsamble.first;

  for (auto ih = proto_segment.begin(); ih != proto_segment.end(); ++ih) {
    const ME0RecHit& hit = (**ih);
    const ME0EtaPartition* roll  = theEnsamble.second[hit.me0Id()];
    GlobalPoint gp         = roll->toGlobal(hit.localPosition());
    // Locat w,r,t, to the first layer;
    LocalPoint  lp         = ens->toLocal(gp); 
    // ptc: Local position of hit w.r.t. chamber
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();
    // ptc: Covariance matrix of local errors 
    CLHEP::HepMatrix IC(2,2);
    IC(1,1) = hit.localPositionError().xx();
    IC(1,2) = hit.localPositionError().xy();
    IC(2,2) = hit.localPositionError().yy();
    IC(2,1) = IC(1,2); // since Cov is symmetric
    // ptc: Invert covariance matrix (and trap if it fails!)
    int ierr = 0;
    IC.invert(ierr); // inverts in place
    if (ierr != 0) {
      LogDebug("ME0Segment|ME0") << "ME0Segment::fitSlopes: failed to invert covariance matrix=\n" << IC;      
      //       std::cout<< "ME0Segment::fitSlopes: failed to invert covariance matrix=\n" << IC << "\n"<<std::endl;
    }
    
    M(1,1) += IC(1,1);
    M(1,2) += IC(1,2);
    M(1,3) += IC(1,1) * z;
    M(1,4) += IC(1,2) * z;
    B(1)   += u * IC(1,1) + v * IC(1,2);
    
    M(2,1) += IC(2,1);
    M(2,2) += IC(2,2);
    M(2,3) += IC(2,1) * z;
    M(2,4) += IC(2,2) * z;
    B(2)   += u * IC(2,1) + v * IC(2,2);
    
    M(3,1) += IC(1,1) * z;
    M(3,2) += IC(1,2) * z;
    M(3,3) += IC(1,1) * z * z;
    M(3,4) += IC(1,2) * z * z;
    B(3)   += ( u * IC(1,1) + v * IC(1,2) ) * z;
    
    M(4,1) += IC(2,1) * z;
    M(4,2) += IC(2,2) * z;
    M(4,3) += IC(2,1) * z * z;
    M(4,4) += IC(2,2) * z * z;
    B(4)   += ( u * IC(2,1) + v * IC(2,2) ) * z;
  }
  CLHEP::HepVector p = solve(M, B);
  
  // Update member variables 
  // Note that origin has local z = 0
  protoIntercept = LocalPoint(p(1), p(2), 0.);
  protoSlope_u = p(3);
  protoSlope_v = p(4);
}
/* Method fillChiSquared
 *
 * Determine Chi^2 for the proto wire segment
 *
 */
void ME0SegAlgoMM::fillChiSquared() {
  
  double chsq = 0.; 
  const ME0EtaPartition* ens = theEnsamble.first;
  for (auto ih = proto_segment.begin(); ih != proto_segment.end(); ++ih) {
    const ME0RecHit& hit = (**ih);
    const ME0EtaPartition* roll  = theEnsamble.second[hit.me0Id()];
    GlobalPoint gp         = roll->toGlobal(hit.localPosition());
    // Locat w,r,t, to the first layer;
    LocalPoint  lp         = ens->toLocal(gp); 
    // ptc: Local position of hit w.r.t. chamber
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();
    
    double du = protoIntercept.x() + protoSlope_u * z - u;
    double dv = protoIntercept.y() + protoSlope_v * z - v;
    
    CLHEP::HepMatrix IC(2,2);
    IC(1,1) = hit.localPositionError().xx();
    //    IC(1,1) = hit.localPositionError().xx();
    IC(1,2) = hit.localPositionError().xy();
    IC(2,2) = hit.localPositionError().yy();
    IC(2,1) = IC(1,2);

    
    // Invert covariance matrix
    int ierr = 0;
    IC.invert(ierr);
    if (ierr != 0) {
      LogDebug("ME0Segment|ME0") << "ME0Segment::fillChiSquared: failed to invert covariance matrix=\n" << IC;
      //       std::cout << "ME0Segment::fillChiSquared: failed to invert covariance matrix=\n" << IC << "\n";
      
    }
    
    chsq += du*du*IC(1,1) + 2.*du*dv*IC(1,2) + dv*dv*IC(2,2);
  }

  protoChi2 = chsq;
  protoNDF = 2.*proto_segment.size() - 4;
}
/* fillLocalDirection
 *
 */
void ME0SegAlgoMM::fillLocalDirection() {
  // Always enforce direction of segment to point from IP outwards
  // (Incorrect for particles not coming from IP, of course.)
  
  double dxdz = protoSlope_u;
  double dydz = protoSlope_v;
  double dz   = 1./sqrt(1. + dxdz*dxdz + dydz*dydz);
  double dx   = dz*dxdz;
  double dy   = dz*dydz;
  LocalVector localDir(dx,dy,dz);
  
  // localDir may need sign flip to ensure it points outward from IP
  // ptc: Examine its direction and origin in global z: to point outward
  // the localDir should always have same sign as global z...
  const ME0EtaPartition* ens = theEnsamble.first;

  double globalZpos    = ( ens->toGlobal( protoIntercept ) ).z();
  double globalZdir    = ( ens->toGlobal( localDir ) ).z();
  double directionSign = globalZpos * globalZdir;
  protoDirection       = (directionSign * localDir).unit();
}

/* weightMatrix
 *   
 */
AlgebraicSymMatrix ME0SegAlgoMM::weightMatrix() {
  
  std::vector<const ME0RecHit*>::const_iterator it;
  int nhits = proto_segment.size();
  AlgebraicSymMatrix matrix(2*nhits, 0);
  int row = 0;
  
  for (it = proto_segment.begin(); it != proto_segment.end(); ++it) {
    
    const ME0RecHit& hit = (**it);
    ++row;
    matrix(row, row)   = protoChiUCorrection*hit.localPositionError().xx();
    matrix(row, row+1) = hit.localPositionError().xy();
    ++row;
    matrix(row, row-1) = hit.localPositionError().xy();
    matrix(row, row)   = hit.localPositionError().yy();
  }
  int ierr;
  matrix.invert(ierr);
  return matrix;
}


/* derivativeMatrix
 *
 */
CLHEP::HepMatrix ME0SegAlgoMM::derivativeMatrix(){
  
  int nhits = proto_segment.size();
  CLHEP::HepMatrix matrix(2*nhits, 4);
  int row = 0;
  
  const ME0EtaPartition* ens = theEnsamble.first;

  for (auto ih = proto_segment.begin(); ih != proto_segment.end(); ++ih) {
    const ME0RecHit& hit = (**ih);
    const ME0EtaPartition* roll  = theEnsamble.second[hit.me0Id()];
    GlobalPoint gp         = roll->toGlobal(hit.localPosition());
    // Locat w,r,t, to the first layer;
    LocalPoint  lp         = ens->toLocal(gp); 

    float z = lp.z();
    ++row;
    matrix(row, 1) = 1.;
    matrix(row, 3) = z;
    ++row;
    matrix(row, 2) = 1.;
    matrix(row, 4) = z;
  }
  return matrix;
}

/* calculateError*/
AlgebraicSymMatrix ME0SegAlgoMM::calculateError(){
  
  AlgebraicSymMatrix weights = this->weightMatrix();
  AlgebraicMatrix A = this->derivativeMatrix();
  
  // (AT W A)^-1                                                                                                                                                         
  // from http://www.phys.ufl.edu/~avery/fitting.html, part I                                                                                                            
  int ierr;
  AlgebraicSymMatrix result = weights.similarityT(A);
  result.invert(ierr);

  // blithely assuming the inverting never fails...                                                                                                                      
  return result;
}

void ME0SegAlgoMM::flipErrors( AlgebraicSymMatrix& a ) { 
    
  // The ME0Segment needs the error matrix re-arranged to match
  //  parameters in order (uz, vz, u0, v0) where uz, vz = slopes, u0, v0 = intercepts
    
  AlgebraicSymMatrix hold( a ); 
    
  // errors on slopes into upper left 
  a(1,1) = hold(3,3); 
  a(1,2) = hold(3,4); 
  a(2,1) = hold(4,3); 
  a(2,2) = hold(4,4); 
    
  // errors on positions into lower right 
  a(3,3) = hold(1,1); 
  a(3,4) = hold(1,2); 
  a(4,3) = hold(2,1); 
  a(4,4) = hold(2,2); 
    
  // must also interchange off-diagonal elements of off-diagonal 2x2 submatrices
  a(4,1) = hold(2,3);
  a(3,2) = hold(1,4);
  a(2,3) = hold(4,1); // = hold(1,4)
  a(1,4) = hold(3,2); // = hold(2,3)
} 


