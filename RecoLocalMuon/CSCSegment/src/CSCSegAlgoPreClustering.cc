/**
 * \file CSCSegAlgoPreClustering.cc
 *
 *  \authors: S. Stoynev  - NU
 *            I. Bloch    - FNAL
 *            E. James    - FNAL
 *
 * ported by  D. Fortin   - UCR
 *
 * See header file for description.
 */

#include "RecoLocalMuon/CSCSegment/src/CSCSegAlgoPreClustering.h"

#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>


/* Constructor
 *
 */
CSCSegAlgoPreClustering::CSCSegAlgoPreClustering(const edm::ParameterSet& ps) {
  dXclusBoxMax           = ps.getParameter<double>("dXclusBoxMax");
  dYclusBoxMax           = ps.getParameter<double>("dYclusBoxMax");
  debug                  = ps.getUntrackedParameter<bool>("CSCSegmentDebug");
}


/* Destructor:
 *
 */
CSCSegAlgoPreClustering::~CSCSegAlgoPreClustering(){

}


/* clusterHits
 *
 */
std::vector< std::vector<const CSCRecHit2D*> > 
CSCSegAlgoPreClustering::clusterHits( const CSCChamber* aChamber, const ChamberHitContainer& rechits) {

  theChamber = aChamber;

  std::vector<ChamberHitContainer> rechits_clusters; // this is a collection of groups of rechits

  //float dXclus = 0.0;
  //float dYclus = 0.0;
  float dXclus_box = 0.0;
  float dYclus_box = 0.0;

  std::vector<const CSCRecHit2D*> temp;

  std::vector< ChamberHitContainer > seeds;

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
	    std::cout<<"We should never see this line now!!!"<<std::endl;
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
	if (running_meanX[NNN] == 999999.) continue; //skip seeds that have been marked as used up in merging
	rechits_clusters.push_back(seeds[NNN]);
        mean_x = running_meanX[NNN];
        mean_y = running_meanY[NNN];
        err_x  = (seed_maxX[NNN]-seed_minX[NNN])/3.464101615; // use box size divided by sqrt(12) as position error estimate
        err_y  = (seed_maxY[NNN]-seed_minY[NNN])/3.464101615; // use box size divided by sqrt(12) as position error estimate

      }

  return rechits_clusters; 
}
