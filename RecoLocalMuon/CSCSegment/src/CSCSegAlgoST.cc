
/**
 * \File CSCSegAlgoST.cc
 *
 *  \authors: S. Stoynev - NU
 *            I. Bloch    - FNAL
 *            E. James    - FNAL
 *            A. Sakharov - WSU
 *            T. Cox - UC Davis - segment fit factored out of entangled code - Jan 2015
 */

#include "CSCSegAlgoST.h"
#include "CSCCondSegFit.h"
#include "CSCSegAlgoShowering.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "Geometry/CSCGeometry/interface/CSCLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>


/* constructor
 *
 */
CSCSegAlgoST::CSCSegAlgoST(const edm::ParameterSet& ps) : 
  CSCSegmentAlgorithm(ps), myName_("CSCSegAlgoST"), ps_(ps),  showering_(0) {
	
  debug                  = ps.getUntrackedParameter<bool>("CSCDebug");
  //  minLayersApart         = ps.getParameter<int>("minLayersApart");
  //  nSigmaFromSegment      = ps.getParameter<double>("nSigmaFromSegment");
  minHitsPerSegment      = ps.getParameter<int>("minHitsPerSegment");
  //  muonsPerChamberMax     = ps.getParameter<int>("CSCSegmentPerChamberMax");      
  //  chi2Max                = ps.getParameter<double>("chi2Max");
  dXclusBoxMax           = ps.getParameter<double>("dXclusBoxMax");
  dYclusBoxMax           = ps.getParameter<double>("dYclusBoxMax");
  preClustering          = ps.getParameter<bool>("preClustering");
  preClustering_useChaining    = ps.getParameter<bool>("preClusteringUseChaining");
  Pruning                = ps.getParameter<bool>("Pruning");
  BrutePruning           = ps.getParameter<bool>("BrutePruning");
  BPMinImprovement        = ps.getParameter<double>("BPMinImprovement");
  // maxRecHitsInCluster is the maximal number of hits in a precluster that is being processed
  // This cut is intended to remove messy events. Currently nothing is returned if there are
  // more that maxRecHitsInCluster hits. It could be useful to return an estimate of the 
  // cluster position, which is available.
  maxRecHitsInCluster    = ps.getParameter<int>("maxRecHitsInCluster");
  onlyBestSegment        = ps.getParameter<bool>("onlyBestSegment");

  hitDropLimit4Hits      = ps.getParameter<double>("hitDropLimit4Hits");
  hitDropLimit5Hits      = ps.getParameter<double>("hitDropLimit5Hits");
  hitDropLimit6Hits      = ps.getParameter<double>("hitDropLimit6Hits");
  
  yweightPenaltyThreshold      = ps.getParameter<double>("yweightPenaltyThreshold");
  yweightPenalty               = ps.getParameter<double>("yweightPenalty");
  								   			 
  curvePenaltyThreshold        = ps.getParameter<double>("curvePenaltyThreshold");
  curvePenalty                 = ps.getParameter<double>("curvePenalty");

  useShowering = ps.getParameter<bool>("useShowering");
  if (useShowering) showering_   = new CSCSegAlgoShowering( ps );

  /// Improve the covariance matrix?
  adjustCovariance_ = ps.getParameter<bool>("CorrectTheErrors");

  chi2Norm_3D_      = ps.getParameter<double>("NormChi2Cut3D");
  prePrun_          = ps.getParameter<bool>("prePrun");
  prePrunLimit_     = ps.getParameter<double>("prePrunLimit");

  if (debug) edm::LogVerbatim("CSCSegment") << "CSCSegAlgoST: with factored conditioned segment fit";
}

/* Destructor
 *
 */
CSCSegAlgoST::~CSCSegAlgoST() {
  delete showering_;
}


std::vector<CSCSegment> CSCSegAlgoST::run(const CSCChamber* aChamber, const ChamberHitContainer& rechits) {

  // Set member variable
  theChamber = aChamber; 

  LogTrace("CSCSegAlgoST") << "[CSCSegAlgoST::run] Start building segments in chamber " << theChamber->id();

  // pre-cluster rechits and loop over all sub clusters seperately
  std::vector<CSCSegment>          segments_temp;
  std::vector<CSCSegment>          segments;
  std::vector<ChamberHitContainer> rechits_clusters; // a collection of clusters of rechits

  // Define yweight penalty depending on chamber. 
  // We fixed the relative ratios, but they can be scaled by parameters:
  
  for(int a = 0; a<5; ++a) {
    for(int b = 0; b<5; ++b) {
      a_yweightPenaltyThreshold[a][b] = 0.0;
    }
  }
  
  a_yweightPenaltyThreshold[1][1] = yweightPenaltyThreshold * 10.20;
  a_yweightPenaltyThreshold[1][2] = yweightPenaltyThreshold * 14.00;
  a_yweightPenaltyThreshold[1][3] = yweightPenaltyThreshold * 20.40;
  a_yweightPenaltyThreshold[1][4] = yweightPenaltyThreshold * 10.20;
  a_yweightPenaltyThreshold[2][1] = yweightPenaltyThreshold *  7.60;
  a_yweightPenaltyThreshold[2][2] = yweightPenaltyThreshold * 20.40;
  a_yweightPenaltyThreshold[3][1] = yweightPenaltyThreshold *  7.60;
  a_yweightPenaltyThreshold[3][2] = yweightPenaltyThreshold * 20.40;
  a_yweightPenaltyThreshold[4][1] = yweightPenaltyThreshold *  6.75;
  a_yweightPenaltyThreshold[4][2] = yweightPenaltyThreshold * 20.40;
  
  if(preClustering) {
    // run a pre-clusterer on the given rechits to split clearly-separated segment seeds:
    if(preClustering_useChaining){
      // it uses X,Y,Z information; there are no configurable parameters used;
      // the X, Y, Z "cuts" are just (much) wider than the LCT readout ones
      // (which are actually not step functions); this new code could accomodate
      // the clusterHits one below but we leave it for security and backward 
      // comparison reasons 
      rechits_clusters = chainHits( theChamber, rechits );
    }
    else{
      // it uses X,Y information + configurable parameters
      rechits_clusters = clusterHits( theChamber, rechits );
    }
    // loop over the found clusters:
    for(std::vector<ChamberHitContainer>::iterator sub_rechits = rechits_clusters.begin(); sub_rechits !=  rechits_clusters.end(); ++sub_rechits ) {
      // clear the buffer for the subset of segments:
      segments_temp.clear();
      // build the subset of segments:
      segments_temp = buildSegments( (*sub_rechits) );
      // add the found subset of segments to the collection of all segments in this chamber:
      segments.insert( segments.end(), segments_temp.begin(), segments_temp.end() );
    }
    // Any pruning of hits?
    if( Pruning ) {
      segments_temp.clear(); // segments_temp needed?!?!
      segments_temp = prune_bad_hits( theChamber, segments );
      segments.clear(); // segments_temp needed?!?!
      segments.swap(segments_temp); // segments_temp needed?!?!
    }
  
    // Ganged strips in ME1/1A?
    if ( ("ME1/a" == aChamber->specs()->chamberTypeName()) && aChamber->specs()->gangedStrips() ){
    //  if ( aChamber->specs()->gangedStrips() ){
      findDuplicates(segments);
    }
    return segments;
  }
  else {
    segments = buildSegments(rechits);
    if( Pruning ) {
      segments_temp.clear(); // segments_temp needed?!?!
      segments_temp = prune_bad_hits( theChamber, segments );
      segments.clear(); // segments_temp needed?!?!
      segments.swap(segments_temp); // segments_temp needed?!?!
    }

    // Ganged strips in ME1/1A?
    if ( ("ME1/a" == aChamber->specs()->chamberTypeName()) && aChamber->specs()->gangedStrips() ){
    //  if ( aChamber->specs()->gangedStrips() ){
      findDuplicates(segments);
    }
    return segments;
    //return buildSegments(rechits); 
  }
}

// ********************************************************************;
// *** This method is meant to remove clearly bad hits, using as    ***; 
// *** much information from the chamber as possible (e.g. charge,  ***;
// *** hit position, timing, etc.)                                  ***;
// ********************************************************************;
std::vector<CSCSegment> CSCSegAlgoST::prune_bad_hits(const CSCChamber* aChamber, std::vector<CSCSegment> & segments) {
  
  //   std::cout<<"*************************************************************"<<std::endl;
  //   std::cout<<"Called prune_bad_hits in Chamber "<< theChamber->specs()->chamberTypeName()<<std::endl;
  //   std::cout<<"*************************************************************"<<std::endl;
  
  std::vector<CSCSegment>          segments_temp;
  std::vector<ChamberHitContainer> rechits_clusters; // this is a collection of groups of rechits
  
  const float chi2ndfProbMin = 1.0e-4;
  bool   use_brute_force = BrutePruning;

  int hit_nr = 0;
  int hit_nr_worst = -1;
  //int hit_nr_2ndworst = -1;
  
  for(std::vector<CSCSegment>::iterator it=segments.begin(); it != segments.end(); ++it) {
    
    // do nothing for nhit <= minHitPerSegment
    if( (*it).nRecHits() <= minHitsPerSegment ) continue;
    
    if( !use_brute_force ) {// find worst hit
      
      float chisq    = (*it).chi2();
      int nhits      = (*it).nRecHits();
      LocalPoint localPos = (*it).localPosition();
      LocalVector segDir = (*it).localDirection();
      const CSCChamber* cscchamber = theChamber;
      float globZ       ;
	  
      GlobalPoint globalPosition = cscchamber->toGlobal(localPos);
      globZ = globalPosition.z();
      
      
      if( ChiSquaredProbability((double)chisq,(double)(2*nhits-4)) < chi2ndfProbMin  ) {

	// find (rough) "residuals" (NOT excluding the hit from the fit - speed!) of hits on segment
	std::vector<CSCRecHit2D> theseRecHits = (*it).specificRecHits();
	std::vector<CSCRecHit2D>::const_iterator iRH_worst;
	//float xdist_local       = -99999.;

	float xdist_local_worst_sig = -99999.;
	float xdist_local_2ndworst_sig = -99999.;
	float xdist_local_sig       = -99999.;

	hit_nr = 0;
	hit_nr_worst = -1;
	//hit_nr_2ndworst = -1;

	for ( std::vector<CSCRecHit2D>::const_iterator iRH = theseRecHits.begin(); iRH != theseRecHits.end(); ++iRH) {
	  //mark "worst" hit:
	  
 	  //float z_at_target ;
	  //float radius      ;
	  float loc_x_at_target;
	  //float loc_y_at_target;
	  //float loc_z_at_target;

	  //z_at_target  = 0.;
	  //radius       = 0.;
	  
	  // set the z target in CMS global coordinates:
	  const CSCLayer* csclayerRH = theChamber->layer((*iRH).cscDetId().layer());
	  LocalPoint localPositionRH = (*iRH).localPosition();
	  GlobalPoint globalPositionRH = csclayerRH->toGlobal(localPositionRH);	
	  
	  LocalError rerrlocal = (*iRH).localPositionError();  
	  float xxerr = rerrlocal.xx();
	  
	  float target_z     = globalPositionRH.z();  // target z position in cm (z pos of the hit)
	  
	  if(target_z > 0.) {
	    loc_x_at_target = localPos.x() + (segDir.x()/fabs(segDir.z())*( target_z - globZ ));
	    //loc_y_at_target = localPos.y() + (segDir.y()/fabs(segDir.z())*( target_z - globZ ));
	    //loc_z_at_target = target_z;
	  }
	  else {
	    loc_x_at_target = localPos.x() + ((-1)*segDir.x()/fabs(segDir.z())*( target_z - globZ ));
	    //loc_y_at_target = localPos.y() + ((-1)*segDir.y()/fabs(segDir.z())*( target_z - globZ ));
	    //loc_z_at_target = target_z;
	  }
	  // have to transform the segments coordinates back to the local frame... how?!!!!!!!!!!!!
	  
	  //xdist_local  = fabs(localPositionRH.x() - loc_x_at_target);
	  xdist_local_sig  = fabs((localPositionRH.x() -loc_x_at_target)/(xxerr));
	  
	  if( xdist_local_sig > xdist_local_worst_sig ) {
	    xdist_local_2ndworst_sig = xdist_local_worst_sig;
	    xdist_local_worst_sig    = xdist_local_sig;
	    iRH_worst            = iRH;
	    //hit_nr_2ndworst = hit_nr_worst;
	    hit_nr_worst = hit_nr;
	  }
	  else if(xdist_local_sig > xdist_local_2ndworst_sig) {
	    xdist_local_2ndworst_sig = xdist_local_sig;
	    //hit_nr_2ndworst = hit_nr;
	  }
	  ++hit_nr;
	}

	// reset worst hit number if certain criteria apply.
	// Criteria: 2nd worst hit must be at least a factor of
	// 1.5 better than the worst in terms of sigma:
	if ( xdist_local_worst_sig / xdist_local_2ndworst_sig < 1.5 ) {
	  hit_nr_worst    = -1;
	  //hit_nr_2ndworst = -1;
	}
      }
    }

    // if worst hit was found, refit without worst hit and select if considerably better than original fit.
    // Can also use brute force: refit all n-1 hit segments and choose one over the n hit if considerably "better"
   
    std::vector< CSCRecHit2D > buffer;
    std::vector< std::vector< CSCRecHit2D > > reduced_segments;
    std::vector< CSCRecHit2D > theseRecHits = (*it).specificRecHits();
    float best_red_seg_prob = 0.0;
    // usefor chi2 1 diff   float best_red_seg_prob = 99999.;
    buffer.clear();

    if( ChiSquaredProbability((double)(*it).chi2(),(double)((2*(*it).nRecHits())-4)) < chi2ndfProbMin  ) {
	
      buffer = theseRecHits;

      // Dirty switch: here one can select to refit all possible subsets or just the one without the 
      // tagged worst hit:
      if( use_brute_force ) { // Brute force method: loop over all possible segments:
	for(size_t bi = 0; bi < buffer.size(); ++bi) {
	  reduced_segments.push_back(buffer);
	  reduced_segments[bi].erase(reduced_segments[bi].begin()+(bi),reduced_segments[bi].begin()+(bi+1));
	}
      }
      else { // More elegant but still biased: erase only worst hit
	// Comment: There is not a very strong correlation of the worst hit with the one that one should remove... 
	if( hit_nr_worst >= 0 && hit_nr_worst <= int(buffer.size())  ) {
	  // fill segment in buffer, delete worst hit
	  buffer.erase(buffer.begin()+(hit_nr_worst),buffer.begin()+(hit_nr_worst+1));
	  reduced_segments.push_back(buffer);
	}
	else {
	  // only fill segment in array, do not delete anything
	  reduced_segments.push_back(buffer);
	}
      }
    }
      
    // Loop over the subsegments and fit (only one segment if "use_brute_force" is false):
    for(size_t iSegment=0; iSegment<reduced_segments.size(); ++iSegment) {
      // loop over hits on given segment and push pointers to hits into protosegment
      protoSegment.clear();
      for(size_t m = 0; m<reduced_segments[iSegment].size(); ++m ) {
	protoSegment.push_back(&reduced_segments[iSegment][m]);
      }

      // Create fitter object
      CSCCondSegFit* segfit = new CSCCondSegFit( pset(), chamber(), protoSegment );
      condpass1 = false;
      condpass2 = false;
      segfit->setScaleXError( 1.0 );
      segfit->fit(condpass1, condpass2);

      // Attempt to handle numerical instability of the fit;
      // The same as in the build method;
      // Preprune is not applied;
      if( adjustCovariance() ){
	if(segfit->chi2()/segfit->ndof()>chi2Norm_3D_){
	  condpass1 = true;
	  segfit->fit(condpass1, condpass2);
	}
	if( (segfit->scaleXError() < 1.00005)&&(segfit->chi2()/segfit->ndof()>chi2Norm_3D_) ){
	  condpass2 = true;
	  segfit->fit(condpass1, condpass2);
	}
      }
    
      // calculate error matrix 
      //      AlgebraicSymMatrix temp2 = segfit->covarianceMatrix();

      // build an actual segment
      CSCSegment temp(protoSegment, segfit->intercept(), segfit->localdir(), 
                         segfit->covarianceMatrix(), segfit->chi2() );

      // and finished with this fit
      delete segfit;

      // n-hit segment is (*it)
      // (n-1)-hit segment is temp
      // replace n-hit segment with (n-1)-hit segment if segment probability is BPMinImprovement better
      double oldchi2 = (*it).chi2();
      double olddof  =  2 * (*it).nRecHits() - 4;
      double newchi2 = temp.chi2();
      double newdof  = 2 * temp.nRecHits() - 4;
      if( ( ChiSquaredProbability(oldchi2,olddof) < (1./BPMinImprovement)*
	    ChiSquaredProbability(newchi2,newdof) ) 
	  && ( ChiSquaredProbability(newchi2,newdof) > best_red_seg_prob )
	  && ( ChiSquaredProbability(newchi2,newdof) > 1e-10 )
	) {
	best_red_seg_prob = ChiSquaredProbability(newchi2,newdof);
        // The (n-1)- segment is better than the n-hit segment.
	// If it has at least  minHitsPerSegment  hits replace the n-hit segment
        // with this  better (n-1)-hit segment:
        if( temp.nRecHits() >= minHitsPerSegment ) {
          (*it) = temp;
        }
      }
    } // end of loop over subsegments (iSegment)
    
  } // end loop over segments (it)
  
  return segments;
  
}


// ********************************************************************;
std::vector< std::vector<const CSCRecHit2D*> > CSCSegAlgoST::clusterHits(const CSCChamber* aChamber, const ChamberHitContainer & rechits) {
  theChamber = aChamber; 

  std::vector<ChamberHitContainer> rechits_clusters; // this is a collection of groups of rechits
  //   const float dXclus_box_cut       = 4.; // seems to work reasonably 070116
  //   const float dYclus_box_cut       = 8.; // seems to work reasonably 070116

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

  //std::cout<<"*************************************************************"<<std::endl;
  //std::cout<<"Called clusterHits in Chamber "<< theChamber->specs()->chamberTypeName()<<std::endl;
  //std::cout<<"*************************************************************"<<std::endl;

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
	LogTrace("CSCSegment|CSC") << "[CSCSegAlgoST::clusterHits] ALARM! Skipping used seeds, this should not happen - inform CSC DPG";
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

  //***************************************************************

      return rechits_clusters; 
}


std::vector< std::vector<const CSCRecHit2D*> > CSCSegAlgoST::chainHits(const CSCChamber* aChamber, const ChamberHitContainer & rechits) {

  std::vector<ChamberHitContainer> rechits_chains; // this is a collection of groups of rechits


  std::vector<const CSCRecHit2D*> temp;

  std::vector< ChamberHitContainer > seeds;

  std::vector <bool> usedCluster;

  // split rechits into subvectors and return vector of vectors:
  // Loop over rechits
  // Create one seed per hit
  //std::cout<<" rechits.size() = "<<rechits.size()<<std::endl;
  for(unsigned int i = 0; i < rechits.size(); ++i) {
    temp.clear();
    temp.push_back(rechits[i]);
    seeds.push_back(temp);
    usedCluster.push_back(false);
  }
  // Only ME1/1A can have ganged strips so no need to test name
  bool gangedME11a = false;
  if ( ("ME1/a" == aChamber->specs()->chamberTypeName()) && aChamber->specs()->gangedStrips() ){
  //  if ( aChamber->specs()->gangedStrips() ){
    gangedME11a = true;
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
      bool goodToMerge  = isGoodToMerge(gangedME11a, seeds[NNN], seeds[MMM]);
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

bool CSCSegAlgoST::isGoodToMerge(bool gangedME11a, ChamberHitContainer & newChain, ChamberHitContainer & oldChain) {
  for(size_t iRH_new = 0;iRH_new<newChain.size();++iRH_new){
    int layer_new = newChain[iRH_new]->cscDetId().layer()-1;     
    int middleStrip_new = newChain[iRH_new]->nStrips()/2;
    int centralStrip_new = newChain[iRH_new]->channels(middleStrip_new);
    int centralWire_new = newChain[iRH_new]->hitWire();
    bool layerRequirementOK = false;
    bool stripRequirementOK = false;
    bool wireRequirementOK = false;
    bool goodToMerge = false;
    for(size_t iRH_old = 0;iRH_old<oldChain.size();++iRH_old){      
      int layer_old = oldChain[iRH_old]->cscDetId().layer()-1;
      int middleStrip_old = oldChain[iRH_old]->nStrips()/2;
      int centralStrip_old = oldChain[iRH_old]->channels(middleStrip_old);
      int centralWire_old = oldChain[iRH_old]->hitWire();

      // to be chained, two hits need to be in neighbouring layers...
      // or better allow few missing layers (upto 3 to avoid inefficiencies);
      // however we'll not make an angle correction because it
      // worsen the situation in some of the "regular" cases 
      // (not making the correction means that the conditions for
      // forming a cluster are different if we have missing layers -
      // this could affect events at the boundaries ) 
      if(layer_new==layer_old+1 || layer_new==layer_old-1 ||
	 layer_new==layer_old+2 || layer_new==layer_old-2 ||
	 layer_new==layer_old+3 || layer_new==layer_old-3 ||
	 layer_new==layer_old+4 || layer_new==layer_old-4 ){
        layerRequirementOK = true;
      }
      int allStrips = 48;
      //to be chained, two hits need to be "close" in strip number (can do it in phi
      // but it doesn't really matter); let "close" means upto 2 strips (3?) - 
      // this is more compared to what CLCT readout patterns allow 
      if(centralStrip_new==centralStrip_old ||
         centralStrip_new==centralStrip_old+1 || centralStrip_new==centralStrip_old-1 ||
         centralStrip_new==centralStrip_old+2 || centralStrip_new==centralStrip_old-2){
        stripRequirementOK = true;
      }
      // same for wires (and ALCT patterns)
      if(centralWire_new==centralWire_old ||
         centralWire_new==centralWire_old+1 || centralWire_new==centralWire_old-1 ||
         centralWire_new==centralWire_old+2 || centralWire_new==centralWire_old-2){
        wireRequirementOK = true;
      }

      if(gangedME11a){
	if(centralStrip_new==centralStrip_old+1-allStrips || centralStrip_new==centralStrip_old-1-allStrips ||
	   centralStrip_new==centralStrip_old+2-allStrips || centralStrip_new==centralStrip_old-2-allStrips ||
	   centralStrip_new==centralStrip_old+1+allStrips || centralStrip_new==centralStrip_old-1+allStrips ||
	   centralStrip_new==centralStrip_old+2+allStrips || centralStrip_new==centralStrip_old-2+allStrips){
	  stripRequirementOK = true;
	}
      }
      if(layerRequirementOK && stripRequirementOK && wireRequirementOK){
        goodToMerge = true;
        return goodToMerge;
      }
    }
  }
  return false;
}




double CSCSegAlgoST::theWeight(double coordinate_1, double coordinate_2, double coordinate_3, float layer_1, float layer_2, float layer_3) {
  double sub_weight = 0;
  sub_weight = fabs( 
		    ( (coordinate_2 - coordinate_3) / (layer_2  - layer_3) ) - 
		    ( (coordinate_1 - coordinate_2) / (layer_1  - layer_2) ) 
		    );
  return sub_weight;
}

/* 
 * This algorithm uses a Minimum Spanning Tree (ST) approach to build
 * endcap muon track segments from the rechits in the 6 layers of a CSC.
 */
std::vector<CSCSegment> CSCSegAlgoST::buildSegments(const ChamberHitContainer& rechits) {

  // Clear buffer for segment vector
  std::vector<CSCSegment> segmentInChamber;
  segmentInChamber.clear(); // list of final segments

  // CSC Ring;
  unsigned int thering    = 999;
  unsigned int thestation = 999;
  //unsigned int thecham    = 999;

  std::vector<int> hits_onLayerNumber(6);

  unsigned int UpperLimit = maxRecHitsInCluster;
  if (int(rechits.size()) < minHitsPerSegment) return segmentInChamber;
 
  for(int iarray = 0; iarray <6; ++iarray) { // magic number 6: number of layers in CSC chamber - not gonna change :)
    PAhits_onLayer[iarray].clear();
    hits_onLayerNumber[iarray] = 0;    
  }

  chosen_Psegments.clear();
  chosen_weight_A.clear();

  Psegments.clear();
  Psegments_noLx.clear();
  Psegments_noL1.clear();
  Psegments_noL2.clear();
  Psegments_noL3.clear();
  Psegments_noL4.clear();
  Psegments_noL5.clear();
  Psegments_noL6.clear();

  Psegments_hits.clear();
  
  weight_A.clear();
  weight_noLx_A.clear();
  weight_noL1_A.clear();
  weight_noL2_A.clear();
  weight_noL3_A.clear();
  weight_noL4_A.clear();
  weight_noL5_A.clear();
  weight_noL6_A.clear();

  weight_B.clear();
  weight_noL1_B.clear();
  weight_noL2_B.clear();
  weight_noL3_B.clear();
  weight_noL4_B.clear();
  weight_noL5_B.clear();
  weight_noL6_B.clear();

  curv_A.clear();
  curv_noL1_A.clear();
  curv_noL2_A.clear();
  curv_noL3_A.clear();
  curv_noL4_A.clear();
  curv_noL5_A.clear();
  curv_noL6_A.clear();

  // definition of middle layer for n-hit segment
  int midlayer_pointer[6] = {0,0,2,3,3,4};
  
  // int n_layers_missed_tot = 0;
  int n_layers_occupied_tot = 0;
  int n_layers_processed = 0;

  float min_weight_A = 99999.9;
  float min_weight_noLx_A = 99999.9;

  //float best_weight_B = -1.;
  //float best_weight_noLx_B = -1.;

  //float best_curv_A = -1.;
  //float best_curv_noLx_A = -1.;

  int best_pseg = -1;
  int best_noLx_pseg = -1;
  int best_Layer_noLx = -1;

  //************************************************************************;    
  //***   Start segment building   *****************************************;    
  //************************************************************************;    
  
  // Determine how many layers with hits we have
  // Fill all hits into the layer hit container:
  
  // Have 2 standard arrays: one giving the number of hits per layer. 
  // The other the corresponding hits. 
  
  // Loop all available hits, count hits per layer and fill the hits into array by layer
  for(size_t M = 0; M < rechits.size(); ++M) {
    // add hits to array per layer and count hits per layer:
    hits_onLayerNumber[ rechits[M]->cscDetId().layer()-1 ] += 1;
    if(hits_onLayerNumber[ rechits[M]->cscDetId().layer()-1 ] == 1 ) n_layers_occupied_tot += 1;
    // add hits to vector in array
    PAhits_onLayer[rechits[M]->cscDetId().layer()-1]    .push_back(rechits[M]);	   
  }
 
  // We have now counted the hits per layer and filled pointers to the hits into an array
  
  int tothits = 0;
  int maxhits = 0;
  int nexthits = 0;
  int maxlayer = -1;
  int nextlayer = -1;

  for(size_t i = 0; i< hits_onLayerNumber.size(); ++i){
    //std::cout<<"We have "<<hits_onLayerNumber[i]<<" hits on layer "<<i+1<<std::endl;
    tothits += hits_onLayerNumber[i];
    if (hits_onLayerNumber[i] > maxhits) {
      nextlayer = maxlayer;
      nexthits = maxhits;
      maxlayer = i;
      maxhits = hits_onLayerNumber[i];
    }
    else if (hits_onLayerNumber[i] > nexthits) {
      nextlayer = i;
      nexthits = hits_onLayerNumber[i];
    }
  }


  if (tothits > (int)UpperLimit) {
    if (n_layers_occupied_tot > 4) {
      tothits = tothits - hits_onLayerNumber[maxlayer];
      n_layers_occupied_tot = n_layers_occupied_tot - 1;
      PAhits_onLayer[maxlayer].clear();
      hits_onLayerNumber[maxlayer] = 0;
    }
  }

  if (tothits > (int)UpperLimit) {
    if (n_layers_occupied_tot > 4) {
      tothits = tothits - hits_onLayerNumber[nextlayer];
      n_layers_occupied_tot = n_layers_occupied_tot - 1;
      PAhits_onLayer[nextlayer].clear();
      hits_onLayerNumber[nextlayer] = 0;
    }
  }

  if (tothits > (int)UpperLimit){ 

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // Showering muon - returns nothing if chi2 == -1 (see comment in SegAlgoShowering)
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
    if (useShowering) {
      CSCSegment segShower = showering_->showerSeg(theChamber, rechits);
      if( debug ) dumpSegment( segShower );
      // Make sure have at least 3 hits...
      if ( segShower.nRecHits() < 3 ) return segmentInChamber;
      if ( segShower.chi2() == -1 ) return segmentInChamber;
      segmentInChamber.push_back(segShower);
      return segmentInChamber;  
    } 
    else {
    //  LogTrace("CSCSegment|CSC") << "[CSCSegAlgoST::buildSegments] No. of rechits in the cluster/chamber > " 
    //  << UpperLimit << " ... Segment finding in the cluster/chamber canceled!";
      CSCDetId id = rechits[0]->cscDetId();
      edm::LogVerbatim("CSCSegment|CSC") << "[CSCSegAlgoST::buildSegments] No. of rechits in the cluster/chamber > " 
					 << UpperLimit << " ... Segment finding canceled in CSC" << id;
      return segmentInChamber;  
    }
  }

  // Find out which station, ring and chamber we are in 
  // Used to choose station/ring dependant y-weight cuts

  if( rechits.size() > 0 ) {
    thering = rechits[0]->cscDetId().ring();
    thestation = rechits[0]->cscDetId().station();
    //thecham = rechits[0]->cscDetId().chamber();
  }

  // std::cout<<"We are in Station/ring/chamber: "<<thestation <<" "<< thering<<" "<< thecham<<std::endl;

  // Cut-off parameter - don't reconstruct segments with less than X hits
  if( n_layers_occupied_tot < minHitsPerSegment ) { 
    return segmentInChamber;
  }
  
  // Start building all possible hit combinations:

  // loop over the six chamber layers and form segment candidates from the available hits:

  for(int layer = 0; layer < 6; ++layer) {

    // *****************************************************************
    // *** Set missed layer counter here (not currently implemented) ***
    // *****************************************************************
    // if( PAhits_onLayer[layer].size() == 0 ) {
    //   n_layers_missed_tot += 1;
    // }

    if( PAhits_onLayer[layer].size() > 0 ) {
      n_layers_processed += 1;
    }

    // Save the size of the protosegment before hits were added on the current layer
    int orig_number_of_psegs = Psegments.size();
    int orig_number_of_noL1_psegs = Psegments_noL1.size();
    int orig_number_of_noL2_psegs = Psegments_noL2.size();
    int orig_number_of_noL3_psegs = Psegments_noL3.size();
    int orig_number_of_noL4_psegs = Psegments_noL4.size();
    int orig_number_of_noL5_psegs = Psegments_noL5.size();
    int orig_number_of_noL6_psegs = Psegments_noL6.size();

    // loop over the hits on the layer and initiate protosegments or add hits to protosegments
    for(int hit = 0; hit < int(PAhits_onLayer[layer].size()); ++hit) { // loop all hits on the Layer number "layer"

      // create protosegments from all hits on the first layer with hits
      if( orig_number_of_psegs == 0 ) { // would be faster to turn this around - ask for "orig_number_of_psegs != 0"

	Psegments_hits.push_back(PAhits_onLayer[layer][hit]);

	Psegments.push_back(Psegments_hits); 
	Psegments_noL6.push_back(Psegments_hits); 
	Psegments_noL5.push_back(Psegments_hits); 
	Psegments_noL4.push_back(Psegments_hits); 
	Psegments_noL3.push_back(Psegments_hits); 
	Psegments_noL2.push_back(Psegments_hits); 

	// Initialize weights corresponding to this segment for first hit (with 0)

	curv_A.push_back(0.0);
	curv_noL6_A.push_back(0.0); 
	curv_noL5_A.push_back(0.0); 
	curv_noL4_A.push_back(0.0); 
	curv_noL3_A.push_back(0.0); 
	curv_noL2_A.push_back(0.0); 

	weight_A.push_back(0.0);
	weight_noL6_A.push_back(0.0); 
	weight_noL5_A.push_back(0.0); 
	weight_noL4_A.push_back(0.0); 
	weight_noL3_A.push_back(0.0); 
	weight_noL2_A.push_back(0.0); 

	weight_B.push_back(0.0);
	weight_noL6_B.push_back(0.0); 
	weight_noL5_B.push_back(0.0); 
	weight_noL4_B.push_back(0.0); 
	weight_noL3_B.push_back(0.0); 
	weight_noL2_B.push_back(0.0); 
    
	// reset array for next hit on next layer
	Psegments_hits    .clear();
      }
      else {
	if( orig_number_of_noL1_psegs == 0 ) {

	  Psegments_hits.push_back(PAhits_onLayer[layer][hit]);

	  Psegments_noL1.push_back(Psegments_hits); 

	  // Initialize weight corresponding to this segment for first hit (with 0)

	  curv_noL1_A.push_back(0.0);

	  weight_noL1_A.push_back(0.0);

	  weight_noL1_B.push_back(0.0);
    
	  // reset array for next hit on next layer
	  Psegments_hits    .clear();

	}

	// loop over the protosegments and create a new protosegments for each hit-1 on this layer
	
        for( int pseg = 0; pseg < orig_number_of_psegs; ++pseg ) { 

	  int pseg_pos = (pseg)+((hit)*orig_number_of_psegs);
	  int pseg_noL1_pos = (pseg)+((hit)*orig_number_of_noL1_psegs);
	  int pseg_noL2_pos = (pseg)+((hit)*orig_number_of_noL2_psegs);
	  int pseg_noL3_pos = (pseg)+((hit)*orig_number_of_noL3_psegs);
	  int pseg_noL4_pos = (pseg)+((hit)*orig_number_of_noL4_psegs);
	  int pseg_noL5_pos = (pseg)+((hit)*orig_number_of_noL5_psegs);
	  int pseg_noL6_pos = (pseg)+((hit)*orig_number_of_noL6_psegs);

	  // - Loop all psegs. 
	  // - If not last hit, clone  existing protosegments  (PAhits_onLayer[layer].size()-1) times
	  // - then add the new hits

	  if( ! (hit == int(PAhits_onLayer[layer].size()-1)) ) { // not the last hit - prepare (copy) new protosegments for the following hits
	    // clone psegs (to add next hits or last hit on layer):

	    Psegments.push_back(Psegments[ pseg_pos ]); 
	    if (n_layers_processed != 2 && pseg < orig_number_of_noL1_psegs) Psegments_noL1.push_back(Psegments_noL1[ pseg_noL1_pos ]); 
	    if (n_layers_processed != 2 && pseg < orig_number_of_noL2_psegs) Psegments_noL2.push_back(Psegments_noL2[ pseg_noL2_pos ]); 
	    if (n_layers_processed != 3 && pseg < orig_number_of_noL3_psegs) Psegments_noL3.push_back(Psegments_noL3[ pseg_noL3_pos ]); 
	    if (n_layers_processed != 4 && pseg < orig_number_of_noL4_psegs) Psegments_noL4.push_back(Psegments_noL4[ pseg_noL4_pos ]); 
	    if (n_layers_processed != 5 && pseg < orig_number_of_noL5_psegs) Psegments_noL5.push_back(Psegments_noL5[ pseg_noL5_pos ]); 
	    if (n_layers_processed != 6 && pseg < orig_number_of_noL6_psegs) Psegments_noL6.push_back(Psegments_noL6[ pseg_noL6_pos ]); 
	    // clone weight corresponding to this segment too
	    weight_A.push_back(weight_A[ pseg_pos ]);
	    if (n_layers_processed != 2 && pseg < orig_number_of_noL1_psegs) weight_noL1_A.push_back(weight_noL1_A[ pseg_noL1_pos ]);
	    if (n_layers_processed != 2 && pseg < orig_number_of_noL2_psegs) weight_noL2_A.push_back(weight_noL2_A[ pseg_noL2_pos ]);
	    if (n_layers_processed != 3 && pseg < orig_number_of_noL3_psegs) weight_noL3_A.push_back(weight_noL3_A[ pseg_noL3_pos ]);
	    if (n_layers_processed != 4 && pseg < orig_number_of_noL4_psegs) weight_noL4_A.push_back(weight_noL4_A[ pseg_noL4_pos ]);
	    if (n_layers_processed != 5 && pseg < orig_number_of_noL5_psegs) weight_noL5_A.push_back(weight_noL5_A[ pseg_noL5_pos ]);
	    if (n_layers_processed != 6 && pseg < orig_number_of_noL6_psegs) weight_noL6_A.push_back(weight_noL6_A[ pseg_noL6_pos ]);
	    // clone curvature variable corresponding to this segment too
	    curv_A.push_back(curv_A[ pseg_pos ]);
	    if (n_layers_processed != 2 && pseg < orig_number_of_noL1_psegs) curv_noL1_A.push_back(curv_noL1_A[ pseg_noL1_pos ]);
	    if (n_layers_processed != 2 && pseg < orig_number_of_noL2_psegs) curv_noL2_A.push_back(curv_noL2_A[ pseg_noL2_pos ]);
	    if (n_layers_processed != 3 && pseg < orig_number_of_noL3_psegs) curv_noL3_A.push_back(curv_noL3_A[ pseg_noL3_pos ]);
	    if (n_layers_processed != 4 && pseg < orig_number_of_noL4_psegs) curv_noL4_A.push_back(curv_noL4_A[ pseg_noL4_pos ]);
	    if (n_layers_processed != 5 && pseg < orig_number_of_noL5_psegs) curv_noL5_A.push_back(curv_noL5_A[ pseg_noL5_pos ]);
	    if (n_layers_processed != 6 && pseg < orig_number_of_noL6_psegs) curv_noL6_A.push_back(curv_noL6_A[ pseg_noL6_pos ]);
	    // clone "y"-weight corresponding to this segment too
	    weight_B.push_back(weight_B[ pseg_pos ]);
	    if (n_layers_processed != 2 && pseg < orig_number_of_noL1_psegs) weight_noL1_B.push_back(weight_noL1_B[ pseg_noL1_pos ]);
	    if (n_layers_processed != 2 && pseg < orig_number_of_noL2_psegs) weight_noL2_B.push_back(weight_noL2_B[ pseg_noL2_pos ]);
	    if (n_layers_processed != 3 && pseg < orig_number_of_noL3_psegs) weight_noL3_B.push_back(weight_noL3_B[ pseg_noL3_pos ]);
	    if (n_layers_processed != 4 && pseg < orig_number_of_noL4_psegs) weight_noL4_B.push_back(weight_noL4_B[ pseg_noL4_pos ]);
	    if (n_layers_processed != 5 && pseg < orig_number_of_noL5_psegs) weight_noL5_B.push_back(weight_noL5_B[ pseg_noL5_pos ]);
	    if (n_layers_processed != 6 && pseg < orig_number_of_noL6_psegs) weight_noL6_B.push_back(weight_noL6_B[ pseg_noL6_pos ]);
	  }
	  // add hits to original pseg:
	  Psegments[ pseg_pos ].push_back(PAhits_onLayer[ layer ][ hit ]);
	  if (n_layers_processed != 2 && pseg < orig_number_of_noL1_psegs) Psegments_noL1[ pseg_noL1_pos ].push_back(PAhits_onLayer[ layer ][ hit ]);
	  if (n_layers_processed != 2 && pseg < orig_number_of_noL2_psegs) Psegments_noL2[ pseg_noL2_pos ].push_back(PAhits_onLayer[ layer ][ hit ]);
	  if (n_layers_processed != 3 && pseg < orig_number_of_noL3_psegs) Psegments_noL3[ pseg_noL3_pos ].push_back(PAhits_onLayer[ layer ][ hit ]);
	  if (n_layers_processed != 4 && pseg < orig_number_of_noL4_psegs) Psegments_noL4[ pseg_noL4_pos ].push_back(PAhits_onLayer[ layer ][ hit ]);
	  if (n_layers_processed != 5 && pseg < orig_number_of_noL5_psegs) Psegments_noL5[ pseg_noL5_pos ].push_back(PAhits_onLayer[ layer ][ hit ]);
	  if (n_layers_processed != 6 && pseg < orig_number_of_noL6_psegs) Psegments_noL6[ pseg_noL6_pos ].push_back(PAhits_onLayer[ layer ][ hit ]);
            
	  // calculate/update the weight (only for >2 hits on psegment):

	  if( Psegments[ pseg_pos ].size() > 2 ) {
              
	    // looks more exciting than it is. Here the weight is calculated. It is the difference in x of the last two and one but the last two hits, 
            // divided by the distance of the corresponding hits. Please refer to twiki page XXXX or CMS Note YYY (and use layer_distance)

	    weight_A[ pseg_pos ] += theWeight(
					      (*(Psegments[ pseg_pos ].end()-1 ))->localPosition().x(), 
					      (*(Psegments[ pseg_pos ].end()-2 ))->localPosition().x(),
					      (*(Psegments[ pseg_pos ].end()-3 ))->localPosition().x(),
					      float((*(Psegments[ pseg_pos ].end()-1))->cscDetId().layer()),
					      float((*(Psegments[ pseg_pos ].end()-2))->cscDetId().layer()),
					      float((*(Psegments[ pseg_pos ].end()-3))->cscDetId().layer())
					      );

	    weight_B[ pseg_pos ] += theWeight(
					      (*(Psegments[ pseg_pos ].end()-1 ))->localPosition().y(), 
					      (*(Psegments[ pseg_pos ].end()-2 ))->localPosition().y(),
					      (*(Psegments[ pseg_pos ].end()-3 ))->localPosition().y(),
					      float((*(Psegments[ pseg_pos ].end()-1))->cscDetId().layer()),
					      float((*(Psegments[ pseg_pos ].end()-2))->cscDetId().layer()),
					      float((*(Psegments[ pseg_pos ].end()-3))->cscDetId().layer())
					      );

	    // if we have picked up the last hit go looking for pseg with the lowest (and second lowest?) weight

	    if(int(Psegments[ pseg_pos ].size()) == n_layers_occupied_tot) {

  	      curv_A[ pseg_pos ] += theWeight(
					      (*(Psegments[ pseg_pos ].end()-1 ))->localPosition().x(), 
					      (*(Psegments[ pseg_pos ].end()-midlayer_pointer[n_layers_occupied_tot-1] ))->localPosition().x(),
					      (*(Psegments[ pseg_pos ].end()-n_layers_occupied_tot ))->localPosition().x(),
					      float((*(Psegments[ pseg_pos ].end()-1))->cscDetId().layer()),
					      float((*(Psegments[ pseg_pos ].end()-midlayer_pointer[n_layers_occupied_tot-1] ))->cscDetId().layer()),
					      float((*(Psegments[ pseg_pos ].end()-n_layers_occupied_tot ))->cscDetId().layer())
					      );

              if (curv_A[ pseg_pos ] > curvePenaltyThreshold) weight_A[ pseg_pos ] = weight_A[ pseg_pos ] * curvePenalty;

	      if (weight_B[ pseg_pos ] > a_yweightPenaltyThreshold[thestation][thering]) weight_A[ pseg_pos ] = weight_A[ pseg_pos ] * yweightPenalty;
               
              if (weight_A[ pseg_pos ] < min_weight_A ) {
	        min_weight_A = weight_A[ pseg_pos ];
                //best_weight_B = weight_B[ pseg_pos ];
                //best_curv_A = curv_A[ pseg_pos ];
	        best_pseg = pseg_pos ;
              }

	    }

	    // alternative: fill map with weight and pseg (which is already ordered)? Seems a very good tool to go looking for segments from. 
            // As I understand, the segments would be inserted according to their weight, so the list would "automatically" be sorted.

	  }

          if ( n_layers_occupied_tot > 3 ) {
	    if (pseg < orig_number_of_noL1_psegs && (n_layers_processed != 2)) {
	      if(( Psegments_noL1[ pseg_noL1_pos ].size() > 2 ) ) {
		
		// looks more exciting than it is. Here the weight is calculated. It is the difference in x of the last two and one but the last two hits, 
		// divided by the distance of the corresponding hits. Please refer to twiki page XXXX or CMS Note YYY (and use layer_distance)
		
		weight_noL1_A[ pseg_noL1_pos ] += theWeight(
							    (*(Psegments_noL1[ pseg_noL1_pos ].end()-1 ))->localPosition().x(), 
							    (*(Psegments_noL1[ pseg_noL1_pos ].end()-2 ))->localPosition().x(),
							    (*(Psegments_noL1[ pseg_noL1_pos ].end()-3 ))->localPosition().x(),
							    float((*(Psegments_noL1[ pseg_noL1_pos ].end()-1))->cscDetId().layer()),
							    float((*(Psegments_noL1[ pseg_noL1_pos ].end()-2))->cscDetId().layer()),
							    float((*(Psegments_noL1[ pseg_noL1_pos ].end()-3))->cscDetId().layer())
							    );

		weight_noL1_B[ pseg_noL1_pos ] += theWeight(
							    (*(Psegments_noL1[ pseg_noL1_pos ].end()-1 ))->localPosition().y(), 
							    (*(Psegments_noL1[ pseg_noL1_pos ].end()-2 ))->localPosition().y(),
							    (*(Psegments_noL1[ pseg_noL1_pos ].end()-3 ))->localPosition().y(),
							    float((*(Psegments_noL1[ pseg_noL1_pos ].end()-1))->cscDetId().layer()),
							    float((*(Psegments_noL1[ pseg_noL1_pos ].end()-2))->cscDetId().layer()),
							    float((*(Psegments_noL1[ pseg_noL1_pos ].end()-3))->cscDetId().layer())
							    );

		//if we have picked up the last hit go looking for pseg with the lowest (and second lowest?) weight

		if(int(Psegments_noL1[ pseg_noL1_pos ].size()) == n_layers_occupied_tot -1 ) {

		  curv_noL1_A[ pseg_noL1_pos ] += theWeight(
							    (*(Psegments_noL1[ pseg_noL1_pos ].end()-1 ))->localPosition().x(), 
							    (*(Psegments_noL1[ pseg_noL1_pos ].end()-midlayer_pointer[n_layers_occupied_tot-2] ))->localPosition().x(),
							    (*(Psegments_noL1[ pseg_noL1_pos ].end()-(n_layers_occupied_tot-1) ))->localPosition().x(),
							    float((*(Psegments_noL1[ pseg_noL1_pos ].end()-1 ))->cscDetId().layer()),
							    float((*(Psegments_noL1[ pseg_noL1_pos ].end()-midlayer_pointer[n_layers_occupied_tot-2] ))->cscDetId().layer()),
							    float((*(Psegments_noL1[ pseg_noL1_pos ].end()-(n_layers_occupied_tot-1) ))->cscDetId().layer())
							    );

		  if (curv_noL1_A[ pseg_noL1_pos ] > curvePenaltyThreshold) weight_noL1_A[ pseg_noL1_pos ] = weight_noL1_A[ pseg_noL1_pos ] * curvePenalty;

		  if (weight_noL1_B[ pseg_noL1_pos ] > a_yweightPenaltyThreshold[thestation][thering]) 
		    weight_noL1_A[ pseg_noL1_pos ] = weight_noL1_A[ pseg_noL1_pos ] * yweightPenalty;

		  if (weight_noL1_A[ pseg_noL1_pos ] < min_weight_noLx_A ) {
		    min_weight_noLx_A = weight_noL1_A[ pseg_noL1_pos ];
		    //best_weight_noLx_B = weight_noL1_B[ pseg_noL1_pos ];
		    //best_curv_noLx_A = curv_noL1_A[ pseg_noL1_pos ];
		    best_noLx_pseg = pseg_noL1_pos;
                    best_Layer_noLx = 1;
		  }

		}

		// alternative: fill map with weight and pseg (which is already ordered)? Seems a very good tool to go looking for segments from. 
		// As I understand, the segments would be inserted according to their weight, so the list would "automatically" be sorted.
		
	      }
	    }
	  }

          if ( n_layers_occupied_tot > 3 ) {
	    if (pseg < orig_number_of_noL2_psegs && ( n_layers_processed != 2 )) {
	      if(( Psegments_noL2[ pseg_noL2_pos ].size() > 2 )) {
              
		// looks more exciting than it is. Here the weight is calculated. It is the difference in x of the last two and one but the last two hits, 
		// divided by the distance of the corresponding hits. Please refer to twiki page XXXX or CMS Note YYY (and use layer_distance)

		weight_noL2_A[ pseg_noL2_pos ] += theWeight(
							    (*(Psegments_noL2[ pseg_noL2_pos ].end()-1 ))->localPosition().x(), 
							    (*(Psegments_noL2[ pseg_noL2_pos ].end()-2 ))->localPosition().x(),
							    (*(Psegments_noL2[ pseg_noL2_pos ].end()-3 ))->localPosition().x(),
							    float((*(Psegments_noL2[ pseg_noL2_pos ].end()-1))->cscDetId().layer()),
							    float((*(Psegments_noL2[ pseg_noL2_pos ].end()-2))->cscDetId().layer()),
							    float((*(Psegments_noL2[ pseg_noL2_pos ].end()-3))->cscDetId().layer())
							    );

		weight_noL2_B[ pseg_noL2_pos ] += theWeight(
							    (*(Psegments_noL2[ pseg_noL2_pos ].end()-1 ))->localPosition().y(), 
							    (*(Psegments_noL2[ pseg_noL2_pos ].end()-2 ))->localPosition().y(),
							    (*(Psegments_noL2[ pseg_noL2_pos ].end()-3 ))->localPosition().y(),
							    float((*(Psegments_noL2[ pseg_noL2_pos ].end()-1))->cscDetId().layer()),
							    float((*(Psegments_noL2[ pseg_noL2_pos ].end()-2))->cscDetId().layer()),
							    float((*(Psegments_noL2[ pseg_noL2_pos ].end()-3))->cscDetId().layer())
							    );

		//if we have picked up the last hit go looking for pseg with the lowest (and second lowest?) weight

		if(int(Psegments_noL2[ pseg_noL2_pos ].size()) == n_layers_occupied_tot -1 ) {

		  curv_noL2_A[ pseg_noL2_pos ] += theWeight(
							    (*(Psegments_noL2[ pseg_noL2_pos ].end()-1 ))->localPosition().x(), 
							    (*(Psegments_noL2[ pseg_noL2_pos ].end()-midlayer_pointer[n_layers_occupied_tot-2] ))->localPosition().x(),
							    (*(Psegments_noL2[ pseg_noL2_pos ].end()-(n_layers_occupied_tot-1) ))->localPosition().x(),
							    float((*(Psegments_noL2[ pseg_noL2_pos ].end()-1 ))->cscDetId().layer()),
							    float((*(Psegments_noL2[ pseg_noL2_pos ].end()-midlayer_pointer[n_layers_occupied_tot-2] ))->cscDetId().layer()),
							    float((*(Psegments_noL2[ pseg_noL2_pos ].end()-(n_layers_occupied_tot-1) ))->cscDetId().layer())
							    );

		  if (curv_noL2_A[ pseg_noL2_pos ] > curvePenaltyThreshold) weight_noL2_A[ pseg_noL2_pos ] = weight_noL2_A[ pseg_noL2_pos ] * curvePenalty;

		  if (weight_noL2_B[ pseg_noL2_pos ] > a_yweightPenaltyThreshold[thestation][thering]) 
		    weight_noL2_A[ pseg_noL2_pos ] = weight_noL2_A[ pseg_noL2_pos ] * yweightPenalty;

		  if (weight_noL2_A[ pseg_noL2_pos ] < min_weight_noLx_A ) {
		    min_weight_noLx_A = weight_noL2_A[ pseg_noL2_pos ];
		    //best_weight_noLx_B = weight_noL2_B[ pseg_noL2_pos ];
		    //best_curv_noLx_A = curv_noL2_A[ pseg_noL2_pos ];
		    best_noLx_pseg = pseg_noL2_pos;
                    best_Layer_noLx = 2;
		  }

		}

		// alternative: fill map with weight and pseg (which is already ordered)? Seems a very good tool to go looking for segments from. 
		// As I understand, the segments would be inserted according to their weight, so the list would "automatically" be sorted.

	      }
	    }
	  }

          if ( n_layers_occupied_tot > 3 ) {
	    if (pseg < orig_number_of_noL3_psegs && ( n_layers_processed != 3 )) {
	      if(( Psegments_noL3[ pseg_noL3_pos ].size() > 2 )) {
              
		// looks more exciting than it is. Here the weight is calculated. It is the difference in x of the last two and one but the last two hits, 
		// divided by the distance of the corresponding hits. Please refer to twiki page XXXX or CMS Note YYY (and use layer_distance)

		weight_noL3_A[ pseg_noL3_pos ] += theWeight(
							    (*(Psegments_noL3[ pseg_noL3_pos ].end()-1 ))->localPosition().x(), 
							    (*(Psegments_noL3[ pseg_noL3_pos ].end()-2 ))->localPosition().x(),
							    (*(Psegments_noL3[ pseg_noL3_pos ].end()-3 ))->localPosition().x(),
							    float((*(Psegments_noL3[ pseg_noL3_pos ].end()-1))->cscDetId().layer()),
							    float((*(Psegments_noL3[ pseg_noL3_pos ].end()-2))->cscDetId().layer()),
							    float((*(Psegments_noL3[ pseg_noL3_pos ].end()-3))->cscDetId().layer())
							    );

		weight_noL3_B[ pseg_noL3_pos ] += theWeight(
							    (*(Psegments_noL3[ pseg_noL3_pos ].end()-1 ))->localPosition().y(), 
							    (*(Psegments_noL3[ pseg_noL3_pos ].end()-2 ))->localPosition().y(),
							    (*(Psegments_noL3[ pseg_noL3_pos ].end()-3 ))->localPosition().y(),
							    float((*(Psegments_noL3[ pseg_noL3_pos ].end()-1))->cscDetId().layer()),
							    float((*(Psegments_noL3[ pseg_noL3_pos ].end()-2))->cscDetId().layer()),
							    float((*(Psegments_noL3[ pseg_noL3_pos ].end()-3))->cscDetId().layer())
							    );

		//if we have picked up the last hit go looking for pseg with the lowest (and second lowest?) weight

		if(int(Psegments_noL3[ pseg_noL3_pos ].size()) == n_layers_occupied_tot -1 ) {

		  curv_noL3_A[ pseg_noL3_pos ] += theWeight(
							    (*(Psegments_noL3[ pseg_noL3_pos ].end()-1 ))->localPosition().x(), 
							    (*(Psegments_noL3[ pseg_noL3_pos ].end()-midlayer_pointer[n_layers_occupied_tot-2] ))->localPosition().x(),
							    (*(Psegments_noL3[ pseg_noL3_pos ].end()-(n_layers_occupied_tot-1) ))->localPosition().x(),
							    float((*(Psegments_noL3[ pseg_noL3_pos ].end()-1 ))->cscDetId().layer()),
							    float((*(Psegments_noL3[ pseg_noL3_pos ].end()-midlayer_pointer[n_layers_occupied_tot-2] ))->cscDetId().layer()),
							    float((*(Psegments_noL3[ pseg_noL3_pos ].end()-(n_layers_occupied_tot-1) ))->cscDetId().layer())
							    );

		  if (curv_noL3_A[ pseg_noL3_pos ] > curvePenaltyThreshold) weight_noL3_A[ pseg_noL3_pos ] = weight_noL3_A[ pseg_noL3_pos ] * curvePenalty;

		  if (weight_noL3_B[ pseg_noL3_pos ] > a_yweightPenaltyThreshold[thestation][thering]) 
		    weight_noL3_A[ pseg_noL3_pos ] = weight_noL3_A[ pseg_noL3_pos ] * yweightPenalty;

		  if (weight_noL3_A[ pseg_noL3_pos ] < min_weight_noLx_A ) {
		    min_weight_noLx_A = weight_noL3_A[ pseg_noL3_pos ];
		    //best_weight_noLx_B = weight_noL3_B[ pseg_noL3_pos ];
		    //best_curv_noLx_A = curv_noL3_A[ pseg_noL3_pos ];
		    best_noLx_pseg = pseg_noL3_pos;
                    best_Layer_noLx = 3;
		  }

		}

		// alternative: fill map with weight and pseg (which is already ordered)? Seems a very good tool to go looking for segments from. 
		// As I understand, the segments would be inserted according to their weight, so the list would "automatically" be sorted.

	      }
	    }
	  }

          if ( n_layers_occupied_tot > 3 ) {
	    if (pseg < orig_number_of_noL4_psegs && ( n_layers_processed != 4 )) {
	      if(( Psegments_noL4[ pseg_noL4_pos ].size() > 2 )) {
              
		// looks more exciting than it is. Here the weight is calculated. It is the difference in x of the last two and one but the last two hits, 
		// divided by the distance of the corresponding hits. Please refer to twiki page XXXX or CMS Note YYY (and use layer_distance)
	      
		weight_noL4_A[ pseg_noL4_pos ] += theWeight(
							    (*(Psegments_noL4[ pseg_noL4_pos ].end()-1 ))->localPosition().x(), 
							    (*(Psegments_noL4[ pseg_noL4_pos ].end()-2 ))->localPosition().x(),
							    (*(Psegments_noL4[ pseg_noL4_pos ].end()-3 ))->localPosition().x(),
							    float((*(Psegments_noL4[ pseg_noL4_pos ].end()-1))->cscDetId().layer()),
							    float((*(Psegments_noL4[ pseg_noL4_pos ].end()-2))->cscDetId().layer()),
							    float((*(Psegments_noL4[ pseg_noL4_pos ].end()-3))->cscDetId().layer())
							    );

		weight_noL4_B[ pseg_noL4_pos ] += theWeight(
							    (*(Psegments_noL4[ pseg_noL4_pos ].end()-1 ))->localPosition().y(), 
							    (*(Psegments_noL4[ pseg_noL4_pos ].end()-2 ))->localPosition().y(),
							    (*(Psegments_noL4[ pseg_noL4_pos ].end()-3 ))->localPosition().y(),
							    float((*(Psegments_noL4[ pseg_noL4_pos ].end()-1))->cscDetId().layer()),
							    float((*(Psegments_noL4[ pseg_noL4_pos ].end()-2))->cscDetId().layer()),
							    float((*(Psegments_noL4[ pseg_noL4_pos ].end()-3))->cscDetId().layer())
							    );

		//if we have picked up the last hit go looking for pseg with the lowest (and second lowest?) weight

		if(int(Psegments_noL4[ pseg_noL4_pos ].size()) == n_layers_occupied_tot -1 ) {

		  curv_noL4_A[ pseg_noL4_pos ] += theWeight(
							    (*(Psegments_noL4[ pseg_noL4_pos ].end()-1 ))->localPosition().x(), 
							    (*(Psegments_noL4[ pseg_noL4_pos ].end()-midlayer_pointer[n_layers_occupied_tot-2] ))->localPosition().x(),
							    (*(Psegments_noL4[ pseg_noL4_pos ].end()-(n_layers_occupied_tot-1) ))->localPosition().x(),
							    float((*(Psegments_noL4[ pseg_noL4_pos ].end()-1 ))->cscDetId().layer()),
							    float((*(Psegments_noL4[ pseg_noL4_pos ].end()-midlayer_pointer[n_layers_occupied_tot-2] ))->cscDetId().layer()),
							    float((*(Psegments_noL4[ pseg_noL4_pos ].end()-(n_layers_occupied_tot-1) ))->cscDetId().layer())
							    );

		  if (curv_noL4_A[ pseg_noL4_pos ] > curvePenaltyThreshold) weight_noL4_A[ pseg_noL4_pos ] = weight_noL4_A[ pseg_noL4_pos ] * curvePenalty;

		  if (weight_noL4_B[ pseg_noL4_pos ] > a_yweightPenaltyThreshold[thestation][thering]) 
		    weight_noL4_A[ pseg_noL4_pos ] = weight_noL4_A[ pseg_noL4_pos ] * yweightPenalty;

		  if (weight_noL4_A[ pseg_noL4_pos ] < min_weight_noLx_A ) {
		    min_weight_noLx_A = weight_noL4_A[ pseg_noL4_pos ];
		    //best_weight_noLx_B = weight_noL4_B[ pseg_noL4_pos ];
		    //best_curv_noLx_A = curv_noL4_A[ pseg_noL4_pos ];
		    best_noLx_pseg = pseg_noL4_pos;
                    best_Layer_noLx = 4;
		  }

		}

		// alternative: fill map with weight and pseg (which is already ordered)? Seems a very good tool to go looking for segments from. 
		// As I understand, the segments would be inserted according to their weight, so the list would "automatically" be sorted.

	      }
	    }
	  }

          if ( n_layers_occupied_tot > 4 ) {
	    if (pseg < orig_number_of_noL5_psegs && ( n_layers_processed != 5 )) {
	      if(( Psegments_noL5[ pseg_noL5_pos ].size() > 2 )){
              
		// looks more exciting than it is. Here the weight is calculated. It is the difference in x of the last two and one but the last two hits, 
		// divided by the distance of the corresponding hits. Please refer to twiki page XXXX or CMS Note YYY (and use layer_distance)

		weight_noL5_A[ pseg_noL5_pos ] += theWeight(
							    (*(Psegments_noL5[ pseg_noL5_pos ].end()-1 ))->localPosition().x(), 
							    (*(Psegments_noL5[ pseg_noL5_pos ].end()-2 ))->localPosition().x(),
							    (*(Psegments_noL5[ pseg_noL5_pos ].end()-3 ))->localPosition().x(),
							    float((*(Psegments_noL5[ pseg_noL5_pos ].end()-1))->cscDetId().layer()),
							    float((*(Psegments_noL5[ pseg_noL5_pos ].end()-2))->cscDetId().layer()),
							    float((*(Psegments_noL5[ pseg_noL5_pos ].end()-3))->cscDetId().layer())
							    );

		weight_noL5_B[ pseg_noL5_pos ] += theWeight(
							    (*(Psegments_noL5[ pseg_noL5_pos ].end()-1 ))->localPosition().y(), 
							    (*(Psegments_noL5[ pseg_noL5_pos ].end()-2 ))->localPosition().y(),
							    (*(Psegments_noL5[ pseg_noL5_pos ].end()-3 ))->localPosition().y(),
							    float((*(Psegments_noL5[ pseg_noL5_pos ].end()-1))->cscDetId().layer()),
							    float((*(Psegments_noL5[ pseg_noL5_pos ].end()-2))->cscDetId().layer()),
							    float((*(Psegments_noL5[ pseg_noL5_pos ].end()-3))->cscDetId().layer())
							    );

		//if we have picked up the last hit go looking for pseg with the lowest (and second lowest?) weight

		if(int(Psegments_noL5[ pseg_noL5_pos ].size()) == n_layers_occupied_tot -1 ) {

		  curv_noL5_A[ pseg_noL5_pos ] += theWeight(
							    (*(Psegments_noL5[ pseg_noL5_pos ].end()-1 ))->localPosition().x(), 
							    (*(Psegments_noL5[ pseg_noL5_pos ].end()-midlayer_pointer[n_layers_occupied_tot-2] ))->localPosition().x(),
							    (*(Psegments_noL5[ pseg_noL5_pos ].end()-(n_layers_occupied_tot-1) ))->localPosition().x(),
							    float((*(Psegments_noL5[ pseg_noL5_pos ].end()-1 ))->cscDetId().layer()),
							    float((*(Psegments_noL5[ pseg_noL5_pos ].end()-midlayer_pointer[n_layers_occupied_tot-2] ))->cscDetId().layer()),
							    float((*(Psegments_noL5[ pseg_noL5_pos ].end()-(n_layers_occupied_tot-1) ))->cscDetId().layer())
							    );

		  if (curv_noL5_A[ pseg_noL5_pos ] > curvePenaltyThreshold) weight_noL5_A[ pseg_noL5_pos ] = weight_noL5_A[ pseg_noL5_pos ] * curvePenalty;

		  if (weight_noL5_B[ pseg_noL5_pos ] > a_yweightPenaltyThreshold[thestation][thering]) 
		    weight_noL5_A[ pseg_noL5_pos ] = weight_noL5_A[ pseg_noL5_pos ] * yweightPenalty;

		  if (weight_noL5_A[ pseg_noL5_pos ] < min_weight_noLx_A ) {
		    min_weight_noLx_A = weight_noL5_A[ pseg_noL5_pos ];
		    //best_weight_noLx_B = weight_noL5_B[ pseg_noL5_pos ];
		    //best_curv_noLx_A = curv_noL5_A[ pseg_noL5_pos ];
		    best_noLx_pseg = pseg_noL5_pos;
                    best_Layer_noLx = 5;
		  }

		}

		// alternative: fill map with weight and pseg (which is already ordered)? Seems a very good tool to go looking for segments from. 
		// As I understand, the segments would be inserted according to their weight, so the list would "automatically" be sorted.

	      }
	    }
	  }

          if ( n_layers_occupied_tot > 5 ) {
	    if (pseg < orig_number_of_noL6_psegs && ( n_layers_processed != 6 )) {
	      if(( Psegments_noL6[ pseg_noL6_pos ].size() > 2 )){
              
		// looks more exciting than it is. Here the weight is calculated. It is the difference in x of the last two and one but the last two hits, 
		// divided by the distance of the corresponding hits. Please refer to twiki page XXXX or CMS Note YYY (and use layer_distance)

		weight_noL6_A[ pseg_noL6_pos ] += theWeight(
							    (*(Psegments_noL6[ pseg_noL6_pos ].end()-1 ))->localPosition().x(), 
							    (*(Psegments_noL6[ pseg_noL6_pos ].end()-2 ))->localPosition().x(),
							    (*(Psegments_noL6[ pseg_noL6_pos ].end()-3 ))->localPosition().x(),
							    float((*(Psegments_noL6[ pseg_noL6_pos ].end()-1))->cscDetId().layer()),
							    float((*(Psegments_noL6[ pseg_noL6_pos ].end()-2))->cscDetId().layer()),
							    float((*(Psegments_noL6[ pseg_noL6_pos ].end()-3))->cscDetId().layer())
							    );

		weight_noL6_B[ pseg_noL6_pos ] += theWeight(
							    (*(Psegments_noL6[ pseg_noL6_pos ].end()-1 ))->localPosition().y(), 
							    (*(Psegments_noL6[ pseg_noL6_pos ].end()-2 ))->localPosition().y(),
							    (*(Psegments_noL6[ pseg_noL6_pos ].end()-3 ))->localPosition().y(),
							    float((*(Psegments_noL6[ pseg_noL6_pos ].end()-1))->cscDetId().layer()),
							    float((*(Psegments_noL6[ pseg_noL6_pos ].end()-2))->cscDetId().layer()),
							    float((*(Psegments_noL6[ pseg_noL6_pos ].end()-3))->cscDetId().layer())
							    );

		//if we have picked up the last hit go looking for pseg with the lowest (and second lowest?) weight

		if(int(Psegments_noL6[ pseg_noL6_pos ].size()) == n_layers_occupied_tot -1 ) {

		  curv_noL6_A[ pseg_noL6_pos ] += theWeight(
							    (*(Psegments_noL6[ pseg_noL6_pos ].end()-1 ))->localPosition().x(), 
							    (*(Psegments_noL6[ pseg_noL6_pos ].end()-midlayer_pointer[n_layers_occupied_tot-2] ))->localPosition().x(),
							    (*(Psegments_noL6[ pseg_noL6_pos ].end()-(n_layers_occupied_tot-1) ))->localPosition().x(),
							    float((*(Psegments_noL6[ pseg_noL6_pos ].end()-1 ))->cscDetId().layer()),
							    float((*(Psegments_noL6[ pseg_noL6_pos ].end()-midlayer_pointer[n_layers_occupied_tot-2] ))->cscDetId().layer()),
							    float((*(Psegments_noL6[ pseg_noL6_pos ].end()-(n_layers_occupied_tot-1) ))->cscDetId().layer())
							    );

		  if (curv_noL6_A[ pseg_noL6_pos ] > curvePenaltyThreshold) weight_noL6_A[ pseg_noL6_pos ] = weight_noL6_A[ pseg_noL6_pos ] * curvePenalty;

		  if (weight_noL6_B[ pseg_noL6_pos ] > a_yweightPenaltyThreshold[thestation][thering]) 
		    weight_noL6_A[ pseg_noL6_pos ] = weight_noL6_A[ pseg_noL6_pos ] * yweightPenalty;

		  if (weight_noL6_A[ pseg_noL6_pos ] < min_weight_noLx_A ) {
		    min_weight_noLx_A = weight_noL6_A[ pseg_noL6_pos ];
		    //best_weight_noLx_B = weight_noL6_B[ pseg_noL6_pos ];
		    //best_curv_noLx_A = curv_noL6_A[ pseg_noL6_pos ];
		    best_noLx_pseg = pseg_noL6_pos;
                    best_Layer_noLx = 6;
		  }

		}

		// alternative: fill map with weight and pseg (which is already ordered)? Seems a very good tool to go looking for segments from. 
		// As I understand, the segments would be inserted according to their weight, so the list would "automatically" be sorted.

	      }
	    }
	  }

	}
      }
    }
  }

  //************************************************************************;    
  //***   End segment building   *******************************************;    
  //************************************************************************;    

  // Important part! Here segment(s) are actually chosen. All the good segments
  // could be chosen or some (best) ones only (in order to save time).

  // Check if there is a segment with n-1 hits that has a signifcantly better 
  // weight than the best n hit segment

  // IBL 070828: implicit assumption here is that there will always only be one segment per 
  // cluster - if there are >1 we will need to find out which segment the alternative n-1 hit 
  // protosegment belongs to!


  //float chosen_weight = min_weight_A;
  //float chosen_ywgt = best_weight_B;
  //float chosen_curv = best_curv_A;
  //int chosen_nlayers = n_layers_occupied_tot;
  int chosen_pseg = best_pseg;
  if (best_pseg<0) { 
    return segmentInChamber; 
  }
  chosen_Psegments = (Psegments);
  chosen_weight_A = (weight_A);

  float hit_drop_limit = -999999.999;

  // define different weight improvement requirements depending on how many layers are in the segment candidate
  switch ( n_layers_processed ) {
  case 1 : 
    // do nothing;
    break;
  case 2 :
    // do nothing;
    break;
  case 3 : 
    // do nothing;
    break;
  case 4 : 
    hit_drop_limit =  hitDropLimit6Hits * (1./2.) * hitDropLimit4Hits;
    if ((best_Layer_noLx < 1) || (best_Layer_noLx > 4)) {
      //      std::cout<<"CSCSegAlgoST: For four layers, best_Layer_noLx = "<< best_Layer_noLx << std::endl;
    }
    if ((best_Layer_noLx == 2) || (best_Layer_noLx == 3)) hit_drop_limit = hit_drop_limit * (1./2.); 
    break;
  case 5 : 
    hit_drop_limit =  hitDropLimit6Hits * (2./3.) * hitDropLimit5Hits;
    if ((best_Layer_noLx < 1) || (best_Layer_noLx > 5)) {
      //      std::cout<<"CSCSegAlgoST: For five layers, best_Layer_noLx = "<< best_Layer_noLx << std::endl;
    }
    if ((best_Layer_noLx == 2) || (best_Layer_noLx == 4)) hit_drop_limit = hit_drop_limit * (1./2.); 
    if (best_Layer_noLx == 3) hit_drop_limit = hit_drop_limit * (1./3.); 
    break;
  case 6 : 
    hit_drop_limit =  hitDropLimit6Hits * (3./4.);
    if ((best_Layer_noLx < 1) || (best_Layer_noLx > 6)) {
      //      std::cout<<"CSCSegAlgoST: For six layers, best_Layer_noLx = "<< best_Layer_noLx << std::endl;
    }
    if ((best_Layer_noLx == 2) || (best_Layer_noLx == 5)) hit_drop_limit = hit_drop_limit * (1./2.); 
    if ((best_Layer_noLx == 3) || (best_Layer_noLx == 4)) hit_drop_limit = hit_drop_limit * (1./3.); 
    break;
    
  default : 
    // Fallback - should never occur.
    LogTrace("CSCSegment|CSC") << "[CSCSegAlgoST::buildSegments] Unexpected number of layers with hits - please inform CSC DPG.";
    hit_drop_limit = 0.1;
  }

  // choose the NoLx collection (the one that contains the best N-1 candidate)
  switch ( best_Layer_noLx ) {
  case 1 : 
    Psegments_noLx.clear();
    Psegments_noLx = Psegments_noL1;
    weight_noLx_A.clear();
    weight_noLx_A = weight_noL1_A;
    break;
  case 2 :
    Psegments_noLx.clear();
    Psegments_noLx = Psegments_noL2;
    weight_noLx_A.clear();
    weight_noLx_A = weight_noL2_A;
    break;
  case 3 : 
    Psegments_noLx.clear();
    Psegments_noLx = Psegments_noL3;
    weight_noLx_A.clear();
    weight_noLx_A = weight_noL3_A;
    break;
  case 4 : 
    Psegments_noLx.clear();
    Psegments_noLx = Psegments_noL4;
    weight_noLx_A.clear();
    weight_noLx_A = weight_noL4_A;
    break;
  case 5 : 
    Psegments_noLx.clear();
    Psegments_noLx = Psegments_noL5;
    weight_noLx_A.clear();
    weight_noLx_A = weight_noL5_A;
    break;
  case 6 : 
    Psegments_noLx.clear();
    Psegments_noLx = Psegments_noL6;
    weight_noLx_A.clear();
    weight_noLx_A = weight_noL6_A;
    break;
    
  default : 
    // Fallback - should occur only for preclusters with only 3 layers with hits.
    Psegments_noLx.clear();
    weight_noLx_A.clear();
  }
  
  if( min_weight_A > 0. ) {
    if ( min_weight_noLx_A/min_weight_A < hit_drop_limit ) {
      //chosen_weight = min_weight_noLx_A;
      //chosen_ywgt = best_weight_noLx_B;
      //chosen_curv = best_curv_noLx_A;
      //chosen_nlayers = n_layers_occupied_tot-1;
      chosen_pseg = best_noLx_pseg;
      chosen_Psegments.clear();
      chosen_weight_A.clear();
      chosen_Psegments = (Psegments_noLx);
      chosen_weight_A = (weight_noLx_A);
    }
  }

  if(onlyBestSegment) {
    ChooseSegments2a( chosen_Psegments, chosen_pseg );
  }
  else {
    ChooseSegments3( chosen_Psegments, chosen_weight_A, chosen_pseg ); 
  }

  for(unsigned int iSegment=0; iSegment<GoodSegments.size();++iSegment){
    protoSegment = GoodSegments[iSegment];

    // Create new fitter object
    CSCCondSegFit* segfit = new CSCCondSegFit( pset(), chamber(), protoSegment );
    condpass1 = false;
    condpass2 = false;
    segfit->setScaleXError( 1.0 );
    segfit->fit(condpass1, condpass2);

    // Attempt to handle numerical instability of the fit.
    // (Any segment with chi2/dof > chi2Norm_3D_ is considered
    // as potentially suffering from numerical instability in fit.)
    if( adjustCovariance() ){
    // Call the fit with prefitting option:
    // First fit a straight line to X-Z coordinates and calculate chi2
    // This is done in CSCCondSegFit::correctTheCovX()
    // Scale up errors in X if this chi2 is too big (default 'too big' is >20);
    // Then refit XY-Z with the scaled-up X errors 
      if(segfit->chi2()/segfit->ndof()>chi2Norm_3D_){
	condpass1 = true;
	segfit->fit(condpass1, condpass2);
      }
      if(segfit->scaleXError() < 1.00005){
        LogTrace("CSCWeirdSegment") << "[CSCSegAlgoST::buildSegments] Segment ErrXX scaled and refit " << std::endl;
	if(segfit->chi2()/segfit->ndof()>chi2Norm_3D_){
     // Call the fit with direct adjustment of condition number;
     // If the attempt to improve fit by scaling up X error fails
     // call the procedure to make the condition number of M compatible with
     // the precision of X and Y measurements;
     // Achieved by decreasing abs value of the Covariance
          LogTrace("CSCWeirdSegment") << "[CSCSegAlgoST::buildSegments] Segment ErrXY changed to match cond. number and refit " << std::endl;
	  condpass2 = true;
	  segfit->fit(condpass1, condpass2);
        }
      }
      // Call the pre-pruning procedure;
      // If the attempt to improve fit by scaling up X error is successfull,
      // while scale factor for X errors is too big.
      // Prune the recHit inducing the biggest contribution into X-Z chi^2
      // and refit;
      if(prePrun_ && (sqrt(segfit->scaleXError())>prePrunLimit_) &&
	 (segfit->nhits()>3)){   
        LogTrace("CSCWeirdSegment") << "[CSCSegAlgoST::buildSegments] Scale factor chi2uCorrection too big, pre-Prune and refit " << std::endl;
	protoSegment.erase(protoSegment.begin() + segfit->worstHit(),
			   protoSegment.begin() + segfit->worstHit()+1 );                 

	// Need to create new fitter object to repeat fit with fewer hits
	// Original code maintained current values of condpass1, condpass2, scaleXError - calc in CorrectTheCovX()
	//@@ DO THE SAME THING HERE, BUT IS THAT CORRECT?! It does make a difference. 
	double tempcorr = segfit->scaleXError(); // save current value
	delete segfit;   
	segfit = new CSCCondSegFit( pset(), chamber(), protoSegment );           
        segfit->setScaleXError( tempcorr ); // reset to previous value (rather than init to 1)
	segfit->fit(condpass1, condpass2);
      }
    }
    
    // calculate covariance matrix 
    //    AlgebraicSymMatrix temp2 = segfit->covarianceMatrix();

    // build an actual CSC segment
    CSCSegment temp(protoSegment, segfit->intercept(), segfit->localdir(), 
                       segfit->covarianceMatrix(), segfit->chi2() );
    delete segfit;

    if( debug ) dumpSegment( temp );
 
    segmentInChamber.push_back(temp); 
  }
  return segmentInChamber;
}

void CSCSegAlgoST::ChooseSegments2a(std::vector< ChamberHitContainer > & chosen_segments, int chosen_seg) {
  // just return best segment
  GoodSegments.clear();
  GoodSegments.push_back( chosen_segments[chosen_seg] );
}

void CSCSegAlgoST::ChooseSegments3(std::vector< ChamberHitContainer > & chosen_segments, std::vector< float > & chosen_weight, int chosen_seg) {

  int SumCommonHits = 0;
  GoodSegments.clear();
  int nr_remaining_candidates;
  unsigned int nr_of_segment_candidates;
  
  nr_remaining_candidates = nr_of_segment_candidates = chosen_segments.size();

  // always select and return best protosegment:  
  GoodSegments.push_back( chosen_segments[ chosen_seg ] );

  float chosen_weight_temp = 999999.;
  int chosen_seg_temp = -1;

  // try to find further segment candidates:
  while( nr_remaining_candidates > 0 ) {

    for(unsigned int iCand=0; iCand < nr_of_segment_candidates; ++iCand) {
      //only compare current best to psegs that have not been marked bad:
      if( chosen_weight[iCand] < 0. ) continue;
      SumCommonHits = 0;

      for( int ihits = 0; ihits < int(chosen_segments[iCand].size()); ++ihits ) { // iCand and iiCand NEED to have same nr of hits! (always have by construction)
	if( chosen_segments[iCand][ihits] == chosen_segments[chosen_seg][ihits]) {
	  ++SumCommonHits;
	}
      }

      //mark a pseg bad:
      if(SumCommonHits>1) { // needs to be a card; should be investigated first
	chosen_weight[iCand] = -1.;
	nr_remaining_candidates -= 1;
      }
      else {
	// save the protosegment with the smallest weight
	if( chosen_weight[ iCand ] < chosen_weight_temp ) {
	  chosen_weight_temp = chosen_weight[ iCand ];
	  chosen_seg_temp = iCand ;
	}
      }
    }

    if( chosen_seg_temp > -1 ) GoodSegments.push_back( chosen_segments[ chosen_seg_temp ] );

    chosen_seg = chosen_seg_temp;
    // re-initialze temporary best parameters
    chosen_weight_temp = 999999;
    chosen_seg_temp = -1;
  }
}

void CSCSegAlgoST::ChooseSegments2(int best_seg) {
  //  std::vector <int> CommonHits(6); // nice  concept :)
  std::vector <unsigned int> BadCandidate;
  int SumCommonHits =0;
  GoodSegments.clear();
  BadCandidate.clear();
  for(unsigned int iCand=0;iCand<Psegments.size();++iCand) {
    // skip here if segment was marked bad
    for(unsigned int iiCand=iCand+1;iiCand<Psegments.size();++iiCand){
      // skip here too if segment was marked bad
      SumCommonHits =0;
      if( Psegments[iCand].size() != Psegments[iiCand].size() ) {
	LogTrace("CSCSegment|CSC") << "[CSCSegAlgoST::ChooseSegments2] ALARM!! Should not happen - please inform CSC DPG";
      }
      else {
	for( int ihits = 0; ihits < int(Psegments[iCand].size()); ++ihits ) { // iCand and iiCand NEED to have same nr of hits! (alsways have by construction)
	  if( Psegments[iCand][ihits] == Psegments[iiCand][ihits]) {
	    ++SumCommonHits;
	  }
	}
      }
      if(SumCommonHits>1) {
	if( weight_A[iCand]>weight_A[iiCand] ) { // use weight_A here
	  BadCandidate.push_back(iCand);
	  // rather mark segment bad by an array which is in sync with protosegments!! e.g. set weight = weight*1000 or have an addidional array or set it to weight *= -1
	}
	else{
	  BadCandidate.push_back(iiCand);
	  // rather mark segment bad by an array which is in sync with protosegments!! e.g. set weight = weight*1000 or have an addidional array or set it to weight *= -1
	}
      }
    }
  }
  bool discard;
  for(unsigned int isegm=0;isegm<Psegments.size();++isegm) {
    // For best results another iteration/comparison over Psegments 
    //should be applied here... It would make the program much slower.
    discard = false;
    for(unsigned int ibad=0;ibad<BadCandidate.size();++ibad) {
      // can save this loop if we used an array in sync with Psegments!!!!
      if(isegm == BadCandidate[ibad]) {
	discard = true;
      }
    }
    if(!discard) {
      GoodSegments.push_back( Psegments[isegm] );
    }
  }
}


void CSCSegAlgoST::findDuplicates(std::vector<CSCSegment>  & segments ){
  // this is intended for ME1/1a only - we have ghost segments because of the strips ganging 
  // this function finds them (first the rechits by sharesInput() )
  // if a segment shares all the rechits with another segment it is a duplicate (even if
  // it has less rechits) 
  
  for(std::vector<CSCSegment>::iterator it=segments.begin(); it != segments.end(); ++it) {
    std::vector<CSCSegment*> duplicateSegments;
    for(std::vector<CSCSegment>::iterator it2=segments.begin(); it2 != segments.end(); ++it2) {
      //
      bool allShared = true;
      if(it!=it2){
	allShared = it->sharesRecHits(*it2);
      }
      else{
        allShared = false;
      }
      //
      if(allShared){
        duplicateSegments.push_back(&(*it2));
      }
    }
    it->setDuplicateSegments(duplicateSegments);
  }

}

void CSCSegAlgoST::dumpSegment( const CSCSegment& seg ) const {

  // Only called if pset value 'CSCDebug' is set in config

  edm::LogVerbatim("CSCSegment") << "CSCSegment in " << chamber()->id()
                         << "\nlocal position = " << seg.localPosition()
			 << "\nerror = " << seg.localPositionError() 
			 << "\nlocal direction = " << seg.localDirection()
			 << "\nerror =" << seg.localDirectionError()
			 << "\ncovariance matrix" 
			 << seg.parametersError()
			 << "chi2/ndf = " << seg.chi2() << "/" << seg.degreesOfFreedom()
                         << "\n#rechits = " << seg.specificRecHits().size()
                         << "\ntime = " << seg.time();
}

