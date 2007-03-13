/**
 * \file CSCSegAlgoST.cc
 *
 *  \authors: S. Stoynev - NU
 *            I. Bloch    - FNAL
 *            E. James    - FNAL
 */
 
#include "CSCSegAlgoST.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

// // For clhep Matrix::solve
// #include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"

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
CSCSegAlgoST::CSCSegAlgoST(const edm::ParameterSet& ps) : CSCSegmentAlgorithm(ps), myName("CSCSegAlgoST") {
	
  debug                  = ps.getUntrackedParameter<bool>("CSCDebug");
  //  minLayersApart         = ps.getUntrackedParameter<int>("minLayersApart");
  //  nSigmaFromSegment      = ps.getUntrackedParameter<double>("nSigmaFromSegment");
  minHitsPerSegment      = ps.getUntrackedParameter<int>("minHitsPerSegment");
  //  muonsPerChamberMax     = ps.getUntrackedParameter<int>("CSCSegmentPerChamberMax");      
  //  chi2Max                = ps.getUntrackedParameter<double>("chi2Max");
  dXclusBoxMax           = ps.getUntrackedParameter<double>("dXclusBoxMax");
  dYclusBoxMax           = ps.getUntrackedParameter<double>("dYclusBoxMax");
  preClustering          = ps.getUntrackedParameter<bool>("preClustering");
  Pruning                = ps.getUntrackedParameter<bool>("Pruning");
  BrutePruning           = ps.getUntrackedParameter<bool>("BrutePruning");
  // maxRecHitsInCluster is the maximal number of hits in a precluster that is being processed
  // This cut is intended to remove messy events. Currently nothing is returned if there are
  // more that maxRecHitsInCluster hits. It could be useful to return an estimate of the 
  // cluster position, which is available.
  maxRecHitsInCluster    = ps.getUntrackedParameter<int>("maxRecHitsInCluster");
  onlyBestSegment        = ps.getUntrackedParameter<bool>("onlyBestSegment");
  
  //std::cout<<"Constructor called..."<<std::endl;
}


std::vector<CSCSegment> CSCSegAlgoST::run(const CSCChamber* aChamber, ChamberHitContainer rechits) {

  // Store chamber in temp memory
  theChamber = aChamber; 
  // pre-cluster rechits and loop over all sub clusters seperately
  std::vector<CSCSegment>          segments_temp;
  std::vector<CSCSegment>          segments;
  std::vector<ChamberHitContainer> rechits_clusters; // this is a collection of groups of rechits
  if(preClustering) {
    // run a pre-clusterer on the given rechits to split obviously separated segment seeds:
    rechits_clusters = clusterHits( theChamber, rechits );
    // loop over the found clusters:
    for(std::vector<ChamberHitContainer>::iterator sub_rechits = rechits_clusters.begin(); sub_rechits !=  rechits_clusters.end(); ++sub_rechits ) {
      // clear the buffer for the subset of segments:
      segments_temp.clear();
      // build the subset of segments:
      segments_temp = buildSegments( (*sub_rechits) );
      // add the found subset of segments to the collection of all segments in this chamber:
      segments.insert( segments.end(), segments_temp.begin(), segments_temp.end() );
    }
    // this is the place to prune:
    if( Pruning ) {
      segments_temp.clear(); // segments_temp needed?!?!
      segments_temp = prune_bad_hits( theChamber, segments );
      segments.clear(); // segments_temp needed?!?!
      segments = segments_temp; // segments_temp needed?!?!
    }
    return segments;
  }
  else {
    segments = buildSegments(rechits);
    if( Pruning ) {
      segments_temp.clear(); // segments_temp needed?!?!
      segments_temp = prune_bad_hits( theChamber, segments );
      segments.clear(); // segments_temp needed?!?!
      segments = segments_temp; // segments_temp needed?!?!
    }
    return segments;
    //return buildSegments(rechits); 
  }
}

// ********************************************************************;
// *** This method is meant to remove clear bad hits, using as      ***; 
// *** much information from the chamber as possible (e.g. charge,  ***;
// *** hit position, timing, etc.)                                  ***;
// ********************************************************************;
std::vector<CSCSegment> CSCSegAlgoST::prune_bad_hits(const CSCChamber* aChamber, std::vector<CSCSegment> segments) {
  
//   std::cout<<"*************************************************************"<<std::endl;
//   std::cout<<"Called prune_bad_hits in Chamber "<< theChamber->specs()->chamberTypeName()<<std::endl;
//   std::cout<<"*************************************************************"<<std::endl;
  
  std::vector<CSCSegment>          segments_temp;
  std::vector<ChamberHitContainer> rechits_clusters; // this is a collection of groups of rechits
  
  const float chi2ndfProbMin = 1.0e-4;
  bool   use_brute_force = BrutePruning;

  int hit_nr = 0;
  int hit_nr_worst = -1;
  int hit_nr_2ndworst = -1;
  
  for(std::vector<CSCSegment>::iterator it=segments.begin(); it != segments.end(); it++) {
    
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
	float xdist_local       = -99999.;

	float xdist_local_worst_sig = -99999.;
	float xdist_local_2ndworst_sig = -99999.;
	float xdist_local_sig       = -99999.;

	hit_nr = 0;
	hit_nr_worst = -1;
	hit_nr_2ndworst = -1;

	for ( std::vector<CSCRecHit2D>::const_iterator iRH = theseRecHits.begin(); iRH != theseRecHits.end(); iRH++) {
	  //mark "worst" hit:
	  
 	  float z_at_target ;
	  float radius      ;
	  float loc_x_at_target ;
	  float loc_y_at_target ;
	  float loc_z_at_target ;

	  z_at_target  = 0.;
	  loc_x_at_target  = 0.;
	  loc_y_at_target  = 0.;
	  loc_z_at_target  = 0.;
	  radius       = 0.;
	  
	  // set the z target in CMS global coordinates:
	  const CSCLayer* csclayerRH = theChamber->layer((*iRH).cscDetId().layer());
	  LocalPoint localPositionRH = (*iRH).localPosition();
	  GlobalPoint globalPositionRH = csclayerRH->toGlobal(localPositionRH);	
	  
	  LocalError rerrlocal = (*iRH).localPositionError();  
	  float xxerr = rerrlocal.xx();
	  
	  float target_z     = globalPositionRH.z();  // target z position in cm (z pos of the hit)
	  
	  
	  loc_x_at_target = localPos.x() + (segDir.x()*( target_z - globZ ));
	  loc_y_at_target = localPos.y() + (segDir.y()*( target_z - globZ ));
	  loc_z_at_target = target_z;

	  // have to transform the segments coordinates back to the local frame... how?!!!!!!!!!!!!
	  
	  xdist_local  = fabs(localPositionRH.x() - loc_x_at_target);
	  xdist_local_sig  = fabs((localPositionRH.x() -loc_x_at_target)/(xxerr));
	  
	  if( xdist_local_sig > xdist_local_worst_sig ) {
	    xdist_local_2ndworst_sig = xdist_local_worst_sig;
	    xdist_local_worst_sig    = xdist_local_sig;
	    iRH_worst            = iRH;
	    hit_nr_2ndworst = hit_nr_worst;
	    hit_nr_worst = hit_nr;
	  }
	  else if(xdist_local_sig > xdist_local_2ndworst_sig) {
	    xdist_local_2ndworst_sig = xdist_local_sig;
	    hit_nr_2ndworst = hit_nr;
	  }
	  ++hit_nr;
	}

	// reset worst hit number if certain criteria apply.
	// Criteria: 2nd worst hit must be at least a factor of
	// 1.5 better than the worst in terms of sigma:
	if ( xdist_local_worst_sig / xdist_local_2ndworst_sig < 1.5 ) {
	  hit_nr_worst    = -1;
	  hit_nr_2ndworst = -1;
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
	  for(uint bi = 0; bi < buffer.size(); bi++) {
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
      for(uint iSegment=0; iSegment<reduced_segments.size(); iSegment++) {
	// loop over hits on given segment and push pointers to hits into protosegment
	protoSegment.clear();
	for(uint m = 0; m<reduced_segments[iSegment].size(); ++m ) {
	  protoSegment.push_back(&reduced_segments[iSegment][m]);
	}
 	fitSlopes(); 
 	fillChiSquared();
 	fillLocalDirection();
 	// calculate error matrix
 	AlgebraicSymMatrix protoErrors = calculateError();   
 	// but reorder components to match what's required by TrackingRecHit interface 
 	// i.e. slopes first, then positions 
 	flipErrors( protoErrors ); 
 	//
 	CSCSegment temp(protoSegment, protoIntercept, protoDirection, protoErrors, protoChi2);

	// replace n hit segment with n-1 hit segment, if segment probability is 1e3 better:
 	if( ( ChiSquaredProbability((double)(*it).chi2(),(double)((2*(*it).nRecHits())-4)) 
	      < 
 	      (1.e-3)*(ChiSquaredProbability((double)temp.chi2(),(double)(2*temp.nRecHits()-4))) )
   	    && 
   	    ( (ChiSquaredProbability((double)temp.chi2(),(double)(2*temp.nRecHits()-4))) 
   	      > best_red_seg_prob 
 	      )
	    &&
	    ( (ChiSquaredProbability((double)temp.chi2(),(double)(2*temp.nRecHits()-4))) > 1e-10 )
 	    ) {
	  best_red_seg_prob = ChiSquaredProbability((double)temp.chi2(),(double)(2*temp.nRecHits()-4));
	  // exchange current n hit segment (*it) with better n-1 hit segment:
	  (*it) = temp;
	}
      }
  }
  
  return segments;
  
}
// ********************************************************************;


std::vector< std::vector<const CSCRecHit2D*> > CSCSegAlgoST::clusterHits(const CSCChamber* aChamber, ChamberHitContainer rechits) {
  theChamber = aChamber; 

  std::vector<ChamberHitContainer> rechits_clusters; // this is a collection of groups of rechits
//   const float dXclus_box_cut       = 4.; // seems to work reasonably 070116
//   const float dYclus_box_cut       = 8.; // seems to work reasonably 070116

  float dXclus = 0.0;
  float dYclus = 0.0;
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
      for(uint NNN = 0; NNN < seeds.size(); ++NNN) {
	
	for(uint MMM = NNN+1; MMM < seeds.size(); ++MMM) {
	  if(running_meanX[MMM] == 999999. || running_meanX[NNN] == 999999. ) {
	    std::cout<<"We should never see this line now!!!"<<std::endl;
	    continue; //skip seeds that have been used 
	  }
	  
	  // calculate cut criteria for simple running mean distance cut:
	  dXclus = fabs(running_meanX[NNN] - running_meanX[MMM]);
	  dYclus = fabs(running_meanY[NNN] - running_meanY[MMM]);

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
      for(uint NNN = 0; NNN < seeds.size(); ++NNN) {
	if(running_meanX[NNN] == 999999.) continue; //skip seeds that have been marked as used up in merging
	rechits_clusters.push_back(seeds[NNN]);
      }

      //***************************************************************

  return rechits_clusters; 
}



/* 
 * This algorithm is based on the Minimum Spanning Tree (ST) approach 
 * for building endcap muon track segments out of the rechit's in a CSCChamber.
 */
std::vector<CSCSegment> CSCSegAlgoST::buildSegments(ChamberHitContainer rechits) {

  // Clear buffer for segment vector
  std::vector<CSCSegment> segmentInChamber;
  segmentInChamber.clear(); // list of final segments

  // Stoyans cutoff limit at 20 hits
  unsigned int UpperLimit = maxRecHitsInCluster; 
  if (int(rechits.size()) < minHitsPerSegment){ 
    return segmentInChamber;
  }
  else if(rechits.size()>UpperLimit){ // avoid too messy events; alternative?
    std::cout<<"Number of rechits in the cluster/chamber > "<< UpperLimit<<
      " ... Segment finding in the cluster/chamber canceled! "<<std::endl;
    return segmentInChamber;  
  }

  for(int iarray = 0; iarray <6; ++iarray) { // magic number 6: number of layers in CSC chamber - not gonna change :)
    PAhits_onLayer[iarray].clear();
  }
  weight_A.clear();
  Psegments_hits    .clear();
  Psegments    .clear();

  std::vector<int> hits_onLayerNumber(6);
  //int hits_onLayerNumber[6] = {0,0,0,0,0,0};

  //  int n_layers_missed_tot      = 0;
  int n_layers_occupied_tot    = 0;

  float min_weight_A = 99999.9;
  int best_pseg = -1;

  //************************************************************************;    
  //***   Start segment building   *****************************************;    
  //************************************************************************;    
  
  // Determine how many layers with hits we have
  // Fill all hits into the layer hit container:
  
  // Have 2 standard arrays: one giving the number of hits per layer. 
  // The other the corresponding hits. 
  
  // Loop all available hits, count hits per layer and fill the hits into array by layer
  for(uint M = 0; M < rechits.size(); ++M) {
      // add hits to array per layer and count hits per layer:
      hits_onLayerNumber[ rechits[M]->cscDetId().layer()-1 ] += 1;
      if(hits_onLayerNumber[ rechits[M]->cscDetId().layer()-1 ] == 1 ) n_layers_occupied_tot += 1;
      // add hits to vector in array
      PAhits_onLayer[rechits[M]->cscDetId().layer()-1]    .push_back(rechits[M]);	   
    } 
    // we have now counted the hits per layer and have filled pointers to the hits into an array
    
    // Cut-off parameter - don't reconstruct segments with less than X hits
    if( n_layers_occupied_tot < minHitsPerSegment ) { 
      return segmentInChamber;
    }

    // Start building all possible hit combinations:

    // loop over the layers and form segment candidates from the available hits:
    for(int layer = 0; layer < 6; ++layer) { // loop 6 layers

//       //***************************************************;
//       //*** Set missed layer counter here: ****************;
//       //***************************************************;
//       // increment a counter for the number of layers without hits (missed), if there is no hit on the current layer
//       if( PAhits_onLayer[layer].size() == 0 ) {
//         n_layers_missed_tot += 1;
//       }

      // Save the size of the protosegment before hits were added on the current layer
      int orig_number_of_psegs = Psegments.size();

      // loop over the hits on the layer and initiate protosegments or add hits to protosegments
      for(int hit = 0; hit < int(PAhits_onLayer[layer].size()); ++hit) { // loop all hits on the Layer number "layer"

	// create protosegments from all hits on the first layer with hits
	if( orig_number_of_psegs == 0 ) { // would be faster to turn this around - ask for "orig_number_of_psegs != 0"
	  Psegments_hits    .push_back(PAhits_onLayer[layer][hit]);

	  Psegments    .push_back(Psegments_hits    ); 

          // Initialize weight corresponding to this segment for first hit (with 0)
	  weight_A.push_back(0.0);
          // reset array for next hit on next layer
          Psegments_hits    .clear();
	}
	else {
	  // loop over the protosegments and create a new protosegments for each hit-1 on this layer
	  for( int pseg = 0; pseg < orig_number_of_psegs; ++pseg ) { 

	    int pseg_pos = (pseg)+((hit)*orig_number_of_psegs);

            // - Loop all psegs. 
            // - If not last hit, clone  existing protosegments  (PAhits_onLayer[layer].size()-1) times
            // - then add the new hits
	    if( ! (hit == int(PAhits_onLayer[layer].size()-1)) ) { // not the last hit - prepare (copy) new protosegments for the following hits
              // clone psegs (to add next hits or last hit on layer):
  	      Psegments    .push_back( Psegments[ pseg_pos ]     ); 
              // clone weight corresponding to this segment too
              weight_A.push_back(weight_A[ pseg_pos ]);
	    }
            // add hits to original pseg:
            Psegments[ pseg_pos ].push_back(PAhits_onLayer[ layer ][ hit ]);
            
            // calculate/update the weight (only for >2 hits on psegment):
            if( Psegments[ pseg_pos ].size() > 2 ) {
              
	      // looks more exciting than it is. Here the weight is calculated. It is the difference in x of the last two and one but the last two hits, divided by the
	      // distance of the corresponding hits. Please refer to twiki page XXXX or CMS Note YYY
	      // and use layer_distance

              weight_A[ pseg_pos ] += fabs( 
					   (( (*(Psegments[ pseg_pos ].end()-2))->localPosition().x() 
					      - 
					      (*(Psegments[ pseg_pos ].end()-3))->localPosition().x() ) 
					    / 
					    float( (*(Psegments[ pseg_pos ].end()-2))->cscDetId().layer() 
						   - 
						   (*(Psegments[ pseg_pos ].end()-3))->cscDetId().layer() ))  
					   - 
					   (( (*(Psegments[ pseg_pos ].end()-1 ))->localPosition().x() 
					      - 
					      (*(Psegments[ pseg_pos ].end()-2 ))->localPosition().x() ) 
					    / 
					    float( (*(Psegments[ pseg_pos ].end()-1  ))->cscDetId().layer() 
						   - 
						   (*(Psegments[ pseg_pos ].end()-2))->cscDetId().layer() )) 
					   ); // access hit using .end()-N
              
              
              //if we have picked up the last hit go looking for pseg with the lowest (and second lowest?) weight
              if(int(Psegments[ pseg_pos ].size()) == n_layers_occupied_tot && weight_A[ pseg_pos ] < min_weight_A ) {
		min_weight_A = weight_A[ pseg_pos ];
		best_pseg = pseg_pos ;
              }
              // alternative: fill map with weight and pseg (which is already ordered)? Seems a very good tool to go looking for segments from. As I understand, the segments 
              // would be inserted according to their weight, so the list would "automatically" be sorted.

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
    
    if(onlyBestSegment){
      ChooseSegments2a( best_pseg );
    }
    else{
      //      ChooseSegments2( best_pseg );
      ChooseSegments3( best_pseg ); 
    }

    for(unsigned int iSegment=0; iSegment<GoodSegments.size();iSegment++){
      protoSegment = GoodSegments[iSegment];
      fitSlopes(); 
      fillChiSquared();
      fillLocalDirection();
      // calculate error matrix
      AlgebraicSymMatrix protoErrors = calculateError();   
      // but reorder components to match what's required by TrackingRecHit interface 
      // i.e. slopes first, then positions 
      flipErrors( protoErrors ); 
      //
      CSCSegment temp(protoSegment, protoIntercept, protoDirection, protoErrors, protoChi2);
      segmentInChamber.push_back(temp); 
    }
    return segmentInChamber;
}

void CSCSegAlgoST::ChooseSegments2a(int best_seg){
  // just return best segment
  GoodSegments.clear();
  GoodSegments.push_back( Psegments[best_seg] );
}

void CSCSegAlgoST::ChooseSegments3(int best_seg) {

  int SumCommonHits = 0;
  GoodSegments.clear();
  int nr_remaining_candidates;
  uint nr_of_segment_candidates;
  
  nr_remaining_candidates = nr_of_segment_candidates = Psegments.size();

  // always select and return best protosegment:  
  GoodSegments.push_back( Psegments[ best_seg ] );
  float min_weight_A_temp = 999999.;
  int best_seg_temp = -1;

  // try to find further segment candidates:
  while( nr_remaining_candidates > 0 ) {

    for(unsigned int iCand=0; iCand < nr_of_segment_candidates; ++iCand) {
      //only compare current best to psegs that have not been marked bad:
      if( weight_A[iCand] < 0. ) continue;
      SumCommonHits = 0;

      for( int ihits = 0; ihits < int(Psegments[iCand].size()); ++ihits ) { // iCand and iiCand NEED to have same nr of hits! (alsways have by construction)
	if( Psegments[iCand][ihits] == Psegments[best_seg][ihits]) {
	  SumCommonHits++;
	}
      }

      //mark a pseg bad:
      if(SumCommonHits>1) { // needs to be a card; should be investigated first
	weight_A[iCand] *= -1.;
	nr_remaining_candidates -= 1;
      }
      else {
	// save the protosegment with the smallest weight
	if( weight_A[ iCand ] < min_weight_A_temp ) {
	  min_weight_A_temp = weight_A[ iCand ];
	  best_seg_temp = iCand ;
	}
      }
    }

    if( best_seg_temp > -1 ) GoodSegments.push_back( Psegments[ best_seg_temp ] );
    best_seg = best_seg_temp;
    // re-initialze temporary best parameters
    min_weight_A_temp = 999999;
    best_seg_temp = -1;
  }
}

void CSCSegAlgoST::ChooseSegments2(int best_seg) {
  //  std::vector <int> CommonHits(6); // nice  concept :)
  std::vector <unsigned int> BadCandidate;
  int SumCommonHits =0;
  GoodSegments.clear();
  BadCandidate.clear();
  for(unsigned int iCand=0;iCand<Psegments.size();iCand++) {
    // skip here if segment was marked bad
    for(unsigned int iiCand=iCand+1;iiCand<Psegments.size();iiCand++){
    // skip here too if segment was marked bad
      SumCommonHits =0;
      if( Psegments[iCand].size() != Psegments[iiCand].size() ) {
	std::cout<<"ALARM!! THIS should not happen!!"<<std::endl;
      }
      else {
	for( int ihits = 0; ihits < int(Psegments[iCand].size()); ++ihits ) { // iCand and iiCand NEED to have same nr of hits! (alsways have by construction)
	  if( Psegments[iCand][ihits] == Psegments[iiCand][ihits]) {
	    SumCommonHits++;
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
  for(unsigned int isegm=0;isegm<Psegments.size();isegm++) {
    // For best results another iteration/comparison over Psegments 
    //should be applied here... It would make the program much slower.
    discard = false;
    for(unsigned int ibad=0;ibad<BadCandidate.size();ibad++) {
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

/* Method fitSlopes
 *
 * Perform a Least Square Fit on a segment as per SK algo
 *
 */
void CSCSegAlgoST::fitSlopes() {
  HepMatrix M(4,4,0);
  HepVector B(4,0);
  ChamberHitContainer::const_iterator ih = protoSegment.begin();
  for (ih = protoSegment.begin(); ih != protoSegment.end(); ++ih) {
    const CSCRecHit2D& hit = (**ih);
    const CSCLayer* layer  = theChamber->layer(hit.cscDetId().layer());
    GlobalPoint gp         = layer->toGlobal(hit.localPosition());
    LocalPoint  lp         = theChamber->toLocal(gp); 
    // ptc: Local position of hit w.r.t. chamber
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();
    // ptc: Covariance matrix of local errors 
    HepMatrix IC(2,2);
    IC(1,1) = hit.localPositionError().xx();
    IC(1,2) = hit.localPositionError().xy();
    IC(2,2) = hit.localPositionError().yy();
    IC(2,1) = IC(1,2); // since Cov is symmetric
    // ptc: Invert covariance matrix (and trap if it fails!)
    int ierr = 0;
    IC.invert(ierr); // inverts in place
    if (ierr != 0) {
      LogDebug("CSC") << "CSCSegment::fitSlopes: failed to invert covariance matrix=\n" << IC << "\n";      
//       std::cout<< "CSCSegment::fitSlopes: failed to invert covariance matrix=\n" << IC << "\n"<<std::endl;
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
  HepVector p = solve(M, B);
  
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
void CSCSegAlgoST::fillChiSquared() {
  
  double chsq = 0.;
  
  ChamberHitContainer::const_iterator ih;
  for (ih = protoSegment.begin(); ih != protoSegment.end(); ++ih) {
    
    const CSCRecHit2D& hit = (**ih);
    const CSCLayer* layer  = theChamber->layer(hit.cscDetId().layer());
    GlobalPoint gp         = layer->toGlobal(hit.localPosition());
    LocalPoint lp          = theChamber->toLocal(gp);
    
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();
    
    double du = protoIntercept.x() + protoSlope_u * z - u;
    double dv = protoIntercept.y() + protoSlope_v * z - v;
    
    HepMatrix IC(2,2);
    IC(1,1) = hit.localPositionError().xx();
    IC(1,2) = hit.localPositionError().xy();
    IC(2,2) = hit.localPositionError().yy();
    IC(2,1) = IC(1,2);
    
    // Invert covariance matrix
    int ierr = 0;
    IC.invert(ierr);
    if (ierr != 0) {
      LogDebug("CSC") << "CSCSegment::fillChiSquared: failed to invert covariance matrix=\n" << IC << "\n";
//       std::cout << "CSCSegment::fillChiSquared: failed to invert covariance matrix=\n" << IC << "\n";
      
    }
    
    chsq += du*du*IC(1,1) + 2.*du*dv*IC(1,2) + dv*dv*IC(2,2);
  }

  protoChi2 = chsq;
}
/* fillLocalDirection
 *
 */
void CSCSegAlgoST::fillLocalDirection() {
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
  
  double globalZpos    = ( theChamber->toGlobal( protoIntercept ) ).z();
  double globalZdir    = ( theChamber->toGlobal( localDir ) ).z();
  double directionSign = globalZpos * globalZdir;
  protoDirection       = (directionSign * localDir).unit();
}
/* weightMatrix
 *   
 */
AlgebraicSymMatrix CSCSegAlgoST::weightMatrix() const {
  
  std::vector<const CSCRecHit2D*>::const_iterator it;
  int nhits = protoSegment.size();
  AlgebraicSymMatrix matrix(2*nhits, 0);
  int row = 0;
  
  for (it = protoSegment.begin(); it != protoSegment.end(); ++it) {
    
    const CSCRecHit2D& hit = (**it);
    ++row;
    matrix(row, row)   = hit.localPositionError().xx();
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
HepMatrix CSCSegAlgoST::derivativeMatrix() const {
  
  ChamberHitContainer::const_iterator it;
  int nhits = protoSegment.size();
  HepMatrix matrix(2*nhits, 4);
  int row = 0;
  
  for(it = protoSegment.begin(); it != protoSegment.end(); ++it) {
    
    const CSCRecHit2D& hit = (**it);
    const CSCLayer* layer = theChamber->layer(hit.cscDetId().layer());
    GlobalPoint gp = layer->toGlobal(hit.localPosition());      
    LocalPoint lp = theChamber->toLocal(gp); 
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


/* calculateError
 *
 */
AlgebraicSymMatrix CSCSegAlgoST::calculateError() const {
  
  AlgebraicSymMatrix weights = weightMatrix();
  AlgebraicMatrix A = derivativeMatrix();
  
  // (AT W A)^-1
  // from http://www.phys.ufl.edu/~avery/fitting.html, part I
  int ierr;
  AlgebraicSymMatrix result = weights.similarityT(A);
  result.invert(ierr);
  
  // blithely assuming the inverting never fails...
  return result;
}


void CSCSegAlgoST::flipErrors( AlgebraicSymMatrix& a ) const { 
    
  // The CSCSegment needs the error matrix re-arranged 
    
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
    
  // off-diagonal elements remain unchanged 
    
} 
