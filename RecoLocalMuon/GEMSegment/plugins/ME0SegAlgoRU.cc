/** 
 * \file ME0SegAlgRU.cc 
 *  
 *  \author M. Maggi for ME0
 *  \from V.Palichik & N.Voytishin   
 *  \some functions and structure taken from SK algo by M.Sani and SegFit class by T.Cox 
 */ 

#include "ME0SegAlgoRU.h" 
#include "MuonSegFit.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h" 
#include "DataFormats/GeometryVector/interface/GlobalPoint.h" 
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h" 

#include <algorithm> 
#include <cmath> 
#include <iostream> 
#include <string> 

ME0SegAlgoRU::ME0SegAlgoRU(const edm::ParameterSet& ps)
  : ME0SegmentAlgorithmBase(ps), myName("ME0SegAlgoRU"), sfit_(nullptr) { 

  doCollisions = ps.getParameter<bool>("doCollisions"); 
  chi2_str_   = ps.getParameter<double>("chi2_str"); 
  chi2Norm_2D_   = ps.getParameter<double>("chi2Norm_2D_");    
  dRMax       = ps.getParameter<double>("dRMax"); 
  dPhiMax        = ps.getParameter<double>("dPhiMax"); 
  dRIntMax   = ps.getParameter<double>("dRIntMax"); 
  dPhiIntMax    = ps.getParameter<double>("dPhiIntMax"); 
  chi2Max        = ps.getParameter<double>("chi2Max"); 
  wideSeg        = ps.getParameter<double>("wideSeg"); 
  minLayersApart = ps.getParameter<int>("minLayersApart"); 

  LogDebug("ME0") << myName << " has algorithm cuts set to: \n" 
		  << "--------------------------------------------------------------------\n" 
		  << "dRMax     = " << dRMax << '\n' 
		  << "dPhiMax      = " << dPhiMax << '\n' 
		  << "dRIntMax = " << dRIntMax << '\n' 
		  << "dPhiIntMax  = " << dPhiIntMax << '\n' 
		  << "chi2Max      = " << chi2Max << '\n' 
		  << "wideSeg      = " << wideSeg << '\n' 
		  << "minLayersApart = " << minLayersApart << std::endl; 

  //reset the thresholds for non-collision data 
  if(!doCollisions){ 
    dRMax = 2.0; 
    dPhiMax = 2*dPhiMax; 
    dRIntMax = 2*dRIntMax; 
    dPhiIntMax = 2*dPhiIntMax; 
    chi2Norm_2D_ = 5*chi2Norm_2D_; 
    chi2_str_ = 100; 
    chi2Max = 2*chi2Max; 
  } 
} 

std::vector<ME0Segment> ME0SegAlgoRU::run(const ME0Ensemble& ensemble, const EnsembleHitContainer& rechits){ 
  theEnsemble = ensemble;
  return this->buildSegments(rechits);  
} 

std::vector<ME0Segment> ME0SegAlgoRU::buildSegments(const EnsembleHitContainer& urechits) { 
  EnsembleHitContainer rechits = urechits; 
  LayerIndex layerIndex(rechits.size()); 
  // MM hard coded should work for <=6 layer..
  int recHits_per_layer[6] = {0,0,0,0,0,0}; 

  // MM this should be verified for ME0, noise and PU and high eta are difficult regions...
  //skip events with high multiplicity of hits 
  if (rechits.size()>300){ 
    return std::vector<ME0Segment>(); 
  } 

  int iadd = 0; 
  // determine if smaller layer correspong to smaller abs(z) nearest to IP
  int minlayer = 99;
  int maxlayer = -1;
  float zmin = 0;
  float zmax = 0;
  for(unsigned int i = 0; i < rechits.size(); i++) {  
    int layer = rechits[i]->me0Id().layer();
    recHits_per_layer[layer-1]++;//count rh per chamber 
    layerIndex[i] = layer;
    if (layer > maxlayer ) {
      maxlayer = layer;
      const ME0EtaPartition * thePartition   = (theEnsemble.second).find(rechits[i]->me0Id())->second;
      zmax = (thePartition->toGlobal(rechits[i]->localPosition())).z();
    }
    if (layer < minlayer ) {
      minlayer = layer;
      const ME0EtaPartition * thePartition   = (theEnsemble.second).find(rechits[i]->me0Id())->second;
      zmin = (thePartition->toGlobal(rechits[i]->localPosition())).z();
    }
  }

  if (std::abs(zmin) > std::abs(zmax)){ 
    reverse(layerIndex.begin(), layerIndex.end()); 
    reverse(rechits.begin(), rechits.end()); 
  }     
 
  if (rechits.size() < 2) { 
    return std::vector<ME0Segment>();  
  } 

  // We have at least 2 hits. We intend to try all possible pairs of hits to start  
  // segment building. 'All possible' means each hit lies on different layers in the chamber. 
  // after all same size segs are build we get rid of the overcrossed segments using the chi2 criteria 
  // the hits from the segs that are left are marked as used and are not added to segs in future iterations 
  // the hits from 3p segs are marked as used separately in order to try to assamble them in longer segments  
  // in case there is a second pass    

  // Choose first hit (as close to IP as possible) h1 and second hit 
  // (as far from IP as possible) h2 To do this we iterate over hits 
  // in the chamber by layer - pick two layers.  Then we 
  // iterate over hits within each of these layers and pick h1 and h2 
  // these.  If they are 'close enough' we build an empty 
  // segment.  Then try adding hits to this segment. 

  // Initialize flags that a given hit has been allocated to a segment 
  BoolContainer used(rechits.size(), false); 
  BoolContainer used3p(rechits.size(), false); 

  // This is going to point to fits to hits, and its content will be used to create a ME0Segment
  sfit_ = 0;

  // Define buffer for segments we build  
  std::vector<ME0Segment> segments; 

  EnsembleHitContainerCIt ib = rechits.begin(); 
  EnsembleHitContainerCIt ie = rechits.end(); 

  // Possibly allow 3 passes, second widening scale factor for cuts, third for segments from displaced vertices 
  windowScale = 1.; // scale factor for cuts 

  bool search_disp = false;	   
  int npass = (wideSeg > 1.)? 3 : 2; 

  for (int ipass = 0; ipass < npass; ++ipass) { 
    if(windowScale >1.){ 
      iadd = 1; 
      strip_iadd = 2; 
      chi2D_iadd = 2; 
    } 

    int used_rh = 0;
    for (EnsembleHitContainerCIt i1 = ib; i1 != ie; ++i1) {
      if(used[i1-ib])used_rh++;
    }
    //change the tresholds if it's time to look for displaced mu segments                                                                                                                                                          
    if(doCollisions && search_disp && int(rechits.size()-used_rh)>2){//check if there are enough recHits left to build a segment from displaced vertices                                                                     
      doCollisions = false;
      windowScale = 1.; // scale factor for cuts   
      dRMax = 2.*dRMax;
      dPhiMax = 2*dPhiMax;
      dRIntMax = 2*dRIntMax;
      dPhiIntMax = 2*dPhiIntMax;
      chi2Norm_2D_ = 5*chi2Norm_2D_;
      //      chi2_str_ = 100;
      chi2Max = 2*chi2Max;
    }

    for(unsigned int n_seg_min = 6u; n_seg_min > 2u + iadd; --n_seg_min){ 
      BoolContainer common_used(rechits.size(),false); 
      std::array<BoolContainer, 120> common_used_it = {}; 
      for (unsigned int i = 0; i < common_used_it.size(); i++) { 
	common_used_it[i] = common_used; 
      } 
      EnsembleHitContainer best_proto_segment[120]; 
      float min_chi[120] = {9999}; 
      int common_it = 0; 
      bool first_proto_segment = true; 
      // the first hit is taken from the back
      
      for (EnsembleHitContainerCIt i1 = ib; i1 != ie; ++i1) { 
	  
	//skip if rh is used and the layer tat has big rh multiplicity(>25RHs) 
	if(used[i1-ib] || recHits_per_layer[int(layerIndex[i1-ib])-1]>25 || (n_seg_min == 3 && used3p[i1-ib])) continue; 

	int layer1 = layerIndex[i1-ib];  
	const ME0RecHit* h1 = *i1; 
	// the second hit from the front
	for (EnsembleHitContainerCIt i2 = ie-1; i2 != i1; --i2) { 
	  bool segok = false; 
	  if(used[i2-ib] || recHits_per_layer[int(layerIndex[i2-ib])-1]>25 || (n_seg_min == 3 && used3p[i2-ib])) continue; 

	  int layer2 = layerIndex[i2-ib]; 
	  if((abs(layer2 - layer1) + 1) < int(n_seg_min)) break;//decrease n_seg_min 
          const ME0RecHit* h2 = *i2; 	
	    if (this->areHitsCloseInEta(h1, h2) && this->areHitsCloseInGlobalPhi(h1, h2)) { 
	    proto_segment.clear(); 
	    if (!this->addHit(h1, layer1))continue; 
	    if (!this->addHit(h2, layer2))continue; 

	    // Can only add hits if already have a segment
	    if ( sfit_ ) this->tryAddingHitsToSegment(rechits, used, layerIndex, i1, i2);  
	    segok = this->isSegmentGood(rechits); 
	    if (segok) { 
	      if(proto_segment.size() > n_seg_min){
		this->baseline(n_seg_min); 
		this->updateParameters();
	      }
	      if(sfit_->chi2()/sfit_->ndof() > chi2Norm_2D_*chi2D_iadd || proto_segment.size() < n_seg_min) {
		proto_segment.clear(); 
	      }
	      
	      if (!proto_segment.empty()) { 
		this->updateParameters(); 

	        //add same-size overcrossed protosegments to the collection 
		if(first_proto_segment){ 
		  
		  this->flagHitsAsUsed(rechits, common_used_it[0]); 
		  min_chi[0] = sfit_->chi2()/sfit_->ndof(); 
		  best_proto_segment[0] = proto_segment; 
		  
		  first_proto_segment = false; 
		}else{  //for the rest of found proto_segments  
		  common_it++; 
		  this->flagHitsAsUsed(rechits, common_used_it[common_it]); 

		  min_chi[common_it] = sfit_->chi2()/sfit_->ndof(); 
		  best_proto_segment[common_it] = proto_segment; 
		  EnsembleHitContainerCIt hi, iu, ik; 
		  int iter = common_it; 
		  for(iu = ib; iu != ie; ++iu) { 
		    for(hi = proto_segment.begin(); hi != proto_segment.end(); ++hi) { 
		      if(*hi == *iu) { 
			int merge_nr = -1; 
			for(int k = 0; k < iter+1; k++){ 
			  if(common_used_it[k][iu-ib] == true){ 
			    if(merge_nr != -1){ 

			      //merge the k and merge_nr collections of flaged hits into the merge_nr collection and unmark the k collection hits                                   
			      for(ik = ib; ik != ie; ++ik) { 
				if(common_used_it[k][ik-ib] == true){ 
				  common_used_it[merge_nr][ik-ib] = true; 
				  common_used_it[k][ik-ib] = false; 
				} 
			      } 
			      //change best_protoseg if min_chi_2 is smaller                                                                                                                       
			      if(min_chi[k] < min_chi[merge_nr]){ 
				min_chi[merge_nr] = min_chi[k]; 
				best_proto_segment[merge_nr] = best_proto_segment[k]; 
				best_proto_segment[k].clear(); 
				min_chi[k] = 9999; 
			      } 
			      common_it--; 
			    } 
			    else{ 
			      merge_nr = k; 
			    } 
			  }//if(common_used[k][iu-ib] == true)                                                                                                                                     
			}//for k                                                                                                                                                                    
		      }//if                                                                                                                                                                        
		    }//for proto_seg                                                                                                                                                                 
		  }//for rec_hits   
		}//else 
	      }//proto seg not empty 
	    }
	  }  //   h1 & h2 close 
	  if (segok)  {
	    //            break; 
	  }
	}  //  i2 
      }  //  i1 

      //add the reconstructed segments 
      for(int j = 0;j < common_it+1; j++){ 
	
	proto_segment = best_proto_segment[j]; 
	best_proto_segment[j].clear(); 
	//SKIP empty proto-segments
        if(proto_segment.size() == 0) continue;
	this->updateParameters();   
    
	// Create an actual ME0Segment - retrieve all info from the fit
	// calculate the timing fron rec hits associated to the TrackingRecHits used 
	// to fit the segment 
	float averageTime=0.;
	for(EnsembleHitContainer::iterator ir=proto_segment.begin(); ir<proto_segment.end(); ++ir ) {
	  averageTime += (*ir)->tof();
	}
	if(proto_segment.size() != 0)
	  averageTime=averageTime/(proto_segment.size());
	float timeUncrt=0.;
	for(EnsembleHitContainer::iterator ir=proto_segment.begin(); ir<proto_segment.end(); ++ir ) {
	  timeUncrt += pow( (*ir)->tof()-averageTime,2);
	}
	if(proto_segment.size() > 1)
	  timeUncrt=timeUncrt/(proto_segment.size()-1);
	timeUncrt = sqrt(timeUncrt);
	std::sort(proto_segment.begin(),proto_segment.end(),sortByLayer());
	ME0Segment temp(proto_segment, sfit_->intercept(),
			sfit_->localdir(), sfit_->covarianceMatrix(), sfit_->chi2(), averageTime, timeUncrt);
	sfit_ = 0;
	segments.push_back(temp);                         
	//if the segment has 3 hits flag them as used in a particular way 
	if(proto_segment.size() == 3){ 
	  this->flagHitsAsUsed(rechits, used3p); 
	} 
	else{ 
	  this->flagHitsAsUsed(rechits, used);   
	} 
	proto_segment.clear(); 
      }  
    }//for n_seg_min 

    if(!doCollisions && search_disp){

      //reset params and flags for the next ensemble                                                                                                                           
      search_disp = false;
      doCollisions = true;
      dRMax = dRMax/2.0;
      dPhiMax = dPhiMax/2;
      dRIntMax = dRIntMax/2;
      dPhiIntMax = dPhiIntMax/2;
      chi2Norm_2D_ = chi2Norm_2D_/5;
      chi2_str_ = 100;
      chi2Max = chi2Max/2;
    }     

    std::vector<ME0Segment>::iterator it =segments.begin(); 
    bool good_segs = false; 
    while(it != segments.end()) { 
      if ((*it).nRecHits() > 3){ 
	good_segs = true; 
	break; 
      } 
      ++it;	 
    }     

    if (good_segs) {  // only change window if not enough good segments were found (bool can be changed to int if a >0 number of good segs is required) 
      search_disp = true;
      continue;//proceed to search the segs from displaced vertices
    }

    // Increase cut windows by factor of wideSeg only for collisions 
    if(!doCollisions && !search_disp) break;     
    windowScale = wideSeg; 
  }  //  ipass 
  /* Specific to CSC not used at the moment for ME0
  //get rid of enchansed 3p segments 
  std::vector<ME0Segment>::iterator it =segments.begin(); 
  while(it != segments.end()) { 
    if((*it).nRecHits() == 3){ 
      bool found_common = false; 
      const std::vector<ME0RecHit>& theseRH = (*it).specificRecHits(); 
      for (EnsembleHitContainerCIt i1 = ib; i1 != ie; ++i1) { 
	if(used[i1-ib] && used3p[i1-ib]){ 
	  const ME0RecHit* sh1 = *i1; 

	  ME0DetId id = sh1->me0Id(); 
	  int sh1layer = id.layer(); 
	  int RH_centerid     =  sh1->nStrips()/2; 
	  int RH_centerStrip =  sh1->channels(RH_centerid); 
	  int RH_wg = sh1->hitWire();
	  std::vector<ME0RecHit>::const_iterator sh; 
	  for(sh = theseRH.begin(); sh != theseRH.end(); ++sh){ 
	    ME0DetId idRH = sh->me0Id();

	    //find segment hit coord 
	    int shlayer = idRH.layer(); 
	    int SegRH_centerid     =  sh->nStrips()/2; 
	    int SegRH_centerStrip =  sh->channels(SegRH_centerid); 
	    int SegRH_wg = sh->hitWire(); 

	    if(sh1layer == shlayer && SegRH_centerStrip == RH_centerStrip && SegRH_wg == RH_wg){ 
	      //remove the enchansed 3p segment 
	      segments.erase(it,(it+1)); 
	      found_common = true; 
	      break; 
	    } 
	  }//theserh 
	} 
	if(found_common) break;//current seg has already been erased 
      }//camber hits 
      if(!found_common)++it;  
    }//its a 3p seg 
    else{ 
      ++it;//go to the next seg 
    }  
  }//while 
  */
  // Give the segments to the ME0Ensemble 
  return segments; 
}//build segments 

void ME0SegAlgoRU::tryAddingHitsToSegment(const EnsembleHitContainer& rechits,  
					  const BoolContainer& used, const LayerIndex& layerIndex, 
					  const EnsembleHitContainerCIt i1, const EnsembleHitContainerCIt i2) { 

  // Iterate over the layers with hits in the chamber 
  // Skip the layers containing the segment endpoints 
  // Test each hit on the other layers to see if it is near the segment 
  // If it is, see whether there is already a hit on the segment from the same layer 
  //    - if so, and there are more than 2 hits on the segment, copy the segment, 
  //      replace the old hit with the new hit. If the new segment chi2 is better 
  //      then replace the original segment with the new one (by swap) 
  //    - if not, copy the segment, add the hit. If the new chi2/dof is still satisfactory 
  //      then replace the original segment with the new one (by swap) 

  EnsembleHitContainerCIt ib = rechits.begin(); 
  EnsembleHitContainerCIt ie = rechits.end(); 
  
  for (EnsembleHitContainerCIt i = ib; i != ie; ++i) { 
    if(layerIndex[i1-ib]<layerIndex[i2-ib]){ 
      if (layerIndex[i-ib] <= layerIndex[i1-ib] || layerIndex[i-ib] >= layerIndex[i2-ib] || i  == i1 || i == i2 || used[i-ib]){  
	if ( i  == i1 || i == i2 || used[i-ib]) 
	  continue;  
      } 
    } 
    else{ 
      if (layerIndex[i-ib] >= layerIndex[i1-ib] || layerIndex[i-ib] <= layerIndex[i2-ib] || i  == i1 || i == i2 || used[i-ib]){ 
	if ( i  == i1 || i == i2 || used[i-ib])                                                                                                              
	  continue; 
      }  
    } 
    int layer = layerIndex[i-ib]; 
    const ME0RecHit* h = *i; 
    if (this->isHitNearSegment(h)) {

      // Don't consider alternate hits on layers holding the two starting points 
      if (this->hasHitOnLayer(layer)) { 
	if (proto_segment.size() <= 2) continue; 
	this->compareProtoSegment(h, layer); 
      }  
      else{ 
	this->increaseProtoSegment(h, layer, chi2D_iadd);  
      }
      
    }   // h & seg close 
  }   // i 
} 

bool ME0SegAlgoRU::areHitsCloseInEta(const ME0RecHit* h1, const ME0RecHit* h2) const { 
  // check that hits from different layer gets eta partition number +1 max..

  ME0DetId id1 = h1->me0Id();  
  const ME0EtaPartition* part1 = theEnsemble.second.find(id1)->second;
  GlobalPoint gp1 = part1->toGlobal(h1->localPosition());	 
  int etaP1 = id1.roll();

  ME0DetId id2 = h2->me0Id();  
  const ME0EtaPartition* part2 = theEnsemble.second.find(id2)->second;
  GlobalPoint gp2 = part2->toGlobal(h2->localPosition());	 
  int etaP2 = id2.roll();

  //find z to understand the direction 
  float h1z = gp1.z(); 
  float h2z = gp2.z(); 
  bool good = false;
  float dR = 9999;
  if (doCollisions){
    if ( abs(h1z) > abs(h2z) ) {
      good = (etaP1==etaP2 || etaP1 == etaP2-1);
    }else{
      good = (etaP1==etaP2 || etaP2 == etaP1-1);
    }	 
    dR = fabs(gp1.perp()-gp2.perp());
  }else{
    good = std::abs(etaP1-etaP2) <= 1;
    dR = 0;
  }
  return (good && dR<dRMax);
} 

bool ME0SegAlgoRU::areHitsCloseInGlobalPhi(const ME0RecHit* h1, const ME0RecHit* h2) const { 

  ME0DetId id1 = h1->me0Id();  
  const ME0EtaPartition* part1 = theEnsemble.second.find(id1)->second;
  GlobalPoint gp1 = part1->toGlobal(h1->localPosition());	 

  ME0DetId id2 = h2->me0Id();  
  const ME0EtaPartition* part2 = theEnsemble.second.find(id2)->second;
  GlobalPoint gp2 = part2->toGlobal(h2->localPosition());	 

  float dphi12 = deltaPhi(gp1.barePhi(),gp2.barePhi());
  return fabs(dphi12) < dPhiMax; 
} 

bool ME0SegAlgoRU::isHitNearSegment(const ME0RecHit* h) const { 

  // Is hit near segment?  
  // Requires deltaphi and deltaR within ranges specified in parameter set. 
  // Note that to make intuitive cuts on delta(phi) one must work in 
  // phi range (-pi, +pi] not [0, 2pi) 

  ME0DetId besId = (*(proto_segment.begin()))->me0Id();
  const ME0EtaPartition* l1 = theEnsemble.second.find(besId)->second;
  GlobalPoint gp1 = l1->toGlobal((*(proto_segment.begin()))->localPosition()); 

  ME0DetId nesId = (*(proto_segment.begin()+1))->me0Id();
  const ME0EtaPartition* l2 = theEnsemble.second.find(nesId)->second;
  GlobalPoint gp2 = l2->toGlobal((*(proto_segment.begin()+1))->localPosition()); 


  const ME0EtaPartition* l = theEnsemble.second.find(h->me0Id())->second;
  GlobalPoint hp = l->toGlobal(h->localPosition()); 
  
  float hphi = hp.phi();          // in (-pi, +pi] 
  if (hphi < 0.) 
    hphi += 2.*M_PI;            // into range [0, 2pi) 
  float sphi = this->phiAtZ(hp.z());    // in [0, 2*pi) 
  float phidif = sphi-hphi; 
  if (phidif < 0.) 
    phidif += 2.*M_PI;          // into range [0, 2pi) 
  if (phidif > M_PI) 
    phidif -= 2.*M_PI;          // into range (-pi, pi] 

  SVector6 r_glob; 
  r_glob((*(proto_segment.begin()))->me0Id().layer()-1) = gp1.perp(); 
  r_glob((*(proto_segment.begin()+1))->me0Id().layer()-1) = gp2.perp(); 

  float R =  hp.perp(); 
  int layer = h->me0Id().layer(); 

  float r_interpolated = this->fit_r_phi(r_glob,layer); 
  float dr = fabs(r_interpolated - R); 
  
  return (fabs(phidif) <  dPhiIntMax && fabs(dr) < dRIntMax);
} 

float ME0SegAlgoRU::phiAtZ(float z) const { 

  if ( !sfit_ ) return 0.;

  // Returns a phi in [ 0, 2*pi )
  //  const ME0EtaPartition* l1 = theEnsemble.second.find((*(proto_segment.begin()))->me0Id())->second;
  const ME0EtaPartition* l1 = theEnsemble.first;
  GlobalPoint gp = l1->toGlobal(sfit_->intercept());
  GlobalVector gv = l1->toGlobal(sfit_->localdir());

  float x = gp.x() + (gv.x()/gv.z())*(z - gp.z());
  float y = gp.y() + (gv.y()/gv.z())*(z - gp.z());

  float phi = atan2(y, x);
  if (phi < 0.f ) phi += 2. * M_PI;

  return phi ;
}

bool ME0SegAlgoRU::isSegmentGood(const EnsembleHitContainer& rechitsInChamber) const { 

  // If the chamber has 20 hits or fewer, require at least 3 hits on segment 
  // If the chamber has >20 hits require at least 4 hits 
  //@@ THESE VALUES SHOULD BECOME PARAMETERS? 
  bool ok = false; 

  unsigned int iadd = ( rechitsInChamber.size() > 20)?  1 : 0;   

  if (windowScale > 1.) 
    iadd = 1; 

  if (proto_segment.size() >= 3+iadd) 
    ok = true; 
  return ok; 
} 

void ME0SegAlgoRU::flagHitsAsUsed(const EnsembleHitContainer& rechitsInChamber,  
				  BoolContainer& used ) const { 

  // Flag hits on segment as used 
  EnsembleHitContainerCIt ib = rechitsInChamber.begin(); 
  EnsembleHitContainerCIt hi, iu; 

  for(hi = proto_segment.begin(); hi != proto_segment.end(); ++hi) { 
    for(iu = ib; iu != rechitsInChamber.end(); ++iu) { 
      if(*hi == *iu) 
	used[iu-ib] = true; 
    } 
  } 
} 

bool ME0SegAlgoRU::addHit(const ME0RecHit* aHit, int layer) { 

  // Return true if hit was added successfully
  // (and then parameters are updated).
  // Return false if there is already a hit on the same layer, or insert failed.

  EnsembleHitContainer::const_iterator it;
  
  for(it = proto_segment.begin(); it != proto_segment.end(); it++)
    if (((*it)->me0Id().layer() == layer) && (aHit != (*it)))
      return false;
  proto_segment.push_back(aHit);
  // make a fit
  this->updateParameters();
  return true;
} 

void ME0SegAlgoRU::updateParameters() { 
  // update the current MuonSegFit one and make the fit
  // for ME0 we take the me0rechit from the proto_segment we transform into Tracking Rechits 
  // the local rest frame is the ME0Chamber of layer 1 the refPart
  MuonSegFit::MuonRecHitContainer muonRecHits;
  const ME0EtaPartition * refPart = theEnsemble.first;
  for (auto rh=proto_segment.begin();rh<proto_segment.end(); rh++){
    const ME0EtaPartition * thePartition   = (theEnsemble.second.find((*rh)->me0Id()))->second;
    GlobalPoint gp = thePartition->toGlobal((*rh)->localPosition());
    const LocalPoint lp = refPart->toLocal(gp);
    ME0RecHit *newRH = (*rh)->clone();
    newRH->setPosition(lp);
    MuonSegFit::MuonRecHitPtr trkRecHit(newRH);
    muonRecHits.push_back(trkRecHit);
  }
  sfit_.reset(new MuonSegFit(muonRecHits));
  sfit_->fit();
} 

float ME0SegAlgoRU::fit_r_phi(SVector6 points, int layer) const{ 
  //find R or Phi on the given layer using the given points for the interpolation 
  float Sx  = 0; 
  float Sy  = 0; 
  float Sxx = 0; 
  float Sxy = 0; 

  for (int i=1;i<7;i++){ 
    if (points(i-1)== 0.) continue; 
    Sy = Sy + (points(i-1)); 
    Sx = Sx + i; 
    Sxx = Sxx + (i*i); 
    Sxy = Sxy + ((i)*points(i-1)); 
  } 
  float delta = 2*Sxx - Sx*Sx; 
  float intercept = (Sxx*Sy - Sx*Sxy)/delta; 
  float slope = (2*Sxy - Sx*Sy)/delta;                                                                                                                        
  return (intercept + slope*layer); 
} 

void ME0SegAlgoRU::baseline(int n_seg_min){ 

  int nhits      = proto_segment.size(); 
  EnsembleHitContainer::const_iterator iRH_worst; 
  //initialise vectors for strip position and error within strip
  SVector6 sp; 
  SVector6 se;  

  unsigned int init_size = proto_segment.size(); 
  
  EnsembleHitContainer buffer; 
  buffer.clear(); 
  buffer.reserve(init_size);
  while (buffer.size()< init_size){ 
    EnsembleHitContainer::iterator min; 
    int min_layer = 99; 
    for(EnsembleHitContainer::iterator k = proto_segment.begin(); k != proto_segment.end(); k++){             
      const ME0RecHit* iRHk = *k;  
      ME0DetId idRHk = iRHk->me0Id(); 
      int kLayer   = idRHk.layer(); 
      if(kLayer < min_layer){ 
	min_layer = kLayer; 
	min = k; 
      } 
    } 
    buffer.push_back(*min); 
    proto_segment.erase(min);
  }//while 
  proto_segment.clear();  

  for (EnsembleHitContainer::const_iterator cand = buffer.begin(); cand != buffer.end(); cand++) { 
    proto_segment.push_back(*cand); 
  } 
  
  float phifirst = 0;
  bool first = true;
  // fitting the phi global position of the rechits relative to the first phi (to avoid xing the -pi, pi transition....
  for(EnsembleHitContainer::const_iterator iRH = proto_segment.begin(); iRH != proto_segment.end(); iRH++){       
    const ME0RecHit* iRHp = *iRH;
    ME0DetId idRH = iRHp->me0Id();
    int kLayer = idRH.layer();
    const ME0EtaPartition * thePartition   = (theEnsemble.second).find(idRH)->second;
    GlobalPoint gp = thePartition->toGlobal(iRHp->localPosition());
    float pphi = gp.phi();
    float prad = gp.perp();
    if (first) sp(kLayer) = 0;
    else
      sp(kLayer) =  pphi*prad -phifirst;      
    se(kLayer) = sqrt(iRHp->localPositionError().xx());
    if (first){
      phifirst = pphi*prad;
      first = false;
    }
  }

  float chi2_str; 
  this->fitX(sp, se, -1, -1, chi2_str); 

  //----------------------------------------------------- 
  // Optimal point rejection method 
  //----------------------------------------------------- 

  float minSum = 1000; 
  int i1b = 0; 
  int i2b = 0;  
  int iworst = -1;  
  int bad_layer = -1; 
  EnsembleHitContainer::const_iterator rh_to_be_deleted_1; 
  EnsembleHitContainer::const_iterator rh_to_be_deleted_2; 
  if ( nhits > n_seg_min && (chi2_str/(nhits-2)) > chi2_str_*chi2D_iadd){ 
    for (EnsembleHitContainer::const_iterator i1 = proto_segment.begin(); i1 != proto_segment.end();++i1) { 
      ++i1b; 
      const ME0RecHit* i1_1 = *i1;  
      ME0DetId idRH1 = i1_1->me0Id(); 
      int z1 = idRH1.layer(); 
      i2b = i1b; 
      for (EnsembleHitContainer::const_iterator i2 = i1+1; i2 != proto_segment.end(); ++i2) { 
	++i2b;  
	const ME0RecHit* i2_1 = *i2; 
	ME0DetId idRH2 = i2_1->me0Id(); 
	int z2 = idRH2.layer(); 
	int irej = 0; 

	for ( EnsembleHitContainer::const_iterator ir = proto_segment.begin(); ir != proto_segment.end(); ++ir) { 
	  ++irej;  

	  if (ir == i1 || ir == i2) continue;  
	  float dsum = 0; 
	  int hit_nr = 0; 
	  const ME0RecHit* ir_1 = *ir; 
	  ME0DetId idRH = ir_1->me0Id(); 
	  int worst_layer = idRH.layer(); 
	  for (EnsembleHitContainer::const_iterator i = proto_segment.begin(); i != proto_segment.end(); ++i) {  
	    ++hit_nr;   
	    const ME0RecHit* i_1 = *i; 
	    if (i == i1 || i == i2 || i == ir) continue;  
	    float slope = (sp(z2-1)-sp(z1-1))/(z2-z1); 
	    float intersept = sp(z1-1) - slope*z1; 
	    ME0DetId idRH = i_1->me0Id(); 
	    int z = idRH.layer(); 
	    float di = fabs(sp(z-1) - intersept - slope*z); 
	    dsum = dsum + di; 
	  }//i 
	  if (dsum < minSum){ 
	    minSum = dsum;  
	    bad_layer = worst_layer;		   
	    iworst = irej;  
	    rh_to_be_deleted_1 = ir; 
	  } 
	}//ir  
      }//i2 
    }//i1 
    this->fitX(sp, se, bad_layer, -1, chi2_str); 
  }//if chi2prob<1.0e-4 

  //find worst from n-1 hits 
  int iworst2 = -1; 
  int bad_layer2 = -1; 
  if (iworst > -1 && (nhits-1) > n_seg_min && (chi2_str/(nhits-3)) > chi2_str_*chi2D_iadd){ 
    iworst = -1; 
    float minSum = 1000; 
    int i1b = 0; 
    int i2b = 0;  
    for (EnsembleHitContainer::const_iterator i1 = proto_segment.begin(); i1 != proto_segment.end();++i1) { 
      ++i1b;  
      const ME0RecHit* i1_1 = *i1; 
      ME0DetId idRH1 = i1_1->me0Id(); 
      int z1 = idRH1.layer(); 
      i2b = i1b; 
      for ( EnsembleHitContainer::const_iterator i2 = i1+1; i2 != proto_segment.end(); ++i2) { 
	++i2b;  
	const ME0RecHit* i2_1 = *i2; 

	ME0DetId idRH2 = i2_1->me0Id(); 
	int z2 = idRH2.layer(); 
	int irej = 0; 

	for ( EnsembleHitContainer::const_iterator ir = proto_segment.begin(); ir != proto_segment.end(); ++ir) { 

	  ++irej;   
	  int irej2 = 0; 
	  if (ir == i1 || ir == i2 ) continue;  
	  const ME0RecHit* ir_1 = *ir; 
	  ME0DetId idRH = ir_1->me0Id(); 
	  int worst_layer = idRH.layer(); 
	  for (  EnsembleHitContainer::const_iterator ir2 = proto_segment.begin(); ir2 != proto_segment.end(); ++ir2) { 

            ++irej2;   
	    if (ir2 == i1 || ir2 == i2 || ir2 ==ir ) continue;  
	    float dsum = 0; 
	    int hit_nr = 0; 
	    const ME0RecHit* ir2_1 = *ir2; 
	    ME0DetId idRH = ir2_1->me0Id(); 
	    int worst_layer2 = idRH.layer(); 
	    for (  EnsembleHitContainer::const_iterator i = proto_segment.begin(); i != proto_segment.end(); ++i) {  
	      ++hit_nr;  
	      const ME0RecHit* i_1 = *i; 
	      if (i == i1 || i == i2 || i == ir|| i == ir2 ) continue;  
	      float slope = (sp(z2-1)-sp(z1-1))/(z2-z1); 
	      float intersept = sp(z1-1) - slope*z1; 
	      ME0DetId idRH = i_1->me0Id(); 
	      int z = idRH.layer(); 
	      float di = fabs(sp(z-1) - intersept - slope*z); 
	      dsum = dsum + di; 
	    }//i 
	    if (dsum < minSum){ 
	      minSum = dsum;  
	      iworst2 = irej2;  
	      iworst = irej; 
	      bad_layer = worst_layer; 
	      bad_layer2 = worst_layer2;  
	      rh_to_be_deleted_1 = ir; 
	      rh_to_be_deleted_2 = ir2; 
	    } 
	  }//ir2  
	}//ir  
      }//i2 
    }//i1	 

    this->fitX(sp, se, bad_layer ,bad_layer2, chi2_str); 
  }//if prob(n-1)<e-4 

  //---------------------------------- 
  //erase bad_hits 
  //---------------------------------- 

  if( iworst2-1 >= 0 && iworst2 <= int(proto_segment.size())  ) { 
    proto_segment.erase( rh_to_be_deleted_2); 
  } 

  if( iworst-1 >= 0 && iworst <= int(proto_segment.size())  ){ 
    proto_segment.erase(rh_to_be_deleted_1); 
  } 
} 

float ME0SegAlgoRU::fitX(SVector6 points, SVector6 errors, int ir, int ir2, float &chi2_str){ 

  float S   = 0; 
  float Sx  = 0; 
  float Sy  = 0; 
  float Sxx = 0; 
  float Sxy = 0; 
  float sigma2 = 0; 

  for (int i=1;i<7;i++){ 
    if (i == ir || i == ir2 || points(i-1) == 0.) continue; 
    sigma2 = errors(i-1)*errors(i-1); 
    float i1 = i - 3.5; 
    S = S + (1/sigma2); 
    Sy = Sy + (points(i-1)/sigma2); 
    Sx = Sx + ((i1)/sigma2); 
    Sxx = Sxx + (i1*i1)/sigma2; 
    Sxy = Sxy + (((i1)*points(i-1))/sigma2); 
  } 

  float delta = S*Sxx - Sx*Sx; 
  float intercept = (Sxx*Sy - Sx*Sxy)/delta; 
  float slope = (S*Sxy - Sx*Sy)/delta; 

  float chi_str = 0; 
  chi2_str = 0; 

  // calculate chi2_str 
  for (int i=1;i<7;i++){ 
    if (i == ir || i == ir2 || points(i-1) == 0.) continue; 
    chi_str = (points(i-1) - intercept - slope*(i-3.5))/(errors(i-1)); 
    chi2_str = chi2_str + chi_str*chi_str;  
  } 

  return (intercept + slope*0); 
} 

bool ME0SegAlgoRU::hasHitOnLayer(int layer) const { 

  // Is there is already a hit on this layer? 
  EnsembleHitContainerCIt it; 

  for(it = proto_segment.begin(); it != proto_segment.end(); it++) 
    if ((*it)->me0Id().layer() == layer) 
      return true;  

  return false; 
} 

bool ME0SegAlgoRU::replaceHit(const ME0RecHit* h, int layer) { 

  // replace a hit from a layer  
  EnsembleHitContainer::const_iterator it; 
  for (it = proto_segment.begin(); it != proto_segment.end();) { 
    if ((*it)->me0Id().layer() == layer) {
      it = proto_segment.erase(it); 
    } else {
      ++it;    
    }
  } 
  
  return addHit(h, layer);				     
} 

void ME0SegAlgoRU::compareProtoSegment(const ME0RecHit* h, int layer) { 
   // Copy the input MuonSegFit                                                                                                                                                      
  std::unique_ptr<MuonSegFit> oldfit;// =  new MuonSegFit( *sfit_ );                                                                                                                  

  MuonSegFit::MuonRecHitContainer muonRecHits;
  const ME0EtaPartition * refPart = theEnsemble.first;
  for (auto rh=proto_segment.begin();rh<proto_segment.end(); rh++){
    const ME0EtaPartition * thePartition   = (theEnsemble.second.find((*rh)->me0Id()))->second;
    GlobalPoint gp = thePartition->toGlobal((*rh)->localPosition());
    const LocalPoint lp = refPart->toLocal(gp);
    ME0RecHit *newRH = (*rh)->clone();
    newRH->setPosition(lp);
    MuonSegFit::MuonRecHitPtr trkRecHit(newRH);
    muonRecHits.push_back(trkRecHit);
  }
  oldfit.reset(new MuonSegFit( muonRecHits ));
  oldfit->fit();
  auto oldproto = proto_segment;
   // May create a new fit
  bool ok = this->replaceHit(h, layer);
  if ( ( sfit_->chi2() >= oldfit->chi2() ) || !ok ) {
    sfit_ = std::move(oldfit); // reset to the original input fit 
    proto_segment = oldproto;
  }
} 

void ME0SegAlgoRU::increaseProtoSegment(const ME0RecHit* h, int layer, int chi2_factor) { 

  MuonSegFit::MuonRecHitContainer muonRecHits;
  const ME0EtaPartition * refPart = theEnsemble.first;
  for (auto rh=proto_segment.begin();rh<proto_segment.end(); rh++){
    const ME0EtaPartition * thePartition   = (theEnsemble.second.find((*rh)->me0Id()))->second;
    GlobalPoint gp = thePartition->toGlobal((*rh)->localPosition());
    const LocalPoint lp = refPart->toLocal(gp);
    ME0RecHit *newRH = (*rh)->clone();
    newRH->setPosition(lp);
    MuonSegFit::MuonRecHitPtr trkRecHit(newRH);
    muonRecHits.push_back(trkRecHit);
  }

  // Creates a new fit
  std::unique_ptr<MuonSegFit> oldfit;
  oldfit.reset(new MuonSegFit( muonRecHits ));
  oldfit->fit();
  auto oldproto = proto_segment;
  bool ok = this->addHit(h, layer);
  //@@ TEST ON ndof<=0 IS JUST TO ACCEPT nhits=2 CASE??
  if ( !ok || ( (sfit_->ndof() > 0) && (sfit_->chi2()/sfit_->ndof() >= chi2Max)) ) {
    sfit_ = std::move(oldfit);
    proto_segment = oldproto;
  }
}
