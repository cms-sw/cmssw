/**
 * \file CSCSegAlgoTC.cc
 *
 * Last update: 13.02.2015
 *
 */

#include "CSCSegAlgoTC.h"
#include "CSCSegFit.h"

#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "Geometry/CSCGeometry/interface/CSCLayer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

CSCSegAlgoTC::CSCSegAlgoTC(const edm::ParameterSet& ps) 
  : CSCSegmentAlgorithm(ps), sfit_(0), myName("CSCSegAlgoTC") {
  
  debugInfo = ps.getUntrackedParameter<bool>("verboseInfo");
  
  dRPhiMax       = ps.getParameter<double>("dRPhiMax");
  dPhiMax        = ps.getParameter<double>("dPhiMax");
  dRPhiFineMax   = ps.getParameter<double>("dRPhiFineMax");
  dPhiFineMax    = ps.getParameter<double>("dPhiFineMax");
  chi2Max        = ps.getParameter<double>("chi2Max");
  chi2ndfProbMin = ps.getParameter<double>("chi2ndfProbMin");
  minLayersApart = ps.getParameter<int>("minLayersApart");
  SegmentSorting = ps.getParameter<int>("SegmentSorting");
  
  LogDebug("CSC") << myName << " has algorithm cuts set to: \n"
		  << "--------------------------------------------------------------------\n"
		  << "dRPhiMax     = " << dRPhiMax << '\n'
		  << "dPhiMax      = " << dPhiMax << '\n'
		  << "dRPhiFineMax = " << dRPhiFineMax << '\n'
		  << "dPhiFineMax  = " << dPhiFineMax << '\n'
		  << "chi2Max      = " << chi2Max << '\n'
		  << "chi2ndfProbMin = " << chi2ndfProbMin << '\n'
		  << "minLayersApart = " << minLayersApart << '\n'
		  << "SegmentSorting = " << SegmentSorting << std::endl;
}

std::vector<CSCSegment> CSCSegAlgoTC::run(const CSCChamber* aChamber, const ChamberHitContainer& rechits) {
  theChamber = aChamber; 

  return buildSegments(rechits); 
}

std::vector<CSCSegment> CSCSegAlgoTC::buildSegments(const ChamberHitContainer& urechits) {
  
  //  ChamberHitContainer rechits = urechits;
  ChamberHitContainer rechits(urechits);

  //  edm::LogVerbatim("CSCSegment") << "[CSCSegAlgoTC::buildSegments] start building segments in " << theChamber->id();
  //  edm::LogVerbatim("CSCSegment") << "[CSCSegAlgoTC::buildSegments] size of rechit container = " << rechits.size();

  if (rechits.size() < 2) {
    LogDebug("CSC") << myName << ": " << rechits.size() << 
      "	 hit(s) in chamber is not enough to build a segment.\n";
    return std::vector<CSCSegment>(); 
  }

  LayerIndex layerIndex(rechits.size());

  for ( size_t i = 0; i < rechits.size(); ++i ) {
    short ilay = rechits[i]->cscDetId().layer();
    layerIndex[i] = ilay;
    //    edm::LogVerbatim("CSCSegment") << "layerIndex[" << i << "] should   be " << rechits[i]->cscDetId().layer();
    //    edm::LogVerbatim("CSCSegment") << "layerIndex[" << i << "] actually is " << layerIndex[i];
  }
  
  double z1 = theChamber->layer(1)->position().z();
  double z6 = theChamber->layer(6)->position().z();
  
  if ( z1 > 0. ) {
    if ( z1 > z6 ) { 
      reverse(layerIndex.begin(), layerIndex.end());
      reverse(rechits.begin(), rechits.end());
    }    
  }
  else if ( z1 < 0. ) {
    if ( z1 < z6 ) {
      reverse(layerIndex.begin(), layerIndex.end());
      reverse(rechits.begin(), rechits.end());
    }    
  }

  // Dump rechits after sorting?
  //  if (debugInfo) dumpHits(rechits);

  //  if (rechits.size() < 2) {
  //    LogDebug("CSC") << myName << ": " << rechits.size() << 
  //      "	 hit(s) in chamber is not enough to build a segment.\n";
  //    return std::vector<CSCSegment>(); 
  //  }
  
  // We have at least 2 hits. We intend to try all possible pairs of hits to start 
  // segment building. 'All possible' means each hit lies on different layers in the chamber.
  // For now we don't care whether a hit has already been allocated to another segment;
  // we'll sort that out after building all possible segments.
  
  // Choose first hit (as close to IP as possible) h1 and 
  // second hit (as far from IP as possible) h2
  // To do this we iterate over hits in the chamber by layer - pick two layers.
  // @@ Require the two layers are at least 3 layers apart. May need tuning?
  // Then we iterate over hits within each of these layers and pick h1 and h2 from these.
  // If they are 'close enough' we build an empty segment.
  // Then try adding hits to this segment.
  
  // Define buffer for segments we build (at the end we'll sort them somehow, and remove
  // those which share hits with better-quality segments.
  
  
  std::vector<CSCSegment> segments;

  sfit_ = 0;
  
  ChamberHitContainerCIt ib = rechits.begin();
  ChamberHitContainerCIt ie = rechits.end();
  
  for (ChamberHitContainerCIt i1 = ib; i1 != ie; ++i1) {
    
    int layer1 = layerIndex[i1-ib];
  
    const CSCRecHit2D* h1 = *i1;
    
    for (ChamberHitContainerCIt i2 = ie-1; i2 != i1; --i2) {
      
      int layer2 = layerIndex[i2-ib];
      
      if (abs(layer2 - layer1) < minLayersApart)
        break;
      
      const CSCRecHit2D* h2 = *i2;
      
      if (areHitsCloseInLocalX(h1, h2) && areHitsCloseInGlobalPhi(h1, h2)) {
        
        proto_segment.clear();
        
        const CSCLayer* l1 = theChamber->layer(h1->cscDetId().layer());
        GlobalPoint gp1 = l1->toGlobal(h1->localPosition());					
        const CSCLayer* l2 = theChamber->layer(h2->cscDetId().layer());
        GlobalPoint gp2 = l2->toGlobal(h2->localPosition());					
        LogDebug("CSCSegment") << "start new segment from hits " << "h1: " << gp1 << " - h2: " << gp2 << "\n";
	//	edm::LogVerbatim("CSCSegment") << "start new segment from hits " << "h1: " << gp1 << " - h2: " << gp2;
	//	edm::LogVerbatim("CSCSegment") << "on layers " << layer1 << " and " << layer2;
        
        if ( !addHit(h1, layer1) ) { 
          LogDebug("CSCSegment") << "  failed to add hit h1\n";
          continue;
        }
        
        if ( !addHit(h2, layer2) ) { 
          LogDebug("CSCSegment") << "  failed to add hit h2\n";
          continue;
        }
        
        if ( sfit_ ) tryAddingHitsToSegment(rechits, i1, i2); // can only add hits if there's a segment
        
        // if a segment has been found push back it into the segment collection
        if (proto_segment.empty()) {
          LogDebug("CSCSegment") << "No segment found.\n";
	  //	  edm::LogVerbatim("CSCSegment") << "No segment found.";
        }	
        else {
	  //@@ THIS IS GOING TO BE TRICKY - CREATING MANY FITS ON HEAP
	  //@@ BUT MEMBER VARIABLE JUST POINTS TO MOST RECENT ONE
          candidates.push_back( sfit_ ); // store the current fit
          LogDebug("CSCSegment") << "Found a segment.\n";
	  //	  edm::LogVerbatim("CSCSegment") << "Found a segment.";
        }
      }
    }
  }

  //  edm::LogVerbatim("CSCSegment") << "[CSCSegAlgoTC::buildSegments] no. of candidates before pruning = " << candidates.size();
  
  // We've built all possible segments. Now pick the best, non-overlapping subset.
  pruneTheSegments(rechits);
  
  // Create CSCSegments for the surviving candidates
  for(unsigned int i = 0; i < candidates.size(); ++i ) {
    CSCSegFit*sfit = candidates[i];
    //    edm::LogVerbatim("CSCSegment") << "candidate fit " << i+1 << " of " << candidates.size() << " is at " << sfit;
    //    if ( !sfit ) {
    //      edm::LogVerbatim("CSCSegment") << "stored a null pointer for element " << i+1 << " of " << candidates.size();
    //      continue;
    //    }
    CSCSegment temp(sfit->hits(), sfit->intercept(), sfit->localdir(), 
		    sfit->covarianceMatrix(), sfit->chi2() );    
    delete sfit;
    segments.push_back(temp);	
    if (debugInfo) dumpSegment( temp );
  }

  // reset member variables  
  candidates.clear();
  sfit_ = 0;

  //  edm::LogVerbatim("CSCSegment") << "[CSCSegAlgoTC::buildSegments] no. of segments returned = " << segments.size();

  return segments;
}

void CSCSegAlgoTC::tryAddingHitsToSegment(const ChamberHitContainer& rechits, 
		 const ChamberHitContainerCIt i1, const ChamberHitContainerCIt i2) {
  
  // Iterate over the layers with hits in the chamber
  // Skip the layers containing the segment endpoints
  // Test each hit on the other layers to see if it is near the segment
  // If it is, see whether there is already a hit on the segment from the same layer
  //    - if so, and there are more than 2 hits on the segment, copy the segment,
  //      replace the old hit with the new hit. If the new segment chi2 is better
  //      then replace the original segment with the new one (by swap)
  //    - if not, copy the segment, add the hit. If the new chi2/dof is still satisfactory
  //      then replace the original segment with the new one (by swap)

  ChamberHitContainerCIt ib = rechits.begin();
  ChamberHitContainerCIt ie = rechits.end();
  
  for (ChamberHitContainerCIt i = ib; i != ie; ++i) {
    
    if ( i == i1 || i == i2 ) 
      continue;
    
    int layer = (*i)->cscDetId().layer();
    const CSCRecHit2D* h = *i;
    
    if (isHitNearSegment(h)) {
      
      const CSCLayer* l1 = theChamber->layer(h->cscDetId().layer());
      GlobalPoint gp1 = l1->toGlobal(h->localPosition());		
      LogDebug("CSC") << "    hit at global " << gp1 << " is near segment\n.";
      
      if (hasHitOnLayer(layer)) {
	if (proto_segment.size() <= 2) {
	  LogDebug("CSC") << "    " << proto_segment.size() 
			  << " hits on segment...skip hit on same layer.\n";
	  continue;
	}
	
	compareProtoSegment(h, layer);
      } 
      else
	increaseProtoSegment(h, layer);
    }   // h & seg close
  }   // i
}

bool CSCSegAlgoTC::addHit(const CSCRecHit2D* aHit, int layer) {
  
  // Return true if hit was added successfully 
  // (and then parameters are updated).
  // Return false if there is already a hit on the same layer, or insert failed.

  bool ok = true;
  ChamberHitContainer::const_iterator it;
  
  for(it = proto_segment.begin(); it != proto_segment.end(); it++)
    if (((*it)->cscDetId().layer() == layer) && (aHit != *it))
      return false; 
  
  if (ok) {
    proto_segment.push_back(aHit);
    updateParameters();
  }	
  return ok;
}

bool CSCSegAlgoTC::replaceHit(const CSCRecHit2D* h, int layer) {

  // replace a hit from a layer 
  ChamberHitContainer::iterator it;
  for (it = proto_segment.begin(); it != proto_segment.end();) {
    if ((*it)->cscDetId().layer() == layer)
      it = proto_segment.erase(it);
    else
      ++it;   
  }
  
  return addHit(h, layer);

}

void CSCSegAlgoTC::updateParameters(void) {
  
  //@@ DO NOT DELETE EXISTING FIT SINCE WE SAVE IT!!
  //  delete sfit_;
  sfit_ = new CSCSegFit( theChamber, proto_segment );
  sfit_->fit();

}

float CSCSegAlgoTC::phiAtZ(float z) const {
  
  // Returns a phi in [ 0, 2*pi )
  const CSCLayer* l1 = theChamber->layer(1);
  GlobalPoint gp = l1->toGlobal(sfit_->intercept());	
  GlobalVector gv = l1->toGlobal(sfit_->localdir());	
  
  float x = gp.x() + (gv.x()/gv.z())*(z - gp.z());
  float y = gp.y() + (gv.y()/gv.z())*(z - gp.z());
  float phi = atan2(y, x);
  if (phi < 0.f) 
    phi += 2. * M_PI;
  
  return phi ;
}

void CSCSegAlgoTC::compareProtoSegment(const CSCRecHit2D* h, int layer) {

  // Save copy of current fit
  CSCSegFit* oldfit = new CSCSegFit( *sfit_ );

  bool ok = replaceHit(h, layer); // possible new fit
  
  if (ok) {
    LogDebug("CSC") << "    hit in same layer as a hit on segment; try replacing old one..." 
		    << " chi2 new: " << sfit_->chi2() << " old: " << oldfit->chi2() << "\n";
  }
  
  if ( ok && (sfit_->chi2() < oldfit->chi2()) ) {
    LogDebug("CSC")  << "    segment with replaced hit is better.\n";
    delete oldfit;  // new fit is better
  }
  else {
    delete sfit_;   // new fit is worse
    sfit_ = oldfit; // restore original fit
  }
}

void CSCSegAlgoTC::increaseProtoSegment(const CSCRecHit2D* h, int layer) {
  
  // save copy of input fit
  CSCSegFit* oldfit = new CSCSegFit( *sfit_ );

  bool ok = addHit(h, layer); // possible new fit
  
  if (ok) {
    LogDebug("CSC") << "    hit in new layer: added to segment, new chi2: " 
		    << sfit_->chi2() << "\n";
  }
  
  //  int ndf = 2*proto_segment.size() - 4;
  
  if (ok && ((sfit_->chi2() <= 0) || (sfit_->chi2()/sfit_->ndof() < chi2Max))) {
    LogDebug("CSC") << "    segment with added hit is good.\n" ;		
    delete oldfit; // new fit is better
  }	
  else {
    delete sfit_;   // new fit is worse
    sfit_ = oldfit; // restore original fit
  }			
}		

bool CSCSegAlgoTC::areHitsCloseInLocalX(const CSCRecHit2D* h1, const CSCRecHit2D* h2) const {
  float deltaX = (h1->localPosition()-h2->localPosition()).x();
  LogDebug("CSC") << "    Hits at local x= " << h1->localPosition().x() << ", " 
		  << h2->localPosition().x() << " have separation= " << deltaX;
  return (fabs(deltaX) < (dRPhiMax))? true:false;   // +v
}

bool CSCSegAlgoTC::areHitsCloseInGlobalPhi(const CSCRecHit2D* h1, const CSCRecHit2D* h2) const {
  
  const CSCLayer* l1 = theChamber->layer(h1->cscDetId().layer());
  GlobalPoint gp1 = l1->toGlobal(h1->localPosition());	
  const CSCLayer* l2 = theChamber->layer(h2->cscDetId().layer());
  GlobalPoint gp2 = l2->toGlobal(h2->localPosition());	
  
  float h1p = gp1.phi();
  float h2p = gp2.phi();
  float dphi12 = h1p - h2p;
  
  // Into range [-pi, pi) (phi() returns values in this range)
  if (dphi12 < -M_PI) 
    dphi12 += 2.*M_PI;  
  if (dphi12 >  M_PI) 
    dphi12 -= 2.*M_PI;
  LogDebug("CSC") << "    Hits at global phi= " << h1p << ", " 
		  << h2p << " have separation= " << dphi12;
  return (fabs(dphi12) < dPhiMax)? true:false;  // +v
}

bool CSCSegAlgoTC::isHitNearSegment(const CSCRecHit2D* h) const {
  
  // Is hit near segment? 
  // Requires deltaphi and rxy*deltaphi within ranges specified
  // in orcarc, or by default, where rxy=sqrt(x**2+y**2) of hit itself.
  // Note that to make intuitive cuts on delta(phi) one must work in
  // phi range (-pi, +pi] not [0, 2pi

  const CSCLayer* l1 = theChamber->layer(h->cscDetId().layer());
  GlobalPoint hp = l1->toGlobal(h->localPosition());	
  
  float hphi = hp.phi();               // in (-pi, +pi]
  if (hphi < 0.) 
    hphi += 2.*M_PI;                   // into range [0, 2pi)
  float sphi = phiAtZ(hp.z());         // in [0, 2*pi)
  float phidif = sphi-hphi;
  if (phidif < 0.) 
    phidif += 2.*M_PI;                 // into range [0, 2pi)
  if (phidif > M_PI) 
    phidif -= 2.*M_PI;                 // into range (-pi, pi]
  
  float dRPhi = fabs(phidif)*hp.perp();
  LogDebug("CSC") << "    is hit at phi_h= " << hphi << " near segment phi_seg= " << sphi 
		  << "? is " << dRPhi << "<" << dRPhiFineMax << " ? " 
		  << " and is |" << phidif << "|<" << dPhiFineMax << " ?";
  
  return ((dRPhi < dRPhiFineMax) && 
	  (fabs(phidif) < dPhiFineMax))? true:false;  // +v
}

bool CSCSegAlgoTC::hasHitOnLayer(int layer) const {
  
  // Is there is already a hit on this layer?
  ChamberHitContainerCIt it;
  
  for(it = proto_segment.begin(); it != proto_segment.end(); it++)
    if ((*it)->cscDetId().layer() == layer)
      return true; 
  
  return false;
}

void CSCSegAlgoTC::dumpHits(const ChamberHitContainer& rechits) const {
  
  // Dump positions of RecHit's in each CSCChamber
  ChamberHitContainerCIt it;
  
  for(it=rechits.begin(); it!=rechits.end(); it++) {
    
    const CSCLayer* l1 = theChamber->layer((*it)->cscDetId().layer());
    GlobalPoint gp1 = l1->toGlobal((*it)->localPosition());	
    
    LogDebug("CSC") << "Global pos.: " << gp1 << ", phi: " << gp1.phi() << ". Local position: "
		    << (*it)->localPosition() << ", phi: "
		    << (*it)->localPosition().phi() << ". Layer: "
		    << (*it)->cscDetId().layer() << "\n";
  }	
}



bool CSCSegAlgoTC::isSegmentGood(std::vector<CSCSegFit*>::iterator seg, 
				 const ChamberHitContainer& rechitsInChamber, BoolContainer& used) const {
  
  // Apply any selection cuts to segment
  
  // 1) Require a minimum no. of hits
  //   (@@ THESE VALUES SHOULD BECOME PARAMETERS?)
  
  // 2) Ensure no hits on segment are already assigned to another segment
  //    (typically of higher quality)

  size_t iadd = (rechitsInChamber.size() > 20 )?  1 : 0;  
  
  size_t nhits = (*seg)->nhits();

  if (nhits < 3 + iadd)
    return false;
  
  // Additional part of alternative segment selection scheme: reject
  // segments with a chi2 probability of less than chi2ndfProbMin. Relies on list 
  // being sorted with "SegmentSorting == 2", that is first by nrechits and then 
  // by chi2prob in subgroups of same no. of rechits.

  if( SegmentSorting == 2 ){
    double chi2t = (*seg)->chi2();
    double ndoft = 2*nhits - 4 ;
    if( chi2t > 0 && ndoft > 0 )  {
      if ( ChiSquaredProbability(chi2t,ndoft) < chi2ndfProbMin ) { 
	return false;
      }
    }
    else {
      return false; 
    }
  }
  
  ChamberHitContainer hits_ = (*seg)->hits();

  for(size_t ish = 0; ish < nhits; ++ish) {
    
    ChamberHitContainerCIt ib = rechitsInChamber.begin();
    
    for(ChamberHitContainerCIt ic = ib; ic != rechitsInChamber.end(); ++ic) {

      if((hits_[ish] == (*ic)) && used[ic-ib])
	return false;
    }
  }
  
  return true;
}

void CSCSegAlgoTC::flagHitsAsUsed(std::vector<CSCSegFit*>::iterator seg, 
      const ChamberHitContainer& rechitsInChamber, BoolContainer& used) const {

  // Flag hits on segment as used

  ChamberHitContainerCIt ib = rechitsInChamber.begin();
  ChamberHitContainer hits = (*seg)->hits();
   
  for(size_t ish = 0; ish < hits.size(); ++ish) {
    for(ChamberHitContainerCIt iu = ib; iu != rechitsInChamber.end(); ++iu)
      if( hits[ish] == (*iu)) 
	used[iu-ib] = true;
  }
}

void CSCSegAlgoTC::pruneTheSegments(const ChamberHitContainer& rechitsInChamber) {
  
  // Sort the segment store according to segment 'quality' (chi2/#hits ?) and
  // remove any segments which contain hits assigned to higher-quality segments.

  if (candidates.empty()) 
    return;
  
  // Initialize flags that a given hit has been allocated to a segment
  BoolContainer used(rechitsInChamber.size(), false);
  
  // Sort by chi2/#hits
  segmentSort();
  
  // Select best quality segments, requiring hits are assigned to just one segment
  // Want to erase the bad segments, so the iterator must be incremented
  // inside the loop, and only when the erase is not called

  std::vector<CSCSegFit*>::iterator is;

  for (is = candidates.begin();  is !=  candidates.end(); ) {
    
    bool goodSegment = isSegmentGood(is, rechitsInChamber, used);
    
    if (goodSegment) {
      LogDebug("CSC") << "Accepting segment: ";
      flagHitsAsUsed(is, rechitsInChamber, used);
      ++is;
    }
    else {
      LogDebug("CSC") << "Rejecting segment: ";
      delete *is; // delete the CSCSegFit* 
      is = candidates.erase(is); // erase the element in container
    }
  }
}

void CSCSegAlgoTC::segmentSort() {
  
  // The segment collection is sorted according to e.g. chi2/#hits 
  
  for(size_t i=0; i<candidates.size()-1; ++i) {
    for(size_t j=i+1; j<candidates.size(); ++j) {
      
      size_t ni = (candidates[i]->hits()).size();
      size_t nj = (candidates[j]->hits()).size();
      
      // Sort criterion: first sort by no. of rechits, then in groups of rechits by chi2prob
      if( SegmentSorting == 2 ){ 
	if ( nj > ni ) { // sort by no. of rechits
	  CSCSegFit* temp = candidates[j];
	  candidates[j] = candidates[i];
	  candidates[i] = temp;
	}
	// sort by chi2 probability in subgroups with equal nr of rechits
	//	if(chi2s[i] != 0. && 2*n2-4 > 0 ) {
        if( candidates[i]->chi2() > 0 && candidates[i]->ndof() > 0 ) {
	  if( nj == ni && 
	      ( ChiSquaredProbability( candidates[i]->chi2(),(double)(candidates[i]->ndof()) ) < 
	        ChiSquaredProbability( candidates[j]->chi2(),(double)(candidates[j]->ndof()) ) ) 
            ){
	  CSCSegFit* temp = candidates[j];
	  candidates[j] = candidates[i];
	  candidates[i] = temp;
	  }
	}
      }
      else if( SegmentSorting == 1 ){
	if ((candidates[i]->chi2()/ni) > (candidates[j]->chi2()/nj)) {
	  CSCSegFit* temp = candidates[j];
	  candidates[j] = candidates[i];
	  candidates[i] = temp;
	}
      }
      else{
          LogDebug("CSC") << "No valid segment sorting specified. Algorithm misconfigured! \n";
      }
    }
  }     
}

void CSCSegAlgoTC::dumpSegment( const CSCSegment& seg ) const {
  
  edm::LogVerbatim("CSCSegment") << "CSCSegment in " << theChamber->id()
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
