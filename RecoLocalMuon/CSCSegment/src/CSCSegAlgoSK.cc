/**
 * \file CSCSegAlgoSK.cc
 *
 */
 
#include "CSCSegAlgoSK.h"
#include "CSCSegFit.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

CSCSegAlgoSK::CSCSegAlgoSK(const edm::ParameterSet& ps) 
  : CSCSegmentAlgorithm(ps), myName("CSCSegAlgoSK"), sfit_(0) {
	
  debugInfo = ps.getUntrackedParameter<bool>("verboseInfo");
    
  dRPhiMax       = ps.getParameter<double>("dRPhiMax");
  dPhiMax        = ps.getParameter<double>("dPhiMax");
  dRPhiFineMax   = ps.getParameter<double>("dRPhiFineMax");
  dPhiFineMax    = ps.getParameter<double>("dPhiFineMax");
  chi2Max        = ps.getParameter<double>("chi2Max");
  wideSeg        = ps.getParameter<double>("wideSeg");
  minLayersApart = ps.getParameter<int>("minLayersApart");
	
  LogDebug("CSC") << myName << " has algorithm cuts set to: \n"
		  << "--------------------------------------------------------------------\n"
		  << "dRPhiMax     = " << dRPhiMax << '\n'
		  << "dPhiMax      = " << dPhiMax << '\n'
		  << "dRPhiFineMax = " << dRPhiFineMax << '\n'
		  << "dPhiFineMax  = " << dPhiFineMax << '\n'
		  << "chi2Max      = " << chi2Max << '\n'
		  << "wideSeg      = " << wideSeg << '\n'
		  << "minLayersApart = " << minLayersApart << std::endl;
}

std::vector<CSCSegment> CSCSegAlgoSK::run(const CSCChamber* aChamber, const ChamberHitContainer& rechits) {
    theChamber = aChamber; 
    return buildSegments(rechits); 
}

std::vector<CSCSegment> CSCSegAlgoSK::buildSegments(const ChamberHitContainer& urechits) {
	
  LogDebug("CSC") << "*********************************************";
  LogDebug("CSC") << "Start segment building in the new chamber: " << theChamber->specs()->chamberTypeName();
  LogDebug("CSC") << "*********************************************";
  
  ChamberHitContainer rechits = urechits;
  LayerIndex layerIndex(rechits.size());
  
  for(unsigned int i = 0; i < rechits.size(); i++) {
    
    layerIndex[i] = rechits[i]->cscDetId().layer();
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
  
  //  if (debugInfo) dumpHits(rechits);

  if (rechits.size() < 2) {
    LogDebug("CSC") << myName << ": " << rechits.size() << 
      "	 hit(s) in chamber is not enough to build a segment.\n";
    return std::vector<CSCSegment>(); 
  }
	
  // We have at least 2 hits. We intend to try all possible pairs of hits to start 
  // segment building. 'All possible' means each hit lies on different layers in the chamber.
  // BUT... once a hit has been assigned to a segment, we don't consider
  // it again.
  
  // Choose first hit (as close to IP as possible) h1 and 
  // second hit (as far from IP as possible) h2
  // To do this we iterate over hits in the chamber by layer - pick two layers.
  // @@ Require the two layers are at least 3 layers apart. May need tuning?
  // Then we iterate over hits within each of these layers and pick h1 and h2 from these.
  // If they are 'close enough' we build an empty segment.
  // Then try adding hits to this segment.
  
  // Initialize flags that a given hit has been allocated to a segment
  BoolContainer used(rechits.size(), false);
  
  // Define buffer for segments we build 
  std::vector<CSCSegment> segments;

  // This is going to point to fits to hits, and its content will be used to create a CSCSegment
  sfit_ = 0;
  
  ChamberHitContainerCIt ib = rechits.begin();
  ChamberHitContainerCIt ie = rechits.end();
  
  // Possibly allow 2 passes, second widening scale factor for cuts
  windowScale = 1.; // scale factor for cuts
  
  int npass = (wideSeg > 1.)? 2 : 1;
  
  for (int ipass = 0; ipass < npass; ++ipass) {
    for (ChamberHitContainerCIt i1 = ib; i1 != ie; ++i1) {
      bool segok = false;
      if(used[i1-ib]) 
        continue;
      
      int layer1 = layerIndex[i1-ib]; //(*i1)->cscDetId().layer();
      const CSCRecHit2D* h1 = *i1;
      
      for (ChamberHitContainerCIt i2 = ie-1; i2 != i1; --i2) {
        if(used[i2-ib]) 
          continue;
        
        int layer2 = layerIndex[i2-ib]; //(*i2)->cscDetId().layer();
				
        if (abs(layer2 - layer1) < minLayersApart) 
          break;
        const CSCRecHit2D* h2 = *i2;
        
        if (areHitsCloseInLocalX(h1, h2) && areHitsCloseInGlobalPhi(h1, h2)) {
          
          proto_segment.clear();
          
          const CSCLayer* l1 = theChamber->layer(layer1);
          GlobalPoint gp1 = l1->toGlobal(h1->localPosition());					
          const CSCLayer* l2 = theChamber->layer(layer2);
          GlobalPoint gp2 = l2->toGlobal(h2->localPosition());					
          LogDebug("CSC") << "start new segment from hits " << "h1: " 
                          << gp1 << " - h2: " << gp2 << "\n";

	  //@@ TRY ADDING A HIT - AND FIT           
          if (!addHit(h1, layer1)) { 
            LogDebug("CSC") << "  failed to add hit h1\n";
            continue;
          }
          
          if (!addHit(h2, layer2)) { 
            LogDebug("CSC") << "  failed to add hit h2\n";
            continue;
          }
          
	  // Can only add hits if already have a segment
          if ( sfit_ ) tryAddingHitsToSegment(rechits, used, layerIndex, i1, i2); 
          
          // Check no. of hits on segment, and if enough flag them as used
          // and store the segment
          segok = isSegmentGood(rechits);
          if (segok) {
            flagHitsAsUsed(rechits, used);
            // Copy the proto_segment and its properties inside a CSCSegment.
            // Then fill the segment vector..
            
            if (proto_segment.empty()) {
              LogDebug("CSC") << "No segment has been found !!!\n";
            }	
            else {
	      // Create an actual CSCSegment - retrieve all info from the fit
              CSCSegment temp(sfit_->hits(), sfit_->intercept(), 
	  	        sfit_->localdir(), sfit_->covarianceMatrix(), sfit_->chi2() );
              delete sfit_;
              sfit_ = 0;              
              LogDebug("CSC") << "Found a segment !!!\n";
              if ( debugInfo ) dumpSegment( temp );
              segments.push_back(temp);	
            }
          }
        }  //   h1 & h2 close
        
        if (segok) 
          break;
      }  //  i2
    }  //  i1
    
    if (segments.size() > 1)  
      break;  // only change window if no segments found
    
    // Increase cut windows by factor of wideSeg
    windowScale = wideSeg;
    
  }  //  ipass
  
  // Give the segments to the CSCChamber
  return segments;
}

void CSCSegAlgoSK::tryAddingHitsToSegment(const ChamberHitContainer& rechits, 
					  const BoolContainer& used, const LayerIndex& layerIndex,
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
    if (i == i1 || i == i2 || used[i-ib]) 
      continue; 
    
    int layer = layerIndex[i-ib]; 
    const CSCRecHit2D* h = *i;
    if (isHitNearSegment(h)) {
      
      GlobalPoint gp1 = theChamber->layer(layer)->toGlobal(h->localPosition());		
      LogDebug("CSC") << "    hit at global " << gp1 << " is near segment\n.";
      
      // Don't consider alternate hits on layers holding the two starting points
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

bool CSCSegAlgoSK::areHitsCloseInLocalX(const CSCRecHit2D* h1, const CSCRecHit2D* h2) const {
  float deltaX = ( h1->localPosition() - h2->localPosition() ).x();
  LogDebug("CSC") << "    Hits at local x= " << h1->localPosition().x() << ", " 
		  << h2->localPosition().x() << " have separation= " << deltaX;
  return (fabs(deltaX) < (dRPhiMax * windowScale))? true:false;   // +v
}

bool CSCSegAlgoSK::areHitsCloseInGlobalPhi(const CSCRecHit2D* h1, const CSCRecHit2D* h2) const {
  
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
  return (fabs(dphi12) < (dPhiMax * windowScale))? true:false;  // +v
}

bool CSCSegAlgoSK::isHitNearSegment(const CSCRecHit2D* h) const {
  
  // Is hit near segment? 
  // Requires deltaphi and rxy*deltaphi within ranges specified
  // in parameter set, where rxy=sqrt(x**2+y**2) of hit itself.
  // Note that to make intuitive cuts on delta(phi) one must work in
  // phi range (-pi, +pi] not [0, 2pi)
  
  const CSCLayer* l1 = theChamber->layer(h->cscDetId().layer());
  GlobalPoint hp = l1->toGlobal(h->localPosition());	
  
  float hphi = hp.phi();          // in (-pi, +pi]
  if (hphi < 0.) 
    hphi += 2.*M_PI;            // into range [0, 2pi)
  float sphi = phiAtZ(hp.z());    // in [0, 2*pi)
  float phidif = sphi-hphi;
  if (phidif < 0.) 
    phidif += 2.*M_PI;          // into range [0, 2pi)
  if (phidif > M_PI) 
    phidif -= 2.*M_PI;          // into range (-pi, pi]
  
  float dRPhi = fabs(phidif)*hp.perp();
  LogDebug("CSC") << "    is hit at phi_h= " << hphi << " near segment phi_seg= " << sphi 
		  << "? is " << dRPhi << "<" << dRPhiFineMax*windowScale << " ? " 
		  << " and is |" << phidif << "|<" << dPhiFineMax*windowScale << " ?";
  
  return ((dRPhi < dRPhiFineMax*windowScale) && 
	  (fabs(phidif) < dPhiFineMax*windowScale))? true:false;  // +v
}

float CSCSegAlgoSK::phiAtZ(float z) const {
  
  if ( !sfit_ ) {
      edm::LogVerbatim("CSCSegment") << "[CSCSegAlgoSK::phiAtZ] Segment fit undefined";
      return 0.;
  }

  // Returns a phi in [ 0, 2*pi )
  const CSCLayer* l1 = theChamber->layer((*(proto_segment.begin()))->cscDetId().layer());
  GlobalPoint gp = l1->toGlobal(sfit_->intercept());	
  GlobalVector gv = l1->toGlobal(sfit_->localdir());	

  LogTrace("CSCSegment") << "[CSCSegAlgoSK::phiAtZ] Global intercept = " << gp << ", direction = " << gv;  

  float x = gp.x() + (gv.x()/gv.z())*(z - gp.z());
  float y = gp.y() + (gv.y()/gv.z())*(z - gp.z());
  float phi = atan2(y, x);
  if (phi < 0.f ) 
    phi += 2. * M_PI;
  
  return phi ;
}

void CSCSegAlgoSK::dumpHits(const ChamberHitContainer& rechits) const {
  
  // Dump positions of RecHit's in each CSCChamber
  ChamberHitContainerCIt it;
  edm::LogInfo("CSCSegment") << "CSCChamber rechit dump.\n";  	
  for(it=rechits.begin(); it!=rechits.end(); it++) {
    
    const CSCLayer* l1 = theChamber->layer((*it)->cscDetId().layer());
    GlobalPoint gp1 = l1->toGlobal((*it)->localPosition());	
    
    edm::LogInfo("CSCSegment") << "Global pos.: " << gp1 << ", phi: " << gp1.phi() << ". Local position: "
			<< (*it)->localPosition() << ", phi: "
			<< (*it)->localPosition().phi() << ". Layer: "
			<< (*it)->cscDetId().layer() << "\n";
  }	
}

bool CSCSegAlgoSK::isSegmentGood(const ChamberHitContainer& rechitsInChamber) const {
  
  // If the chamber has 20 hits or fewer, require at least 3 hits on segment
  // If the chamber has >20 hits require at least 4 hits

  bool ok = false;
  
  unsigned int iadd = ( rechitsInChamber.size() > 20 )?  1 : 0;  
  
  if (windowScale > 1.)
    iadd = 1;
  
  if (proto_segment.size() >= 3+iadd)
    ok = true;
  
  return ok;
}

void CSCSegAlgoSK::flagHitsAsUsed(const ChamberHitContainer& rechitsInChamber, 
				  BoolContainer& used ) const {
  
  // Flag hits on segment as used
  ChamberHitContainerCIt ib = rechitsInChamber.begin();
  ChamberHitContainerCIt hi, iu;
  
  for(hi = proto_segment.begin(); hi != proto_segment.end(); ++hi) {
    for(iu = ib; iu != rechitsInChamber.end(); ++iu) {
      if(*hi == *iu)
	used[iu-ib] = true;
    }
  }
}

bool CSCSegAlgoSK::addHit(const CSCRecHit2D* aHit, int layer) {
  
  // Return true if hit was added successfully 
  // (and then parameters are updated).
  // Return false if there is already a hit on the same layer, or insert failed.
  
  ChamberHitContainer::const_iterator it;

  for(it = proto_segment.begin(); it != proto_segment.end(); it++)
    if (((*it)->cscDetId().layer() == layer) && (aHit != (*it)))
       return false; 
  
  proto_segment.push_back(aHit);

  // make a fit
  updateParameters();
  return true;
}

void CSCSegAlgoSK::updateParameters(void ) {

  // Delete input CSCSegFit, create a new one and make the fit
  delete sfit_;
  sfit_ = new CSCSegFit( theChamber, proto_segment );
  sfit_->fit();
}

bool CSCSegAlgoSK::hasHitOnLayer(int layer) const {
  
  // Is there is already a hit on this layer?
  ChamberHitContainerCIt it;
  
  for(it = proto_segment.begin(); it != proto_segment.end(); it++)
    if ((*it)->cscDetId().layer() == layer)
      return true; 
  
  return false;
}

bool CSCSegAlgoSK::replaceHit(const CSCRecHit2D* h, int layer) {
  
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

void CSCSegAlgoSK::compareProtoSegment(const CSCRecHit2D* h, int layer) {
  
  // Copy the input CSCSegFit
  CSCSegFit* oldfit = new CSCSegFit( *sfit_ );
  
  // May create a new fit
  bool ok = replaceHit(h, layer);
  
  if (ok) {
    LogDebug("CSCSegment") << "    hit in same layer as a hit on segment; try replacing old one..." 
		    << " chi2 new: " << sfit_->chi2() << " old: " << oldfit->chi2() << "\n";
  }
  
  if ( ( sfit_->chi2() < oldfit->chi2() ) && ok ) {
    LogDebug("CSC")  << "    segment with replaced hit is better.\n";
    delete oldfit;  // new fit is better 
  }
  else {
    // keep original fit
    delete sfit_; // now the new fit
    sfit_ = oldfit;  // reset to the original input fit
  }
}

void CSCSegAlgoSK::increaseProtoSegment(const CSCRecHit2D* h, int layer) {
  
  // Copy input fit
  CSCSegFit* oldfit = new CSCSegFit( *sfit_ );

  // Creates a new fit  
  bool ok = addHit(h, layer);
  
  if (ok) {
    LogDebug("CSCSegment") << "    hit in new layer: added to segment, new chi2: " 
		    << sfit_->chi2() << "\n";
  }
  
  //  int ndf = 2*proto_segment.size() - 4;

  //@@ TEST ON ndof<=0 IS JUST TO ACCEPT nhits=2 CASE??  
  if ( ok && ( (sfit_->ndof() <= 0) || (sfit_->chi2()/sfit_->ndof() < chi2Max)) ) {
    LogDebug("CSCSegment") << "    segment with added hit is good.\n" ; 
    delete oldfit;  // new fit is better 
  }	
  else {
    // reset to original fit
    delete sfit_;
    sfit_ = oldfit;
  }			
}		

void CSCSegAlgoSK::dumpSegment( const CSCSegment& seg ) const {

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
