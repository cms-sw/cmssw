/**
 * \file CSCSegAlgoSK.cc
 *
 *  $Date: 2013/05/28 15:41:45 $
 *  $Revision: 1.16 $
 *  \author M. Sani
 */
 
#include "CSCSegAlgoSK.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
// For clhep Matrix::solve
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

CSCSegAlgoSK::CSCSegAlgoSK(const edm::ParameterSet& ps) : CSCSegmentAlgorithm(ps),
	myName("CSCSegAlgoSK") {
	
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
  
  if (debugInfo) {
    // dump after sorting
    dumpHits(rechits);
  }
  
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
          
          if (!addHit(h1, layer1)) { 
            LogDebug("CSC") << "  failed to add hit h1\n";
            continue;
          }
          
          if (!addHit(h2, layer2)) { 
            LogDebug("CSC") << "  failed to add hit h2\n";
            continue;
          }
          
          tryAddingHitsToSegment(rechits, used, layerIndex, i1, i2); 
          
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
              
              // calculate error matrix...
              AlgebraicSymMatrix errors = calculateError();

	      // but reorder components to match what's required by TrackingRecHit interface
	      // i.e. slopes first, then positions

              flipErrors( errors );

              CSCSegment temp(proto_segment, theOrigin, theDirection, errors, theChi2); 
              
              LogDebug("CSC") << "Found a segment !!!\n";
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
  float h1x = h1->localPosition().x();
  float h2x = h2->localPosition().x();
  float deltaX = (h1->localPosition()-h2->localPosition()).x();
  LogDebug("CSC") << "    Hits at local x= " << h1x << ", " 
		  << h2x << " have separation= " << deltaX;
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
  
  // Returns a phi in [ 0, 2*pi )
  const CSCLayer* l1 = theChamber->layer((*(proto_segment.begin()))->cscDetId().layer());
  GlobalPoint gp = l1->toGlobal(theOrigin);	
  GlobalVector gv = l1->toGlobal(theDirection);	
  
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
  edm::LogInfo("CSC") << "CSCChamber rechit dump.\n";  	
  for(it=rechits.begin(); it!=rechits.end(); it++) {
    
    const CSCLayer* l1 = theChamber->layer((*it)->cscDetId().layer());
    GlobalPoint gp1 = l1->toGlobal((*it)->localPosition());	
    
    edm::LogInfo("CSC") << "Global pos.: " << gp1 << ", phi: " << gp1.phi() << ". Local position: "
			<< (*it)->localPosition() << ", phi: "
			<< (*it)->localPosition().phi() << ". Layer: "
			<< (*it)->cscDetId().layer() << "\n";
  }	
}

bool CSCSegAlgoSK::isSegmentGood(const ChamberHitContainer& rechitsInChamber) const {
  
  // If the chamber has 20 hits or fewer, require at least 3 hits on segment
  // If the chamber has >20 hits require at least 4 hits
  //@@ THESE VALUES SHOULD BECOME PARAMETERS?
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
  updateParameters();
  
  return true;
}

void CSCSegAlgoSK::updateParameters() {
  
  // Note that we need local position of a RecHit w.r.t. the CHAMBER
  // and the RecHit itself only knows its local position w.r.t.
  // the LAYER, so need to explicitly transform to global position.
  
  //  no. of hits in the RecHitsOnSegment
  //  By construction this is the no. of layers with hits
  //  since we allow just one hit per layer in a segment.
  
  int nh = proto_segment.size();
  
  // First hit added to a segment must always fail here
  if (nh < 2) 
    return;
  
  if (nh == 2) {
    
    // Once we have two hits we can calculate a straight line 
    // (or rather, a straight line for each projection in xz and yz.)
    ChamberHitContainer::const_iterator ih = proto_segment.begin();
    int il1 = (*ih)->cscDetId().layer();
    const CSCRecHit2D& h1 = (**ih);
    ++ih;    
    int il2 = (*ih)->cscDetId().layer();
    const CSCRecHit2D& h2 = (**ih);
    
    //@@ Skip if on same layer, but should be impossible
    if (il1 == il2) 
      return;
    
    const CSCLayer* layer1 = theChamber->layer(il1);
    const CSCLayer* layer2 = theChamber->layer(il2);
    
    GlobalPoint h1glopos = layer1->toGlobal(h1.localPosition());
    GlobalPoint h2glopos = layer2->toGlobal(h2.localPosition());
    
    // localPosition is position of hit wrt layer (so local z = 0)
    theOrigin = h1.localPosition();
    
    // We want hit wrt chamber (and local z will be != 0)
    LocalPoint h1pos = theChamber->toLocal(h1glopos);  
    LocalPoint h2pos = theChamber->toLocal(h2glopos);  
    
    float dz = h2pos.z()-h1pos.z();
    uz = (h2pos.x()-h1pos.x())/dz ;
    vz = (h2pos.y()-h1pos.y())/dz ;
    
    theChi2 = 0.;
  }
  else if (nh > 2) {
    
    // When we have more than two hits then we can fit projections to straight lines
    fitSlopes();  
    fillChiSquared();
  } // end of 'if' testing no. of hits
  
  fillLocalDirection(); 
}

void CSCSegAlgoSK::fitSlopes() {
  
  // Update parameters of fit
  // ptc 13-Aug-02: This does a linear least-squares fit
  // to the hits associated with the segment, in the z projection.
  
  // In principle perhaps one could fit the strip and wire
  // measurements (u, v respectively), to
  // u = u0 + uz * z
  // v = v0 + vz * z
  // where u0, uz, v0, vz are the parameters resulting from the fit.
  // But what is actually done is fit to the local x, y coordinates 
  // of the RecHits. However the strip measurement controls the precision
  // of x, and the wire measurement controls that of y.
  // Precision in local coordinate:
  //       u (strip, sigma~200um), v (wire, sigma~1cm)
  
  // I have verified that this code agrees with the formulation given
  // on p246-247 of 'Data analysis techniques for high-energy physics
  // experiments' by Bock, Grote, Notz & Regler, and that on p111-113
  // of 'Statistics' by Barlow.
  
  // Formulate the matrix equation representing the least-squares fit
  // We have a vector of measurements m, which is a 2n x 1 dim matrix
  // The transpose mT is (u1, v1, u2, v2, ..., un, vn)
  // where ui is the strip-associated measurement and vi is the
  // wire-associated measurement for a given RecHit i.
  // The fit is to
  // u = u0 + uz * z
  // v = v0 + vz * z
  // where u0, uz, v0, vz are the parameters to be obtained from the fit.
  // These are contained in a vector p which is a 4x1 dim matrix, and
  // its transpose pT is (u0, v0, uz, vz). Note the ordering!
  // The covariance matrix for each pair of measurements is 2 x 2 and
  // the inverse of this is the error matrix E.
  // The error matrix for the whole set of n measurements is a diagonal
  // matrix with diagonal elements the individual 2 x 2 error matrices
  // (because the inverse of a diagonal matrix is a diagonal matrix
  // with each element the inverse of the original.)
  
  // The matrix 'matrix' in method 'CSCSegment::weightMatrix()' is this 
  // block-diagonal overall covariance matrix. It is inverted to the 
  // block-diagonal error matrix right before it is returned.
  
  // Use the matrix A defined by
  //    1   0   z1  0
  //    0   1   0   z1
  //    1   0   z2  0
  //    0   1   0   z2
  //    ..  ..  ..  ..
  //    1   0   zn  0
  //    0   1   0   zn
  
  // The matrix A is returned by 'CSCSegment::derivativeMatrix()'.
  
  // Then the normal equations are encapsulated in the matrix equation
  //
  //    (AT E A)p = (AT E)m
  //
  // where AT is the transpose of A.
  // We'll call the combined matrix on the LHS, M, and that on the RHS, B:
  //     M p = B
  
  // We solve this for the parameter vector, p.
  // The elements of M and B then involve sums over the hits
  
  // The error matrix of the parameters is obtained by 
  // (AT E A)^-1 calculated in 'calculateError()'.
  
  // The 4 values in p can be accessed from 'CSCSegment::parameters()'
  // in the order uz, vz, u0, v0.
  
  // NOTE 1
  // Do the #hits = 2 case separately.
  // (I hope they're not on the same layer! They should not be, by construction.)
  
  // NOTE 2
  // We need local position of a RecHit w.r.t. the CHAMBER
  // and the RecHit itself only knows its local position w.r.t.
  // the LAYER, so we must explicitly transform global position.
  
  CLHEP::HepMatrix M(4,4,0);
  CLHEP::HepVector B(4,0);
  
  ChamberHitContainer::const_iterator ih = proto_segment.begin();
  
  for (ih = proto_segment.begin(); ih != proto_segment.end(); ++ih) {
    
    const CSCRecHit2D& hit = (**ih);
    const CSCLayer* layer = theChamber->layer(hit.cscDetId().layer());
    GlobalPoint gp = layer->toGlobal(hit.localPosition());
    LocalPoint  lp  = theChamber->toLocal(gp); 
    
    // ptc: Local position of hit w.r.t. chamber
    double u = lp.x();
    double v = lp.y();
    double z = lp.z();
    
    // ptc: Covariance matrix of local errors 
    CLHEP::HepMatrix IC(2,2);
    IC(1,1) = hit.localPositionError().xx();
    IC(1,2) = hit.localPositionError().xy();
    IC(2,1) = IC(1,2); // since Cov is symmetric
    IC(2,2) = hit.localPositionError().yy();
    
    // ptc: Invert covariance matrix (and trap if it fails!)
    int ierr = 0;
    IC.invert(ierr); // inverts in place
    if (ierr != 0) {
      LogDebug("CSC") << "CSCSegment::fitSlopes: failed to invert covariance matrix=\n" << IC << "\n";
      
      //@@ NOW WHAT TO DO? Exception? Return? Ignore?
      //@@ Implement throw
    }
    
    // ptc: Note that IC is no longer Cov but Cov^-1
    M(1,1) += IC(1,1);
    M(1,2) += IC(1,2);
    M(1,3) += IC(1,1) * z;
    M(1,4) += IC(1,2) * z;
    B(1) += u * IC(1,1) + v * IC(1,2);
    
    M(2,1) += IC(2,1);
    M(2,2) += IC(2,2);
    M(2,3) += IC(2,1) * z;
    M(2,4) += IC(2,2) * z;
    B(2) += u * IC(2,1) + v * IC(2,2);
    
    M(3,1) += IC(1,1) * z;
    M(3,2) += IC(1,2) * z;
    M(3,3) += IC(1,1) * z * z;
    M(3,4) += IC(1,2) * z * z;
    B(3) += ( u * IC(1,1) + v * IC(1,2) ) * z;
    
    M(4,1) += IC(2,1) * z;
    M(4,2) += IC(2,2) * z;
    M(4,3) += IC(2,1) * z * z;
    M(4,4) += IC(2,2) * z * z;
    B(4) += ( u * IC(2,1) + v * IC(2,2) ) * z;
  }
  
  // Solve the matrix equation using CLHEP's 'solve'
  //@@ ptc: CAN solve FAIL?? UNCLEAR FROM (LACK OF) CLHEP DOC
  CLHEP::HepVector p = solve(M, B);
  
  // Update member variables uz, vz, theOrigin
  // Note it has local z = 0
  theOrigin = LocalPoint(p(1), p(2), 0.);
  uz = p(3);
  vz = p(4);
}

void CSCSegAlgoSK::fillChiSquared() {
  
  // The chi-squared is (m-Ap)T E (m-Ap)
  // where T denotes transpose.
  // This collapses to a simple sum over contributions from each
  // pair of measurements.
  float u0 = theOrigin.x();
  float v0 = theOrigin.y();
  double chsq = 0.;
  
  ChamberHitContainer::const_iterator ih;
  for (ih = proto_segment.begin(); ih != proto_segment.end(); ++ih) {
    
    const CSCRecHit2D& hit = (**ih);
    const CSCLayer* layer = theChamber->layer(hit.cscDetId().layer());
    GlobalPoint gp = layer->toGlobal(hit.localPosition());
    LocalPoint lp = theChamber->toLocal(gp);  // FIX !!
    
    double hu = lp.x();
    double hv = lp.y();
    double hz = lp.z();
    
    double du = u0 + uz * hz - hu;
    double dv = v0 + vz * hz - hv;
    
    CLHEP::HepMatrix IC(2,2);
    IC(1,1) = hit.localPositionError().xx();
    IC(1,2) = hit.localPositionError().xy();
    IC(2,1) = IC(1,2);
    IC(2,2) = hit.localPositionError().yy();
    
    // Invert covariance matrix
    int ierr = 0;
    IC.invert(ierr);
    if (ierr != 0) {
      LogDebug("CSC") << "CSCSegment::fillChiSquared: failed to invert covariance matrix=\n" << IC << "\n";
      
      // @@ NOW WHAT TO DO? Exception? Return? Ignore?
    }
    
    chsq += du*du*IC(1,1) + 2.*du*dv*IC(1,2) + dv*dv*IC(2,2);
  }
  theChi2 = chsq;
}

void CSCSegAlgoSK::fillLocalDirection() {
  // Always enforce direction of segment to point from IP outwards
  // (Incorrect for particles not coming from IP, of course.)
  
  double dxdz = uz;
  double dydz = vz;
  double dz = 1./sqrt(1. + dxdz*dxdz + dydz*dydz);
  double dx = dz*dxdz;
  double dy = dz*dydz;
  LocalVector localDir(dx,dy,dz);
  
  // localDir may need sign flip to ensure it points outward from IP
  // ptc: Examine its direction and origin in global z: to point outward
  // the localDir should always have same sign as global z...
  
  double globalZpos = ( theChamber->toGlobal( theOrigin ) ).z();
  double globalZdir = ( theChamber->toGlobal( localDir ) ).z();
  double directionSign = globalZpos * globalZdir;
  
  theDirection = (directionSign * localDir).unit();
  
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
  
  // compare the chi2 of two segments
  double oldChi2 = theChi2;
  LocalPoint oldOrigin = theOrigin;
  LocalVector oldDirection = theDirection;
  ChamberHitContainer oldSegment = proto_segment;
  
  bool ok = replaceHit(h, layer);
  
  if (ok) {
    LogDebug("CSC") << "    hit in same layer as a hit on segment; try replacing old one..." 
		    << " chi2 new: " << theChi2 << " old: " << oldChi2 << "\n";
  }
  
  if ((theChi2 < oldChi2) && (ok)) {
    LogDebug("CSC")  << "    segment with replaced hit is better.\n";
  }
  else {
    proto_segment = oldSegment;
    theChi2 = oldChi2;
    theOrigin = oldOrigin;
    theDirection = oldDirection;
  }
}

void CSCSegAlgoSK::increaseProtoSegment(const CSCRecHit2D* h, int layer) {
  
  double oldChi2 = theChi2;
  LocalPoint oldOrigin = theOrigin;
  LocalVector oldDirection = theDirection;
  ChamberHitContainer oldSegment = proto_segment;
  
  bool ok = addHit(h, layer);
  
  if (ok) {
    LogDebug("CSC") << "    hit in new layer: added to segment, new chi2: " 
		    << theChi2 << "\n";
  }
  
  int ndf = 2*proto_segment.size() - 4;
  
  if (ok && ((ndf <= 0) || (theChi2/ndf < chi2Max))) {
    LogDebug("CSC") << "    segment with added hit is good.\n" ;		// FIX
  }	
  else {
    proto_segment = oldSegment;
    theChi2 = oldChi2;
    theOrigin = oldOrigin;
    theDirection = oldDirection;
  }			
}		

CLHEP::HepMatrix CSCSegAlgoSK::derivativeMatrix() const {
  
  ChamberHitContainer::const_iterator it;
  int nhits = proto_segment.size();
  CLHEP::HepMatrix matrix(2*nhits, 4);
  int row = 0;
  
  for(it = proto_segment.begin(); it != proto_segment.end(); ++it) {
    
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


AlgebraicSymMatrix CSCSegAlgoSK::weightMatrix() const {
  
  std::vector<const CSCRecHit2D*>::const_iterator it;
  int nhits = proto_segment.size();
  AlgebraicSymMatrix matrix(2*nhits, 0);
  int row = 0;
  
  for (it = proto_segment.begin(); it != proto_segment.end(); ++it) {
    
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

AlgebraicSymMatrix CSCSegAlgoSK::calculateError() const {
  
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

void CSCSegAlgoSK::flipErrors( AlgebraicSymMatrix& a ) const {

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
