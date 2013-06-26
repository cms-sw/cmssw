/**
 * \file CSCSegAlgoHitPruning.cc
 *
 *  \authors: S. Stoynev  - NU
 *            I. Bloch    - FNAL
 *            E. James    - FNAL
 *            D. Fortin   - UC Riverside
 *
 * See header file for description.
 */

#include "RecoLocalMuon/CSCSegment/src/CSCSegAlgoHitPruning.h"

#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>


/* Constructor
 *
 */
CSCSegAlgoHitPruning::CSCSegAlgoHitPruning(const edm::ParameterSet& ps) {
  BrutePruning           = ps.getParameter<bool>("BrutePruning");

}


/* Destructor:
 *
 */
CSCSegAlgoHitPruning::~CSCSegAlgoHitPruning(){

}


/* pruneBadHits
 *
 */
std::vector<CSCSegment> CSCSegAlgoHitPruning::pruneBadHits(const CSCChamber* aChamber, const std::vector<CSCSegment>& _segments) {

  theChamber = aChamber;

  std::vector<CSCSegment>          segments_temp;
  std::vector<ChamberHitContainer> rechits_clusters; 
  std::vector<CSCSegment> segments = _segments;
  const float chi2ndfProbMin = 1.0e-4;
  bool use_brute_force = BrutePruning;

  int hit_nr = 0;
  int hit_nr_worst = -1;
  //int hit_nr_2ndworst = -1;
  
  for (std::vector<CSCSegment>::iterator it=segments.begin(); it != segments.end(); it++) {
    
    if ( !use_brute_force ) {// find worst hit
      
      float chisq    = (*it).chi2();
      int nhits      = (*it).nRecHits();
      LocalPoint localPos = (*it).localPosition();
      LocalVector segDir = (*it).localDirection();
      const CSCChamber* cscchamber = theChamber;
      float globZ       ;
          
      GlobalPoint globalPosition = cscchamber->toGlobal(localPos);
      globZ = globalPosition.z();
      
      
      if ( ChiSquaredProbability((double)chisq,(double)(2*nhits-4)) < chi2ndfProbMin  ) {

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

        for ( std::vector<CSCRecHit2D>::const_iterator iRH = theseRecHits.begin(); iRH != theseRecHits.end(); iRH++ ) {
          //mark "worst" hit:
          
          //float z_at_target ;
          //float radius      ;
          float loc_x_at_target ;
          //float loc_y_at_target ;
          //float loc_z_at_target ;

          //z_at_target  = 0.;
          loc_x_at_target  = 0.;
          //loc_y_at_target  = 0.;
          //loc_z_at_target  = 0.;
          //radius       = 0.;
          
          // set the z target in CMS global coordinates:
          const CSCLayer* csclayerRH = theChamber->layer((*iRH).cscDetId().layer());
          LocalPoint localPositionRH = (*iRH).localPosition();
          GlobalPoint globalPositionRH = csclayerRH->toGlobal(localPositionRH); 
          
          LocalError rerrlocal = (*iRH).localPositionError();  
          float xxerr = rerrlocal.xx();
          
          float target_z     = globalPositionRH.z();  // target z position in cm (z pos of the hit)
          
          loc_x_at_target = localPos.x() + (segDir.x()*( target_z - globZ ));
          //loc_y_at_target = localPos.y() + (segDir.y()*( target_z - globZ ));
          //loc_z_at_target = target_z;

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
          for(size_t bi = 0; bi < buffer.size(); bi++) {
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
      for (size_t iSegment=0; iSegment<reduced_segments.size(); iSegment++ ) {
        // loop over hits on given segment and push pointers to hits into protosegment
        protoSegment.clear();
        for (size_t m = 0; m<reduced_segments[iSegment].size(); ++m ) {
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


/* Method fitSlopes
 *
 * Perform a Least Square Fit on a segment as per SK algo
 *
 */
void CSCSegAlgoHitPruning::fitSlopes() {
  CLHEP::HepMatrix M(4,4,0);
  CLHEP::HepVector B(4,0);
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
    CLHEP::HepMatrix IC(2,2);
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
void CSCSegAlgoHitPruning::fillChiSquared() {
  
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
    
    CLHEP::HepMatrix IC(2,2);
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
void CSCSegAlgoHitPruning::fillLocalDirection() {
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
AlgebraicSymMatrix CSCSegAlgoHitPruning::weightMatrix() const {
  
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
CLHEP::HepMatrix CSCSegAlgoHitPruning::derivativeMatrix() const {
  
  ChamberHitContainer::const_iterator it;
  int nhits = protoSegment.size();
  CLHEP::HepMatrix matrix(2*nhits, 4);
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
AlgebraicSymMatrix CSCSegAlgoHitPruning::calculateError() const {
  
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


void CSCSegAlgoHitPruning::flipErrors( AlgebraicSymMatrix& a ) const { 
    
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

