/**
 * \file CSCSegAlgoShowering.cc
 *
 *  Last update: 17.02.2015
 *
 */

#include "RecoLocalMuon/CSCSegment/src/CSCSegAlgoShowering.h"
#include "RecoLocalMuon/CSCSegment/src/CSCSegFit.h"

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
CSCSegAlgoShowering::CSCSegAlgoShowering(const edm::ParameterSet& ps) : sfit_(0) {
//  debug                  = ps.getUntrackedParameter<bool>("CSCSegmentDebug");
  dRPhiFineMax           = ps.getParameter<double>("dRPhiFineMax");
  dPhiFineMax            = ps.getParameter<double>("dPhiFineMax");
  tanThetaMax            = ps.getParameter<double>("tanThetaMax");
  tanPhiMax              = ps.getParameter<double>("tanPhiMax");	
  maxRatioResidual       = ps.getParameter<double>("maxRatioResidualPrune");
//  maxDR                  = ps.getParameter<double>("maxDR");
  maxDTheta              = ps.getParameter<double>("maxDTheta");
  maxDPhi                = ps.getParameter<double>("maxDPhi");
}


/* Destructor:
 *
 */
CSCSegAlgoShowering::~CSCSegAlgoShowering(){

}


/* showerSeg
 *
 */
CSCSegment CSCSegAlgoShowering::showerSeg( const CSCChamber* aChamber, const ChamberHitContainer& rechits ) {

  theChamber = aChamber;
  // Initialize parameters
  std::vector<float> x, y, gz;
  std::vector<int> n;
  
  
  for (int i = 0; i < 6; ++i) {
    x.push_back(0.);
    y.push_back(0.);
    gz.push_back(0.);
    n.push_back(0);
  }

  // Loop over hits to find center-of-mass position in each layer
  for (ChamberHitContainer::const_iterator it = rechits.begin(); it != rechits.end(); ++it ) {
    const CSCRecHit2D& hit = (**it);
    const CSCDetId id = hit.cscDetId();
    int l_id = id.layer();
    const CSCLayer* layer  = theChamber->layer(hit.cscDetId().layer());
    GlobalPoint gp         = layer->toGlobal(hit.localPosition());
    LocalPoint  lp         = theChamber->toLocal(gp);

    ++n[l_id -1];
    x[l_id -1] += lp.x();
    y[l_id -1] += lp.y();
    gz[l_id -1] += gp.z();
  }


  // Determine center of mass for each layer and average center of mass for chamber
  float avgChamberX = 0.;
  float avgChamberY = 0.;
  int n_lay = 0;

  for (unsigned i = 0; i < 6; ++i) {
    if (n[i] < 1 ) continue;
 
    x[i] = x[i]/n[i];
    y[i] = y[i]/n[i];
    avgChamberX += x[i];
    avgChamberY += y[i];
    n_lay++;

  }

  if ( n_lay > 0) {
    avgChamberX = avgChamberX / n_lay;
    avgChamberY = avgChamberY / n_lay;
  }

  // Create a FakeSegment origin that points back to the IP
  // Keep all this in global coordinates until last minute to avoid screwing up +/- signs !

  LocalPoint   lpCOM(avgChamberX, avgChamberY, 0.);
  GlobalPoint  gpCOM = theChamber->toGlobal(lpCOM);
  GlobalVector gvCOM(gpCOM.x(), gpCOM.y(), gpCOM.z());

  float Gdxdz = gpCOM.x()/gpCOM.z();
  float Gdydz = gpCOM.y()/gpCOM.z();

  // Figure out the intersection of this vector with each layer of the chamber
  // by projecting the vector
  std::vector<LocalPoint> layerPoints;

  for (size_t i = 0; i!=6; ++i) {
    // Get the layer z coordinates in global frame
    const CSCLayer* layer = theChamber->layer(i+1);
    LocalPoint temp(0., 0., 0.);
    GlobalPoint gp = layer->toGlobal(temp);
    float layer_Z = gp.z();

    // Then compute interesection of vector with that plane
    float layer_X = Gdxdz * layer_Z;
    float layer_Y = Gdydz * layer_Z;
    GlobalPoint Gintersect(layer_X, layer_Y, layer_Z);
    LocalPoint  Lintersect = theChamber->toLocal(Gintersect);

    float layerX = Lintersect.x();
    float layerY = Lintersect.y();
    float layerZ = Lintersect.z();
    LocalPoint layerPoint(layerX, layerY, layerZ);
    layerPoints.push_back(layerPoint);
  }


  std::vector<float> r_closest;
  std::vector<int> id;
  for (size_t i = 0; i!=6; ++i ) {
    id.push_back(-1);
    r_closest.push_back(9999.);
  }

  int idx = 0;

  // Loop over all hits and find hit closest to com for that layer.
  for (ChamberHitContainer::const_iterator it = rechits.begin(); it != rechits.end(); ++it ) {    
    const CSCRecHit2D& hit = (**it);
    int layId = hit.cscDetId().layer();

    const CSCLayer* layer  = theChamber->layer(layId);
    GlobalPoint gp         = layer->toGlobal(hit.localPosition());
    LocalPoint  lp         = theChamber->toLocal(gp);

    float d_x = lp.x() - layerPoints[layId-1].x();
    float d_y = lp.y() - layerPoints[layId-1].y();

    LocalPoint diff(d_x, d_y, 0.);

    if ( fabs(diff.mag() ) < r_closest[layId-1] ) {
       r_closest[layId-1] =  fabs(diff.mag());
       id[layId-1] = idx;
    }
    ++idx;
  }

  // Now fill vector of rechits closest to center of mass:
  protoSegment.clear();
  idx = 0;

  // Loop over all hits and find hit closest to com for that layer.
  for (ChamberHitContainer::const_iterator it = rechits.begin(); it != rechits.end(); ++it ) {    
    const CSCRecHit2D& hit = (**it);
    int layId = hit.cscDetId().layer();

    if ( idx == id[layId-1] )protoSegment.push_back(*it);

    ++idx;    
  }

  // Reorder hits in protosegment
  if ( gz[0] > 0. ) {
    if ( gz[0] > gz[5] ) { 
      reverse( protoSegment.begin(), protoSegment.end() );
    }    
  }
  else if ( gz[0] < 0. ) {
    if ( gz[0] < gz[5] ) {
      reverse( protoSegment.begin(), protoSegment.end() );
    }    
  }

  // Fit the protosegment
  updateParameters();

  // If there is one very bad hit on segment, remove it and refit
  if (protoSegment.size() > 4) pruneFromResidual();

  // If any hit on a layer is closer to segment than original, replace it and refit
  for (ChamberHitContainer::const_iterator it = rechits.begin(); it != rechits.end(); it++ ) {
    const CSCRecHit2D* h = *it;
    int layer = h->cscDetId().layer();
    if ( isHitNearSegment( h ) ) compareProtoSegment( h, layer );
  }

  // Check again for a bad hit, and remove and refit if necessary
  if ( sfit_->nhits() > 5 ) pruneFromResidual( );

  // Does the fitted line point to the IP?
  // If it doesn't, the algorithm has probably failed i.e. that's life!

  GlobalVector protoGlobalDir = theChamber->toGlobal( sfit_->localdir() );  
  double protoTheta = protoGlobalDir.theta();
  double protoPhi = protoGlobalDir.phi();
  double simTheta = gvCOM.theta();
  double simPhi = gvCOM.phi();
  
  float dTheta = fabs(protoTheta - simTheta);
  float dPhi   = fabs(protoPhi - simPhi);
  //  float dR = sqrt(dEta*dEta + dPhi*dPhi);
  
  // Flag the segment with chi2=-1 of the segment isn't pointing toward origin      
  // i.e. flag that the algorithm has probably just failed (I presume we expect
  // a segment to point to the IP if the algorithm is successful!)

  double theFlag = -1.;
  if (dTheta > maxDTheta || dPhi > maxDPhi) {
  }
  else {
    theFlag = sfit_->chi2(); // If it points to IP, just pass fit chi2 as usual
  }

  // Create an actual CSCSegment - retrieve all info from the fit
  CSCSegment temp(sfit_->hits(), sfit_->intercept(), 
  		  sfit_->localdir(), sfit_->covarianceMatrix(), theFlag );
  delete sfit_;
  sfit_ = 0;

  return temp;
} 




/* isHitNearSegment
 *
 * Compare rechit with expected position from proto_segment
 */
bool CSCSegAlgoShowering::isHitNearSegment( const CSCRecHit2D* hit ) const {

  const CSCLayer* layer = theChamber->layer(hit->cscDetId().layer());

  // hit phi position in global coordinates
  GlobalPoint Hgp = layer->toGlobal(hit->localPosition());
  double Hphi = Hgp.phi();                                
  if (Hphi < 0.) Hphi += 2.*M_PI;
  LocalPoint Hlp = theChamber->toLocal(Hgp);
  double z = Hlp.z();  

  double LocalX = sfit_->xfit(z);
  double LocalY = sfit_->yfit(z);
  LocalPoint Slp(LocalX, LocalY, z);
  GlobalPoint Sgp = theChamber->toGlobal(Slp); 
  double Sphi = Sgp.phi();
  if (Sphi < 0.) Sphi += 2.*M_PI;
  double R = sqrt(Sgp.x()*Sgp.x() + Sgp.y()*Sgp.y());
  
  double deltaPhi = Sphi - Hphi;
  if (deltaPhi >  2.*M_PI) deltaPhi -= 2.*M_PI;
  if (deltaPhi < -2.*M_PI) deltaPhi += 2.*M_PI;
  if (deltaPhi < 0.) deltaPhi = -deltaPhi; 

  double RdeltaPhi = R * deltaPhi;

  if (RdeltaPhi < dRPhiFineMax && deltaPhi < dPhiFineMax ) return true;

  return false;
}


/* Method addHit
 *
 * Test if can add hit to proto segment. If so, try to add it.
 *
 */
bool CSCSegAlgoShowering::addHit(const CSCRecHit2D* aHit, int layer) {
  
  // Return true if hit was added successfully
  // Return false if there is already a hit on the same layer
  
  bool ok = true;
  
  // Test that we are not trying to add the same hit again
  for ( ChamberHitContainer::const_iterator it = protoSegment.begin(); it != protoSegment.end(); it++ ) 
    if ( aHit == (*it)  ) return false;
  
  protoSegment.push_back(aHit);

  return ok;
}    


/* Method compareProtoSegment
 *      
 * For hit coming from the same layer of an existing hit within the proto segment
 * test if achieve better chi^2 by using this hit than the other
 *
 */ 
void CSCSegAlgoShowering::compareProtoSegment(const CSCRecHit2D* h, int layer) {
  
  // Try adding the hit to existing segment, and removing one from the  same layer
  ChamberHitContainer::iterator it;
  for ( it = protoSegment.begin(); it != protoSegment.end(); ) {
    if ( (*it)->cscDetId().layer() == layer ) {
      it = protoSegment.erase(it);
    } else {
      ++it;
    }
  }
  bool ok = addHit(h, layer);
  if (ok) {
    CSCSegFit* newfit = new CSCSegFit(theChamber, protoSegment);
    newfit->fit();
    if ( newfit->chi2() > sfit_->chi2() ) {
      // new fit is worse: revert to old fit
      delete newfit;
    }
    else {
      // new fit is better
      delete sfit_;
      sfit_ = newfit;
    }
  }
}


// Look for a hit with a large deviation from fit

void CSCSegAlgoShowering::pruneFromResidual(void){

  //@@ THIS FUNCTION HAS 3 RETURNS PATHS!

  // Only prune if have at least 5 hits 
  if ( protoSegment.size() < 5 ) return;

  // Now Study residuals
      
  float maxResidual = 0.;
  float sumResidual = 0.;
  int nHits = 0;

  ChamberHitContainer::iterator ih;
  ChamberHitContainer::iterator ibad = protoSegment.end();

  for ( ih = protoSegment.begin(); ih != protoSegment.end(); ++ih ) {
    const CSCRecHit2D& hit = (**ih);
    const CSCLayer* layer  = theChamber->layer(hit.cscDetId().layer());
    GlobalPoint gp         = layer->toGlobal(hit.localPosition());
    LocalPoint lp          = theChamber->toLocal(gp);

    double u = lp.x();
    double v = lp.y();
    double z = lp.z();

    //    double du = segfit->xdev( u, z ); 
    //    double dv = segfit->ydev( v, z );

    float residual = sfit_->Rdev(u, v, z); // == sqrt(du*du + dv*dv)

    sumResidual += residual;
    ++nHits;
    if ( residual > maxResidual ) {
      maxResidual = residual;
      ibad = ih;
    }
  }

  float corrAvgResidual = (sumResidual - maxResidual)/(nHits -1);

  // Keep all hits 
  if ( maxResidual/corrAvgResidual < maxRatioResidual ) return;

  // Drop worst hit
  if( ibad != protoSegment.end() ) protoSegment.erase(ibad);

  // Make a new fit
  updateParameters();

  return;
}

void CSCSegAlgoShowering::updateParameters(void) {
  // Create fit for the hits in the protosegment & run it
  delete sfit_;
  sfit_ = new CSCSegFit( theChamber, protoSegment );
  sfit_->fit();
}





