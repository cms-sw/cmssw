#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"

/** \file DirectMuonNavigation
 *
 *  $Date: 2011/04/21 01:34:24 $
 *  $Revision: 1.19 $
 *  \author Chang Liu  -  Purdue University
 */

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <algorithm>

using namespace std;

DirectMuonNavigation::DirectMuonNavigation(const edm::ESHandle<MuonDetLayerGeometry>& muonLayout) : theMuonDetLayerGeometry(muonLayout), epsilon_(100.), theEndcapFlag(true), theBarrelFlag(true) {
}

DirectMuonNavigation::DirectMuonNavigation(const edm::ESHandle<MuonDetLayerGeometry>& muonLayout, const edm::ParameterSet& par) : theMuonDetLayerGeometry(muonLayout), epsilon_(100.), theEndcapFlag(par.getParameter<bool>("Endcap")), theBarrelFlag(par.getParameter<bool>("Barrel")) {

}


/* return compatible layers for given trajectory state  */ 
vector<const DetLayer*> 
DirectMuonNavigation::compatibleLayers( const FreeTrajectoryState& fts,
                                        PropagationDirection dir ) const {

  float z0 = fts.position().z();
  float zm = fts.momentum().z();

  bool inOut = outward(fts);

  vector<const DetLayer*> output;

  // check direction and position of FTS to get a correct order of DetLayers

  if (inOut) { 
     if ((zm * z0) >= 0) {
        if (theBarrelFlag) inOutBarrel(fts,output);
        if (theEndcapFlag) {
           if ( z0 >= 0 ) inOutForward(fts,output);
           else inOutBackward(fts,output);
        } 
      } else {
        if (theEndcapFlag) {
        if ( z0 >= 0 ) outInForward(fts,output);
           else outInBackward(fts,output);
        }
        if (theBarrelFlag) inOutBarrel(fts,output);
      } 
   } else {
     if ((zm * z0) >= 0) {
        if (theBarrelFlag) outInBarrel(fts,output);
        if (theEndcapFlag) {
          if ( z0 >= 0 ) inOutForward(fts,output);
          else inOutBackward(fts,output);
        }
      } else {
        if (theEndcapFlag) {
          if ( z0 >= 0 ) outInForward(fts,output);
          else outInBackward(fts,output);
        }
        if (theBarrelFlag) outInBarrel(fts,output);
      } 
   }

  if ( dir == oppositeToMomentum ) std::reverse(output.begin(),output.end());

  return output;
}

/*
return compatible endcap layers on BOTH ends;
used for beam-halo muons
*/
vector<const DetLayer*> 
DirectMuonNavigation::compatibleEndcapLayers( const FreeTrajectoryState& fts,
                                              PropagationDirection dir ) const {

  float zm = fts.momentum().z();

  vector<const DetLayer*> output;

  // collect all endcap layers on 2 sides
  outInBackward(fts,output);
  inOutForward(fts,output);

  // check direction FTS to get a correct order of DetLayers
  if ( ( zm > 0 && dir == oppositeToMomentum ) || 
       ( zm < 0 && dir == alongMomentum )  )
           std::reverse(output.begin(),output.end());

  return output;
}

void DirectMuonNavigation::inOutBarrel(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {

  bool cont = false;
  const vector<DetLayer*>& barrel = theMuonDetLayerGeometry->allBarrelLayers();

  for (vector<DetLayer*>::const_iterator iter_B = barrel.begin(); iter_B != barrel.end(); iter_B++){

      if( cont ) output.push_back((*iter_B));
      else if ( checkCompatible(fts,dynamic_cast<const BarrelDetLayer*>(*iter_B))) {
      output.push_back((*iter_B));
      cont = true;
      }
  }
}


void DirectMuonNavigation::outInBarrel(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {

// default barrel layers are in out, reverse order 
  const vector<DetLayer*>& barrel = theMuonDetLayerGeometry->allBarrelLayers();

  bool cont = false;
  vector<DetLayer*>::const_iterator rbegin = barrel.end(); 
  rbegin--;
  vector<DetLayer*>::const_iterator rend = barrel.begin();
  rend--;

  for (vector<DetLayer*>::const_iterator iter_B = rbegin; iter_B != rend; iter_B--){
      if( cont ) output.push_back((*iter_B));
      else if ( checkCompatible(fts,dynamic_cast<BarrelDetLayer*>(*iter_B))) {
      output.push_back((*iter_B));
      cont = true;
      }
  }
}

void DirectMuonNavigation::inOutForward(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {

  const vector<DetLayer*>& forward = theMuonDetLayerGeometry->allForwardLayers();
  bool cont = false;
  for (vector<DetLayer*>::const_iterator iter_E = forward.begin(); iter_E != forward.end(); 
	 iter_E++){
      if( cont ) output.push_back((*iter_E));
      else if ( checkCompatible(fts,dynamic_cast<ForwardDetLayer*>(*iter_E))) {
	output.push_back((*iter_E));
	cont = true;
      }
    }
}

void DirectMuonNavigation::outInForward(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {
// default forward layers are in out, reverse order

  bool cont = false;
  const vector<DetLayer*>& forward = theMuonDetLayerGeometry->allForwardLayers();
  vector<DetLayer*>::const_iterator rbegin = forward.end();
  rbegin--;
  vector<DetLayer*>::const_iterator rend = forward.begin();
  rend--;
  for (vector<DetLayer*>::const_iterator iter_E = rbegin; iter_E != rend;
         iter_E--){
      if( cont ) output.push_back((*iter_E));
      else if ( checkCompatible(fts,dynamic_cast<ForwardDetLayer*>(*iter_E))) {
        output.push_back((*iter_E));
        cont = true;
      }
    }
}

void DirectMuonNavigation::inOutBackward(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {
  bool cont = false;
  const vector<DetLayer*>& backward = theMuonDetLayerGeometry->allBackwardLayers();

  for (vector<DetLayer*>::const_iterator iter_E = backward.begin(); iter_E != backward.end(); 
       iter_E++){
      if( cont ) output.push_back((*iter_E));
      else if ( checkCompatible(fts,dynamic_cast<ForwardDetLayer*>(*iter_E))) {
	output.push_back((*iter_E));
	cont = true;
      }
   }
}

void DirectMuonNavigation::outInBackward(const FreeTrajectoryState& fts, vector<const DetLayer*>& output) const {

  bool cont = false;
  const vector<DetLayer*>& backward = theMuonDetLayerGeometry->allBackwardLayers();

  vector<DetLayer*>::const_iterator rbegin = backward.end();
  rbegin--;
  vector<DetLayer*>::const_iterator rend = backward.begin();
  rend--;
  for (vector<DetLayer*>::const_iterator iter_E = rbegin; iter_E != rend;
       iter_E--){
      if( cont ) output.push_back((*iter_E));
      else if ( checkCompatible(fts,dynamic_cast<ForwardDetLayer*>(*iter_E))) {
        output.push_back((*iter_E));
        cont = true;
      }
   }

}


bool DirectMuonNavigation::checkCompatible(const FreeTrajectoryState& fts,const BarrelDetLayer* dl) const {

  float z0 = fts.position().z();
  float r0 = fts.position().perp();
  float zm = fts.momentum().z();
  float rm = fts.momentum().perp();
  float slope = zm/rm; 
  if (!outward(fts) ) slope = -slope;
  const BoundCylinder bc = dl->specificSurface();
  float radius = bc.radius();
  float length = bc.bounds().length()/2.;

  float z1 = slope*(radius - r0) + z0;
  return ( fabs(z1) <= fabs(length)+epsilon_ );

}

bool DirectMuonNavigation::checkCompatible(const FreeTrajectoryState& fts,const ForwardDetLayer* dl) const {

  float z0 = fts.position().z();
  float r0 = fts.position().perp();
  float zm = fts.momentum().z();
  float rm = fts.momentum().perp();
  float slope = rm/zm; 

  if (!outward(fts) ) slope = -slope;

  const BoundDisk bd = dl->specificSurface();

  float outRadius = bd.outerRadius();
  float inRadius = bd.innerRadius();
  float z = bd.position().z();

  float r1 = slope*(z - z0) + r0;
  return (r1 >= inRadius-epsilon_ && r1 <= outRadius+epsilon_);

}

bool DirectMuonNavigation::outward(const FreeTrajectoryState& fts) const {
 
//  return (fts.position().basicVector().dot(fts.momentum().basicVector())>0);

  float x0 = fts.position().x();
  float y0 = fts.position().y();

  float xm = fts.momentum().x();
  float ym = fts.momentum().y();

  return ((x0 * xm + y0 * ym ) > 0);


}
