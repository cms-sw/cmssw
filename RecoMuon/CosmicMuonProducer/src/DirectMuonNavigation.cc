#include "RecoMuon/CosmicMuonProducer/interface/DirectMuonNavigation.h"
/** \file DirectMuonNavigation
 *
 *  $Date:  $
 *  $Revision:  $
 *  \author Chang Liu  -  Purdue University
 */

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "Utilities/General/interface/CMSexception.h"
#include "TrackingTools/DetLayers/interface/Enumerators.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"


using namespace std;

DirectMuonNavigation::DirectMuonNavigation(const MuonDetLayerGeometry * muonLayout) : theMuonDetLayerGeometry(muonLayout) {
   epsilon_ = 10.;
  // get all barrel DetLayers (DT + RPC)
  vector<DetLayer*> barrel = muonLayout->allBarrelLayers();
  for ( vector<DetLayer*>::const_iterator i = barrel.begin(); i != barrel.end(); i++ ) {
    BarrelDetLayer* mbp = dynamic_cast<BarrelDetLayer*>(*i);
    if ( mbp == 0 ) throw Genexception("Bad BarrelDetLayer");
    addBarrelLayer(mbp);
  }
                                                                                
  // get all endcap DetLayers (CSC + RPC)
  vector<DetLayer*> csc = muonLayout->allEndcapLayers();
  for ( vector<DetLayer*>::const_iterator i = csc.begin(); i != csc.end(); i++ ) {
    ForwardDetLayer* mep = dynamic_cast<ForwardDetLayer*>(*i);
    if ( mep == 0 ) throw Genexception("Bad ForwardDetLayer");
    addEndcapLayer(mep);
  }

}
/* Operations */ 
vector<const DetLayer*> 
DirectMuonNavigation::compatibleLayers( const FreeTrajectoryState& fts,
                                        PropagationDirection dir ) const {

  float z0 = fts.position().z();

  vector<const DetLayer*> output;
  vector<const ForwardDetLayer*>::const_iterator iter_E;
  vector<const BarrelDetLayer*>::const_iterator iter_B;
  bool cont = false;
  // add barrel first 
  for (iter_B = theBarrelLayers.begin(); iter_B != theBarrelLayers.end() ;
       iter_B++){
      if( cont ) output.push_back((*iter_B));
      else if ( checkCompatible(fts,(*iter_B))) {
      output.push_back((*iter_B));
      cont = true;
    }
  }

  cont = false;

  if ( z0 > 0. ) {
    for (iter_E = theForwardLayers.begin(); iter_E != theForwardLayers.end() ; 
	 iter_E++){
      if( cont ) output.push_back((*iter_E));
      else if ( checkCompatible(fts,(*iter_E))) {
	output.push_back((*iter_E));
	cont = true;
      }
    }
  } else {
    for (iter_E = theBackwardLayers.begin(); iter_E != theBackwardLayers.end() ; 
	 iter_E++){
      if( cont ) output.push_back((*iter_E));
      else if ( checkCompatible(fts,(*iter_E))) {
	output.push_back((*iter_E));
	cont = true;
      }
    }
  }

  return output;
}


void DirectMuonNavigation::addBarrelLayer(BarrelDetLayer* mbp) {

  theBarrelLayers.push_back(mbp);
}


void DirectMuonNavigation::addEndcapLayer(ForwardDetLayer* mep) {

  BoundDisk* bd = dynamic_cast<BoundDisk*>(const_cast<BoundSurface*>(&(mep->surface())));
  float z = bd->position().z();

  if ( z > 0. ) {
    theForwardLayers.push_back(mep);
  } else {
    theBackwardLayers.push_back(mep);
  }

}

bool DirectMuonNavigation::checkCompatible(const FreeTrajectoryState& fts,const BarrelDetLayer* dl) const {

  float z0 = fts.position().z();
  float r0 = fts.position().perp();
  float zm = fts.momentum().z();
  float rm = fts.momentum().perp();
  float slope = zm/rm; 
  if (!outward(fts) ) slope = -slope;

  const BoundCylinder* bc = dynamic_cast<const BoundCylinder*>(&dl->surface());

  float radius = bc->radius();
  float length = bc->bounds().length()/2.;

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

  const BoundDisk* bd = dynamic_cast<const BoundDisk*>(&dl->surface());

  float outRadius = bd->outerRadius();
  float inRadius = bd->innerRadius();
  float z = bd->position().z();

  float r1 = slope*(z - z0) + r0;
  return (r1 >= inRadius-epsilon_ && r1 <= outRadius+epsilon_);

}

bool DirectMuonNavigation::outward(const FreeTrajectoryState& fts) const {
  float x0 = fts.position().x();
  float y0 = fts.position().y();
  float r0 = fts.position().perp();

  float xm = fts.momentum().x();
  float ym = fts.momentum().y();

  float delta = 0.01;

  float x1 = x0 + xm * delta;
  float y1 = y0 + ym * delta;

  float r1 = sqrt(x1*x1+y1*y1);

  return (r1 >= r0);
}


