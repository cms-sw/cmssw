/** \class MuonNavigationSchool
 *
 * Description:
 *  Navigation school for the muon system
 *  This class defines which DetLayers are reacheable from each Muon DetLayer
 *  (DT, CSC and RPC). The reacheableness is based on an eta range criteria.
 *
 * $Date: 2013/02/23 09:08:03 $
 * $Revision: 1.13 $
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 * Chang Liu:
 * The class links maps for nextLayers and compatibleLayers in the same time.
 *
 */

#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"

/* Collaborating Class Header */
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Navigation/interface/MuonBarrelNavigableLayer.h"
#include "RecoMuon/Navigation/interface/MuonForwardNavigableLayer.h"
#include "RecoMuon/Navigation/interface/MuonEtaRange.h"
#include "RecoMuon/Navigation/interface/MuonDetLayerMap.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <iostream>
using namespace std;

/// Constructor
MuonNavigationSchool::MuonNavigationSchool(const MuonDetLayerGeometry * muonLayout, bool enableRPC ) : theMuonDetLayerGeometry(muonLayout) {

  theAllDetLayersInSystem=&muonLayout->allLayers(); 

  // get all barrel DetLayers (DT + optional RPC) 
  vector<DetLayer*> barrel;
  if ( enableRPC ) barrel = muonLayout->allBarrelLayers();
  else barrel = muonLayout->allDTLayers();

  for ( vector<DetLayer*>::const_iterator i = barrel.begin(); i != barrel.end(); i++ ) {
    BarrelDetLayer* mbp = dynamic_cast<BarrelDetLayer*>(*i);
    if ( mbp == 0 ) throw cms::Exception("MuonNavigationSchool", "Bad BarrelDetLayer");
    addBarrelLayer(mbp);
  }

  // get all endcap DetLayers (CSC + optional RPC)
  vector<DetLayer*> endcap;
  if ( enableRPC ) endcap = muonLayout->allEndcapLayers();
  else endcap = muonLayout->allCSCLayers();

  for ( vector<DetLayer*>::const_iterator i = endcap.begin(); i != endcap.end(); i++ ) {
    ForwardDetLayer* mep = dynamic_cast<ForwardDetLayer*>(*i);
    if ( mep == 0 ) throw cms::Exception("MuonNavigationSchool", "Bad ForwardDetLayer");
    addEndcapLayer(mep);
  }

  // create outward links for all DetLayers
  linkBarrelLayers();
  linkEndcapLayers(theForwardLayers,theForwardNLC);
  linkEndcapLayers(theBackwardLayers,theBackwardNLC);

  // create inverse links
  createInverseLinks();

}


/// Destructor
MuonNavigationSchool::~MuonNavigationSchool() {

   for_each(theBarrelNLC.begin(),theBarrelNLC.end(), delete_layer());
   for_each(theForwardNLC.begin(),theForwardNLC.end(), delete_layer());
   for_each(theBackwardNLC.begin(),theBackwardNLC.end(), delete_layer());

}


/// return all Navigable layers
MuonNavigationSchool::StateType 
MuonNavigationSchool::navigableLayers() const {

  StateType result;
  
  vector<MuonBarrelNavigableLayer*>::const_iterator ib;
  vector<MuonForwardNavigableLayer*>::const_iterator ie;

  for ( ib = theBarrelNLC.begin(); ib != theBarrelNLC.end(); ib++ ) {
    result.push_back(*ib);
  }

  for ( ie = theForwardNLC.begin(); ie != theForwardNLC.end(); ie++ ) {
    result.push_back(*ie);
  }

  for ( ie = theBackwardNLC.begin(); ie != theBackwardNLC.end(); ie++ ) {
    result.push_back(*ie);
  }
  
  return result;

}


/// create barrel layer map
void MuonNavigationSchool::addBarrelLayer(BarrelDetLayer* mbp) {

  const BoundCylinder& bc = mbp->specificSurface();
  float radius = bc.radius();
  float length = bc.bounds().length()/2.;

  float eta_max = calculateEta(radius, length);
  float eta_min = -eta_max;

  theBarrelLayers[mbp] = MuonEtaRange(eta_max, eta_min);

}


/// create forwrad/backward layer maps
void MuonNavigationSchool::addEndcapLayer(ForwardDetLayer* mep) {

  const BoundDisk& bd = mep->specificSurface();
  float outRadius = bd.outerRadius();
  float inRadius = bd.innerRadius();
  float thick = bd.bounds().length()/2.;
  float z = bd.position().z();

  if ( z > 0. ) {
    float eta_min = calculateEta(outRadius, z-thick);
    float eta_max = calculateEta(inRadius, z+thick);
    theForwardLayers[mep] = MuonEtaRange(eta_max, eta_min);
  } else {
    float eta_max = calculateEta(outRadius, z+thick);
    float eta_min = calculateEta(inRadius, z-thick);
    theBackwardLayers[mep] = MuonEtaRange(eta_max, eta_min);
  }

}


/// calculate pseudorapidity from r and z
float MuonNavigationSchool::calculateEta(const float& r, const float& z) const {

  if ( z > 0 ) return -log((tan(atan(r/z)/2.)));
  return log(-(tan(atan(r/z)/2.)));

}

/// linking barrel layers outwards
void MuonNavigationSchool::linkBarrelLayers() {

  for (MapBI bl  = theBarrelLayers.begin();
             bl != theBarrelLayers.end(); bl++) {

    MuonEtaRange range = (*bl).second;

    // first add next barrel layer
    MapBI plusOne(bl);
    plusOne++;
    MapB outerBarrel;
    MapB allOuterBarrel;
    if ( plusOne != theBarrelLayers.end() ) { outerBarrel.insert(*plusOne);}
    // add all outer barrel layers
    for ( MapBI iMBI = plusOne; iMBI!= theBarrelLayers.end(); iMBI++){
      allOuterBarrel.insert(*iMBI);
    }
    // then add all compatible backward layers with an eta criteria
    MapE allOuterBackward;
    for (MapEI el  = theBackwardLayers.begin();
               el != theBackwardLayers.end(); el++) {
      if ( (*el).second.isCompatible(range) ) {
        allOuterBackward.insert(*el);
      }
    }
    //add the backward next layer with an eta criteria
    MapE outerBackward;
    for (MapEI el  = theBackwardLayers.begin();
               el != theBackwardLayers.end(); el++) {
      if ( (*el).second.isCompatible(range) ) {
        outerBackward.insert(*el);
        break;
      }
    }

    // then add all compatible forward layers with an eta criteria
    MapE allOuterForward;
    for (MapEI el  = theForwardLayers.begin();
               el != theForwardLayers.end(); el++) {
      if ( (*el).second.isCompatible(range) ) {
        allOuterForward.insert(*el);
      }
    }

    // then add forward next layer with an eta criteria
    MapE outerForward;
    for (MapEI el  = theForwardLayers.begin();
               el != theForwardLayers.end(); el++) {
      if ( (*el).second.isCompatible(range) ) {
        outerForward.insert(*el);
        break;
      }
    }

    theBarrelNLC.push_back(new MuonBarrelNavigableLayer(
                       (*bl).first,outerBarrel, outerBackward, outerForward,
                       allOuterBarrel,allOuterBackward,allOuterForward));

  }

}
/// linking forward/backward layers outwards
void MuonNavigationSchool::linkEndcapLayers(const MapE& layers,
                                            vector<MuonForwardNavigableLayer*>& result) {

  for (MapEI el = layers.begin(); el != layers.end(); el++) {

    MuonEtaRange range = (*el).second;
    // first add next endcap layer (if compatible)
    MapEI plusOne(el); 
    plusOne++;
    MapE outerLayers;
    if ( plusOne != layers.end() && (*plusOne).second.isCompatible(range) ) {
        outerLayers.insert(*plusOne);
      if ( !range.isInside((*plusOne).second) ) {
        // then look if the next layer has a wider eta range, if so add it
        MapEI tmpel(plusOne);
        tmpel++;
        MuonEtaRange max((*plusOne).second);
        for ( MapEI l = tmpel; l != layers.end(); l++ ) {
          MuonEtaRange next = (*l).second;
          if ( next.isCompatible(max) && !range.isInside(next) &&
               !next.isInside(max) && next.subtract(max).isInside(range) ) {
            max = max.add(next);
            outerLayers.insert(*l);
          }
        }
      }
    }

    MapE allOuterLayers;
    for (MapEI iMEI = plusOne; iMEI!=layers.end(); iMEI++){
      if ((*iMEI).second.isCompatible(range)) allOuterLayers.insert(*iMEI);
    }
    
    result.push_back(new MuonForwardNavigableLayer(
                   (*el).first,outerLayers, allOuterLayers));
  }

}


/// create inverse links (i.e. inwards)
void MuonNavigationSchool::createInverseLinks() const {

  // set outward link
  NavigationSetter setter(*this);

  // find for each layer which are the layers pointing to it
  typedef map<const DetLayer*, MapB, less<const DetLayer*> > BarrelMapType;
  typedef map<const DetLayer*, MapE, less<const DetLayer*> > ForwardMapType;

  // map of all DetLayers which can reach a specific DetLayer
  BarrelMapType reachedBarrelLayersMap;
  ForwardMapType reachedForwardLayersMap;

  // map of all DetLayers which is compatible with a specific DetLayer
  BarrelMapType compatibleBarrelLayersMap;
  ForwardMapType compatibleForwardLayersMap;

  // collect all reacheable layers starting from a barrel layer
  for ( MapBI bli  = theBarrelLayers.begin(); 
              bli != theBarrelLayers.end(); bli++ ) {
    // barrel
    MuonBarrelNavigableLayer* mbnl =
      dynamic_cast<MuonBarrelNavigableLayer*>(((*bli).first)->navigableLayer());
    MapB reacheableB = mbnl->getOuterBarrelLayers();
    for (MapBI i = reacheableB.begin(); i != reacheableB.end(); i++ ) {
      reachedBarrelLayersMap[(*i).first].insert(*bli);
    }
    MapB compatibleB = mbnl->getAllOuterBarrelLayers();
    for (MapBI i = compatibleB.begin(); i != compatibleB.end(); i++ ) {
      compatibleBarrelLayersMap[(*i).first].insert(*bli);
    }
    MapE reacheableE = mbnl->getOuterBackwardLayers();
    for (MapEI i = reacheableE.begin(); i != reacheableE.end(); i++ ) {
      reachedBarrelLayersMap[(*i).first].insert(*bli);
    }
    reacheableE = mbnl->getOuterForwardLayers();
    for (MapEI i = reacheableE.begin(); i != reacheableE.end(); i++ ) {
      reachedBarrelLayersMap[(*i).first].insert(*bli);
    }
    MapE compatibleE = mbnl->getAllOuterBackwardLayers();
    for (MapEI i = compatibleE.begin(); i != compatibleE.end(); i++ ) {
      compatibleBarrelLayersMap[(*i).first].insert(*bli);
    }
    compatibleE = mbnl->getAllOuterForwardLayers();
    for (MapEI i = compatibleE.begin(); i != compatibleE.end(); i++ ) {
      compatibleBarrelLayersMap[(*i).first].insert(*bli);
    }

  }

  // collect all reacheable layer starting from a backward layer
  for ( MapEI eli  = theBackwardLayers.begin(); 
              eli != theBackwardLayers.end(); eli++ ) {
    MapE reacheableE =
      dynamic_cast<MuonForwardNavigableLayer*>(((*eli).first)->navigableLayer())->getOuterEndcapLayers();
    for (MapEI i = reacheableE.begin(); i != reacheableE.end(); i++ ) {
      reachedForwardLayersMap[(*i).first].insert(*eli);
    }
  // collect all compatible layer starting from a backward layer
    MapE compatibleE =
      dynamic_cast<MuonForwardNavigableLayer*>(((*eli).first)->navigableLayer())->getAllOuterEndcapLayers();
    for (MapEI i = compatibleE.begin(); i != compatibleE.end(); i++ ) {
      compatibleForwardLayersMap[(*i).first].insert(*eli);
    }
  }

  for ( MapEI eli  = theForwardLayers.begin(); 
              eli != theForwardLayers.end(); eli++ ) {
  // collect all reacheable layer starting from a forward layer
    MapE reacheableE =
      dynamic_cast<MuonForwardNavigableLayer*>(((*eli).first)->navigableLayer())->getOuterEndcapLayers();
    for (MapEI i = reacheableE.begin(); i != reacheableE.end(); i++ ) {
      reachedForwardLayersMap[(*i).first].insert(*eli);
    }
  // collect all compatible layer starting from a forward layer
    MapE compatibleE =
      dynamic_cast<MuonForwardNavigableLayer*>(((*eli).first)->navigableLayer())->getAllOuterEndcapLayers();
    for (MapEI i = compatibleE.begin(); i != compatibleE.end(); i++ ) {
      compatibleForwardLayersMap[(*i).first].insert(*eli);
    }
  }

  // now set inverse link for barrel layers
  for ( MapBI bli  = theBarrelLayers.begin(); 
              bli != theBarrelLayers.end(); bli++ ) {
    MuonBarrelNavigableLayer* mbnl =
      dynamic_cast<MuonBarrelNavigableLayer*>(((*bli).first)->navigableLayer());
    mbnl->setInwardLinks(reachedBarrelLayersMap[(*bli).first]);
    mbnl->setInwardCompatibleLinks(compatibleBarrelLayersMap[(*bli).first]);

  }
  //BACKWARD
  for ( MapEI eli  = theBackwardLayers.begin(); 
              eli != theBackwardLayers.end(); eli++ ) {
    MuonForwardNavigableLayer* mfnl =      
      dynamic_cast<MuonForwardNavigableLayer*>(((*eli).first)->navigableLayer());
    // for backward next layers
    mfnl->setInwardLinks(reachedBarrelLayersMap[(*eli).first],
                         reachedForwardLayersMap[(*eli).first]);
  // for backward compatible layers
    mfnl->setInwardCompatibleLinks(compatibleBarrelLayersMap[(*eli).first],
                         compatibleForwardLayersMap[(*eli).first]);
  }
  //FORWARD
  for ( MapEI eli  = theForwardLayers.begin(); 
              eli != theForwardLayers.end(); eli++ ) {
    MuonForwardNavigableLayer* mfnl = 
      dynamic_cast<MuonForwardNavigableLayer*>(((*eli).first)->navigableLayer());
  // and for forward next layers
    mfnl->setInwardLinks(reachedBarrelLayersMap[(*eli).first],
                         reachedForwardLayersMap[(*eli).first]);
  // and for forward compatible layers
    mfnl->setInwardCompatibleLinks(compatibleBarrelLayersMap[(*eli).first],
                         compatibleForwardLayersMap[(*eli).first]);
  }

}
