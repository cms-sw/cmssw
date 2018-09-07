/** \class MTDNavigationSchool
 *
 * Description:
 *  Navigation school for the MTD system
 *  This class defines which DetLayers are reacheable from each MTD DetLayer
 *  (BTL and ETL). The reacheableness is based on an eta range criteria.
 *
 *
 * \author : L. Gray - FNAL
 *
 * Modification:
 * 
 */

#include "RecoMTD/Navigation/interface/MTDNavigationSchool.h"

/* Collaborating Class Header */
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/Navigation/interface/BTLNavigableLayer.h"
#include "RecoMTD/Navigation/interface/ETLNavigableLayer.h"
#include "RecoMTD/Navigation/interface/MTDEtaRange.h"
#include "RecoMTD/Navigation/interface/MTDDetLayerMap.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <iostream>
using namespace std;

/// Constructor
MTDNavigationSchool::MTDNavigationSchool(const MTDDetLayerGeometry * mtdLayout, bool enableBTL, bool enableETL ) : theMTDDetLayerGeometry(mtdLayout) {

  theAllDetLayersInSystem=&mtdLayout->allLayers(); 
  theAllNavigableLayer.resize(mtdLayout->allLayers().size(),nullptr);



  // get all barrel DetLayers (BTL) 
  vector<const DetLayer*> barrel;
  if ( enableBTL ) barrel = mtdLayout->allBarrelLayers();
  else barrel = mtdLayout->allBarrelLayers();

  for ( auto i = barrel.begin(); i != barrel.end(); i++ ) {
    const BarrelDetLayer* mbp = dynamic_cast<const BarrelDetLayer*>(*i);
    if ( mbp == nullptr ) throw cms::Exception("MTDNavigationSchool", "Bad BarrelDetLayer");
    addBarrelLayer(mbp);
  }


  // get all endcap DetLayers (ETL)
  vector<const DetLayer*> endcap;
  if ( enableETL ) endcap = mtdLayout->allEndcapLayers();   
  else endcap = mtdLayout->allEndcapLayers();

  for ( auto i = endcap.begin(); i != endcap.end(); i++ ) {
    const ForwardDetLayer* mep = dynamic_cast<const ForwardDetLayer*>(*i);
    if ( mep == nullptr ) throw cms::Exception("MTDNavigationSchool", "Bad ForwardDetLayer");
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
MTDNavigationSchool::~MTDNavigationSchool() {

   for_each(theBarrelNLC.begin(),theBarrelNLC.end(), delete_layer());
   for_each(theForwardNLC.begin(),theForwardNLC.end(), delete_layer());
   for_each(theBackwardNLC.begin(),theBackwardNLC.end(), delete_layer());

}


/// return all Navigable layers
MTDNavigationSchool::StateType 
MTDNavigationSchool::navigableLayers() {

  StateType result;
  
  vector<BTLNavigableLayer*>::const_iterator ib;
  vector<ETLNavigableLayer*>::const_iterator ie;

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
void MTDNavigationSchool::addBarrelLayer(const BarrelDetLayer* mbp) {

  const BoundCylinder& bc = mbp->specificSurface();
  float radius = bc.radius();
  float length = bc.bounds().length()/2.;

  float eta_max = calculateEta(radius, length);
  float eta_min = -eta_max;

  theBarrelLayers[mbp] = MTDEtaRange(eta_max, eta_min);

}


/// create forwrad/backward layer maps
void MTDNavigationSchool::addEndcapLayer(const ForwardDetLayer* mep) {

  const BoundDisk& bd = mep->specificSurface();
  float outRadius = bd.outerRadius();
  float inRadius = bd.innerRadius();
  float thick = bd.bounds().length()/2.;
  float z = bd.position().z();

  if ( z > 0. ) {
    float eta_min = calculateEta(outRadius, z-thick);
    float eta_max = calculateEta(inRadius, z+thick);
    theForwardLayers[mep] = MTDEtaRange(eta_max, eta_min);
  } else {
    float eta_max = calculateEta(outRadius, z+thick);
    float eta_min = calculateEta(inRadius, z-thick);
    theBackwardLayers[mep] = MTDEtaRange(eta_max, eta_min);
  }

}


/// calculate pseudorapidity from r and z
float MTDNavigationSchool::calculateEta(const float& r, const float& z) const {

  if ( z > 0 ) return -log((tan(atan(r/z)/2.)));
  return log(-(tan(atan(r/z)/2.)));

}

/// linking barrel layers outwards
void MTDNavigationSchool::linkBarrelLayers() {

  for (MapBI bl  = theBarrelLayers.begin();
             bl != theBarrelLayers.end(); bl++) {

    MTDEtaRange range = (*bl).second;

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

    theBarrelNLC.push_back(new BTLNavigableLayer(
                       (*bl).first,outerBarrel, outerBackward, outerForward,
                       allOuterBarrel,allOuterBackward,allOuterForward));

  }

}
/// linking forward/backward layers outwards
void MTDNavigationSchool::linkEndcapLayers(const MapE& layers,
                                            vector<ETLNavigableLayer*>& result) {

  for (MapEI el = layers.begin(); el != layers.end(); el++) {

    MTDEtaRange range = (*el).second;
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
        MTDEtaRange max((*plusOne).second);
        for ( MapEI l = tmpel; l != layers.end(); l++ ) {
          MTDEtaRange next = (*l).second;
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
    
    result.push_back(new ETLNavigableLayer(
                   (*el).first,outerLayers, allOuterLayers));
  }

}


/// create inverse links (i.e. inwards)
void MTDNavigationSchool::createInverseLinks()  {

  // set outward link
  // NavigationSetter setter(*this);

  setState(navigableLayers());


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
    BTLNavigableLayer* mbnl =
      dynamic_cast<BTLNavigableLayer*>(theAllNavigableLayer[((*bli).first)->seqNum()]);
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
      dynamic_cast<ETLNavigableLayer*>(theAllNavigableLayer[((*eli).first)->seqNum()])->getOuterEndcapLayers();
    for (MapEI i = reacheableE.begin(); i != reacheableE.end(); i++ ) {
      reachedForwardLayersMap[(*i).first].insert(*eli);
    }
  // collect all compatible layer starting from a backward layer
    MapE compatibleE =
      dynamic_cast<ETLNavigableLayer*>(theAllNavigableLayer[((*eli).first)->seqNum()])->getAllOuterEndcapLayers();
    for (MapEI i = compatibleE.begin(); i != compatibleE.end(); i++ ) {
      compatibleForwardLayersMap[(*i).first].insert(*eli);
    }
  }

  for ( MapEI eli  = theForwardLayers.begin(); 
              eli != theForwardLayers.end(); eli++ ) {
  // collect all reacheable layer starting from a forward layer
    MapE reacheableE =
      dynamic_cast<ETLNavigableLayer*>(theAllNavigableLayer[((*eli).first)->seqNum()])->getOuterEndcapLayers();
    for (MapEI i = reacheableE.begin(); i != reacheableE.end(); i++ ) {
      reachedForwardLayersMap[(*i).first].insert(*eli);
    }
  // collect all compatible layer starting from a forward layer
    MapE compatibleE =
      dynamic_cast<ETLNavigableLayer*>(theAllNavigableLayer[((*eli).first)->seqNum()])->getAllOuterEndcapLayers();
    for (MapEI i = compatibleE.begin(); i != compatibleE.end(); i++ ) {
      compatibleForwardLayersMap[(*i).first].insert(*eli);
    }
  }

  // now set inverse link for barrel layers
  for ( MapBI bli  = theBarrelLayers.begin(); 
              bli != theBarrelLayers.end(); bli++ ) {
    BTLNavigableLayer* mbnl =
      dynamic_cast<BTLNavigableLayer*>(theAllNavigableLayer[((*bli).first)->seqNum()]);
    mbnl->setInwardLinks(reachedBarrelLayersMap[(*bli).first]);
    mbnl->setInwardCompatibleLinks(compatibleBarrelLayersMap[(*bli).first]);

  }
  //BACKWARD
  for ( MapEI eli  = theBackwardLayers.begin(); 
              eli != theBackwardLayers.end(); eli++ ) {
    ETLNavigableLayer* mfnl =      
      dynamic_cast<ETLNavigableLayer*>(theAllNavigableLayer[((*eli).first)->seqNum()]);
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
    ETLNavigableLayer* mfnl = 
      dynamic_cast<ETLNavigableLayer*>(theAllNavigableLayer[((*eli).first)->seqNum()]);
    // and for forward next layers
    mfnl->setInwardLinks(reachedBarrelLayersMap[(*eli).first],
                         reachedForwardLayersMap[(*eli).first]);
  // and for forward compatible layers
    mfnl->setInwardCompatibleLinks(compatibleBarrelLayersMap[(*eli).first],
                         compatibleForwardLayersMap[(*eli).first]);
  }

}
