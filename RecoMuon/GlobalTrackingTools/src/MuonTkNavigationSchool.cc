/** 
 *  Class:  MuonTkNavigationSchool
 *
 *  Navigation School for both the Muon system and
 *  the Tracker.
 * 
 *
 *  $Date: 2008/02/05 16:56:23 $
 *  $Revision: 1.2 $
 *
 * \author : Chang Liu - Purdue University
 * \author : Stefano Lacaprara - INFN Padova
 *
 *
 */

#include "RecoMuon/GlobalTrackingTools/interface/MuonTkNavigationSchool.h"

//---------------
// C++ Headers --
//---------------

#include <functional>
#include <algorithm>
#include <map>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "RecoTracker/TkNavigation/interface/SimpleBarrelNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/SimpleForwardNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/DiskLessInnerRadius.h"
#include "RecoTracker/TkNavigation/interface/SymmetricLayerFinder.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "TrackingTools/DetLayers/src/DetBelowZ.h"
#include "TrackingTools/DetLayers/src/DetLessZ.h"
#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"
#include "RecoMuon/Navigation/interface/MuonBarrelNavigableLayer.h"
#include "RecoMuon/Navigation/interface/MuonForwardNavigableLayer.h"
#include "RecoMuon/Navigation/interface/MuonDetLayerMap.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "Utilities/General/interface/CMSexception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

typedef std::vector<DetLayer*> LayerContainer;

//
// constructor
//
MuonTkNavigationSchool::MuonTkNavigationSchool(const MuonDetLayerGeometry* muonGeom, 
                                               const GeometricSearchTracker* trackerGeom, 
                                               const MagneticField* field) : 
   theMuonDetLayerGeometry(muonGeom), theGeometricSearchTracker(trackerGeom), theMagneticField(field) {

  // need to allocate the vector of DetLayers, to concatenate the two vectors of DetLayers
  // it has to be deleted in the destructor
  std::vector<DetLayer*> * allLayers = new std::vector<DetLayer*>();
  allLayers->reserve(muonGeom->allLayers().size()+trackerGeom->allLayers().size());
  allLayers->insert(allLayers->end(), muonGeom->allLayers().begin(), muonGeom->allLayers().end());
  allLayers->insert(allLayers->end(), trackerGeom->allLayers().begin(), trackerGeom->allLayers().end());
  theAllDetLayersInSystem = allLayers;
  
  // get tracker barrel layers
  std::vector<BarrelDetLayer*> blc = trackerGeom->barrelLayers();
  for ( std::vector<BarrelDetLayer*>::const_iterator i = blc.begin(); i != blc.end(); i++ ) {
     addBarrelLayer(*i);
  }

  // get tracker forward layers
  std::vector<ForwardDetLayer*> flc = trackerGeom->forwardLayers();
  for (std::vector<ForwardDetLayer*>::const_iterator i = flc.begin(); i != flc.end(); i++) {
    addEndcapLayer(*i); 
  }

  // get all muon barrel DetLayers (DT + RPC)
  vector<DetLayer*> barrel = muonGeom->allBarrelLayers();
  for ( vector<DetLayer*>::const_iterator i = barrel.begin(); i != barrel.end(); i++ ) {
    BarrelDetLayer* mbp = dynamic_cast<BarrelDetLayer*>(*i);
    if ( mbp == 0 ) throw Genexception("Bad BarrelDetLayer");
    addBarrelLayer(mbp);
  }

  // get all muon forward (+z) DetLayers (CSC + RPC)
  vector<DetLayer*> endcap = muonGeom->allEndcapLayers();
  for ( vector<DetLayer*>::const_iterator i = endcap.begin(); i != endcap.end(); i++ ) {
    ForwardDetLayer* mep = dynamic_cast<ForwardDetLayer*>(*i);
    if ( mep == 0 ) throw Genexception("Bad ForwardDetLayer");
    addEndcapLayer(mep);
  }

  // create outward links for all DetLayers
  linkBarrelLayers();
  linkEndcapLayers(theForwardLayers,theMuonForwardNLC, theTkForwardNLC);
  linkEndcapLayers(theBackwardLayers,theMuonBackwardNLC, theTkBackwardNLC);

}


//
// destructor
//
MuonTkNavigationSchool::~MuonTkNavigationSchool() {

   for_each(theTkBarrelNLC.begin(),theTkBarrelNLC.end(), delete_layer());
   for_each(theTkForwardNLC.begin(),theTkForwardNLC.end(), delete_layer());
   for_each(theTkBackwardNLC.begin(),theTkBackwardNLC.end(), delete_layer());
   for_each(theMuonBarrelNLC.begin(),theMuonBarrelNLC.end(), delete_layer());
   for_each(theMuonForwardNLC.begin(),theMuonForwardNLC.end(), delete_layer());
   for_each(theMuonBackwardNLC.begin(),theMuonBackwardNLC.end(), delete_layer());

   // delete the vector containing all the detlayers
   delete theAllDetLayersInSystem;

}


/* Operations as NavigationSchool */

//
//
//
vector<NavigableLayer*> MuonTkNavigationSchool::navigableLayers() const {
 
  vector<NavigableLayer*> result;
  
  for ( vector< SimpleBarrelNavigableLayer*>::const_iterator 
	  ib = theTkBarrelNLC.begin(); ib != theTkBarrelNLC.end(); ib++) {
    result.push_back( *ib);
  }
  for ( vector< SimpleForwardNavigableLayer*>::const_iterator 
	  ifl = theTkForwardNLC.begin(); ifl != theTkForwardNLC.end(); ifl++) {
    result.push_back( *ifl);
  }

  for ( vector< SimpleForwardNavigableLayer*>::const_iterator
          ifl = theTkBackwardNLC.begin(); ifl != theTkBackwardNLC.end(); ifl++) {
    result.push_back( *ifl);
  }

  vector<MuonBarrelNavigableLayer*>::const_iterator ib;
  vector<MuonForwardNavigableLayer*>::const_iterator ie;

  for ( ib = theMuonBarrelNLC.begin(); ib != theMuonBarrelNLC.end(); ib++ ) {
    result.push_back(*ib);
  }

  for ( ie = theMuonForwardNLC.begin(); ie != theMuonForwardNLC.end(); ie++ ) {
    result.push_back(*ie);
  }

  for ( ie = theMuonBackwardNLC.begin(); ie != theMuonBackwardNLC.end(); ie++ ) {
    result.push_back(*ie);
  }

  return result;

}


//
//
//
void MuonTkNavigationSchool::addBarrelLayer(BarrelDetLayer* mbp) {

  const BoundCylinder& bc = mbp->specificSurface();
  float radius = bc.radius();
  float length = bc.bounds().length()/2.;
  float eta_max = calculateEta(radius, length);
  float eta_min = -eta_max;
  edm::LogInfo("MuonTkNavigationSchool")<<"BarrelLayer eta: ("<<eta_min<<", "<<eta_max<<"). Radius "<<radius<<", Length "<<length;
  theBarrelLayers[mbp] = MuonEtaRange(eta_max, eta_min);

}


//
// create forwrad/backward layer maps
//
void MuonTkNavigationSchool::addEndcapLayer(ForwardDetLayer* mep) {

  const BoundDisk& bd = mep->specificSurface();
  float outRadius = bd.outerRadius();
  float inRadius = bd.innerRadius();
  float thick = bd.bounds().length()/2.;
  float z = bd.position().z();

  if ( z > 0. ) {
    float eta_min = calculateEta(outRadius, z-thick);
    float eta_max = calculateEta(inRadius, z+thick);
    edm::LogInfo("MuonTkNavigationSchool")<<"ForwardLayer eta: ("<<eta_min<<", "<<eta_max<<"). Radius ("<<inRadius<<", "<<outRadius<<"), Z "<<z;
    theForwardLayers[mep] = MuonEtaRange(eta_max, eta_min);
  } else {
    float eta_max = calculateEta(outRadius, z+thick);
    float eta_min = calculateEta(inRadius, z-thick);
    edm::LogInfo("MuonTkNavigationSchool")<<"BackwardLayer eta: ("<<eta_min<<", "<<eta_max<<"). Radius ("<<inRadius<<", "<<outRadius<<"), Z "<<z;
    theBackwardLayers[mep] = MuonEtaRange(eta_max, eta_min);
  }

}


//
//
//
void MuonTkNavigationSchool::linkBarrelLayers() {

  for (MapBI bl  = theBarrelLayers.begin(); bl != theBarrelLayers.end(); bl++) {

    MuonEtaRange range = (*bl).second;

    float length = fabs((*bl).first->specificSurface().bounds().length()/2.);
    // first add next barrel layer
    MapBI plusOne(bl);
    plusOne++;
    MapB outerBarrel;
    MapB allOuterBarrel;
    if ( plusOne != theBarrelLayers.end() ) { outerBarrel.insert(*plusOne); }
    // add all outer barrel layers
    for ( MapBI iMBI = plusOne; iMBI!= theBarrelLayers.end(); iMBI++) {
      allOuterBarrel.insert(*iMBI);
    }
    // then add all compatible backward layers with an eta criteria
    MapE allOuterBackward;
    for (MapEI el  = theBackwardLayers.begin();
               el != theBackwardLayers.end(); el++) {
      if ( (*el).second.isCompatible(range) ) {
        float z = (*el).first->specificSurface().position().z();
        if (fabs(z) < length) continue;
        allOuterBackward.insert(*el);
      }
    }

    // add the backward next layer with an eta criteria
    MapE outerBackward;
    for (MapEI el  = theBackwardLayers.begin();
               el != theBackwardLayers.end(); el++) {
      if ( (*el).second.isCompatible(range) ) {
        float z = (*el).first->specificSurface().position().z();
        if (fabs(z) < length) continue;
        outerBackward.insert(*el);
        break;
      }
    }

    // then add all compatible forward layers with an eta criteria
    MapE allOuterForward;
    for (MapEI el  = theForwardLayers.begin();
               el != theForwardLayers.end(); el++) {
      if ( (*el).second.isCompatible(range) ) {
        float z = (*el).first->specificSurface().position().z();
        if (fabs(z) < length) continue;
        allOuterForward.insert(*el);
      }
    }

    // then add forward next layer with an eta criteria
    MapE outerForward;
    for (MapEI el  = theForwardLayers.begin();
               el != theForwardLayers.end(); el++) {
      if ( (*el).second.isCompatible(range) ) {
        float z = (*el).first->specificSurface().position().z();
        if (fabs(z) < length) continue;
        outerForward.insert(*el);
        break;
      }
    }

    // first add next inner barrel layer
    MapBI minusOne(bl);
    MapB innerBarrel;
    MapB allInnerBarrel;
    MapE allInnerBackward;
    MapE innerBackward;
    MapE allInnerForward;
    MapE innerForward;

    if ( bl != theBarrelLayers.begin() ) {
      minusOne--;
      innerBarrel.insert(*minusOne);
        // add all inner barrel layers
      for ( MapBI iMBI = minusOne; iMBI != theBarrelLayers.begin(); iMBI--) {
        allInnerBarrel.insert(*iMBI);
      }
      allInnerBarrel.insert(*theBarrelLayers.begin());

      // then add all compatible backward layers with an eta criteria
      for (MapEI el  = theBackwardLayers.end();
                 el != theBackwardLayers.begin(); el--) {
        if (el == theBackwardLayers.end()) continue;  //C.L @@: no -/+ for map iterator
        if ( (*el).second.isCompatible(range) ) {
          float z = (*el).first->specificSurface().position().z();
          if (fabs(z) > length) continue;
          allInnerBackward.insert(*el);
        }
      }
      MapEI el = theBackwardLayers.begin();
      if (el->second.isCompatible(range)) {
        float z = (*el).first->specificSurface().position().z();
        if (fabs(z) < length) {
          allInnerBackward.insert(*el);
        }
      }

      // then add all compatible forward layers with an eta criteria
      for (MapEI el  = theForwardLayers.end();
                 el != theForwardLayers.begin(); el--) {
        if (el == theForwardLayers.end()) continue;
        if ( (*el).second.isCompatible(range) ) {
          float z = (*el).first->specificSurface().position().z();
          if (fabs(z) > length) continue;
          allInnerForward.insert(*el);
        }
      }

      el = theForwardLayers.begin();
      if (el->second.isCompatible(range)) {
        float z = (*el).first->specificSurface().position().z();
        if (fabs(z) < length)  {
          allInnerForward.insert(*el);
        }
      }

      if ( !range.isInside((*minusOne).second) ) {
        MuonEtaRange backwardRange(range.min(), (*minusOne).second.min());
        MuonEtaRange forwardRange((*minusOne).second.max(),range.max());

        // add the backward next layer with an eta criteria
        for (MapEI el  = theBackwardLayers.end();
                   el != theBackwardLayers.begin(); el--) {
          if ( el == theBackwardLayers.end() ) continue; 
          if ( (*el).second.isCompatible(backwardRange) ) {
            float z = (*el).first->specificSurface().position().z();
            if (fabs(z) > length) continue;
            innerBackward.insert(*el);
            backwardRange = backwardRange.subtract((*el).second);
          }
        }

        MapEI el = theBackwardLayers.begin();
        if (el->second.isCompatible(backwardRange)) {
          float z = (*el).first->specificSurface().position().z();
          if (fabs(z) < length)  {
            innerBackward.insert(*el);
          }
        }
      
        // then add forward next layer with an eta criteria
        for (MapEI el  = theForwardLayers.end();
                   el != theForwardLayers.begin(); el--) {
          if ( el == theForwardLayers.end() ) continue; 
          if ( (*el).second.isCompatible(forwardRange) ) {
            float z = (*el).first->specificSurface().position().z();
            if (fabs(z) > length) continue;
            innerForward.insert(*el);
            forwardRange = forwardRange.subtract((*el).second);

          }
        }
        el = theForwardLayers.begin();
        if (el->second.isCompatible(forwardRange)) {
          float z = (*el).first->specificSurface().position().z();
          if (fabs(z) < length) innerForward.insert(*el);
        }
      }
    }

    BarrelDetLayer* mbp = (*bl).first;
    if ( mbp->subDetector() == GeomDetEnumerators::DT || mbp->subDetector() == GeomDetEnumerators::RPCBarrel ) {
      theMuonBarrelNLC.push_back(new MuonBarrelNavigableLayer(mbp,
                                                              outerBarrel, 
                                                              innerBarrel, 
                                                              outerBackward, 
                                                              outerForward, 
                                                              innerBackward, 
                                                              innerForward, 
                                                              allOuterBarrel,
                                                              allInnerBarrel, 
                                                              allOuterBackward,
                                                              allOuterForward, 
                                                              allInnerBackward, 
                                                              allInnerForward));
    }                                                          
    else if ( mbp->subDetector() == GeomDetEnumerators::PixelBarrel || mbp->subDetector() == GeomDetEnumerators::TIB || mbp->subDetector() == GeomDetEnumerators::TOB ) {
      BDLC outerBarrelLayers;
      BDLC innerBarrelLayers;
      BDLC allOuterBarrelLayers;
      BDLC allInnerBarrelLayers;
      FDLC outerBackwardLayers;
      FDLC outerForwardLayers;
      FDLC allOuterBackwardLayers;
      FDLC allOuterForwardLayers;
      FDLC innerBackwardLayers;
      FDLC innerForwardLayers;
      FDLC allInnerBackwardLayers;
      FDLC allInnerForwardLayers;

     for (MapBI ib = outerBarrel.begin(); ib != outerBarrel.end(); ib++) {
       BarrelDetLayer* ibdl = (*ib).first;
       outerBarrelLayers.push_back(ibdl);
     }

     for (MapBI ib = innerBarrel.begin(); ib != innerBarrel.end(); ib++) {
       BarrelDetLayer* ibdl = (*ib).first;
       innerBarrelLayers.push_back(ibdl);
     }
   
     for (MapBI ib = allOuterBarrel.begin(); ib != allOuterBarrel.end(); ib++) {
       BarrelDetLayer* ibdl = (*ib).first;
       allOuterBarrelLayers.push_back(ibdl);
     }

     for (MapBI ib = allInnerBarrel.begin(); ib != allInnerBarrel.end(); ib++) {
       BarrelDetLayer* ibdl = (*ib).first;
       allInnerBarrelLayers.push_back(ibdl);
     }

     for (MapEI ie = outerBackward.begin(); ie != outerBackward.end(); ie++) {
       ForwardDetLayer* ifdl = (*ie).first;
       outerBackwardLayers.push_back(ifdl);
     }

     for (MapEI ie = outerForward.begin(); ie != outerForward.end(); ie++) {
       ForwardDetLayer* ifdl = (*ie).first;
       outerForwardLayers.push_back(ifdl);
     }

     for (MapEI ie = allOuterBackward.begin(); ie != allOuterBackward.end(); ie++) {
       ForwardDetLayer* ifdl = (*ie).first;
       allOuterBackwardLayers.push_back(ifdl);
     }

     for (MapEI ie = allOuterForward.begin(); ie != allOuterForward.end(); ie++) {
       ForwardDetLayer* ifdl = (*ie).first;
       allOuterForwardLayers.push_back(ifdl);
     }

     for (MapEI ie = innerBackward.begin(); ie != innerBackward.end(); ie++) {
       ForwardDetLayer* ifdl = (*ie).first;
       innerBackwardLayers.push_back(ifdl);
     }

     for (MapEI ie = innerForward.begin(); ie != innerForward.end(); ie++) {
       ForwardDetLayer* ifdl = (*ie).first;
       innerForwardLayers.push_back(ifdl);
     }

     for (MapEI ie = allInnerBackward.begin(); ie != allInnerBackward.end(); ie++) {
       ForwardDetLayer* ifdl = (*ie).first;
       allInnerBackwardLayers.push_back(ifdl);
     }

     for (MapEI ie = allInnerForward.begin(); ie != allInnerForward.end(); ie++) {
       ForwardDetLayer* ifdl = (*ie).first;
       allInnerForwardLayers.push_back(ifdl);
     }

     theTkBarrelNLC.push_back(new SimpleBarrelNavigableLayer(mbp,outerBarrelLayers,
                                                                 innerBarrelLayers, 
                                                                 allOuterBarrelLayers, 
                                                                 allInnerBarrelLayers,
                                                                 outerBackwardLayers, 
                                                                 outerForwardLayers, 
                                                                 allOuterBackwardLayers, 
                                                                 allOuterForwardLayers, 
                                                                 innerBackwardLayers, 
                                                                 innerForwardLayers, 
                                                                 allInnerBackwardLayers, 
                                                                 allInnerForwardLayers,
                                                                 theMagneticField, 5.));

    }

  }

}


//
//
//
void MuonTkNavigationSchool::linkEndcapLayers(const MapE& layers,
                                              std::vector<MuonForwardNavigableLayer*>& resultM, 
                                              std::vector<SimpleForwardNavigableLayer*>& resultT) {

  for (MapEI el = layers.begin(); el != layers.end(); el++) {

    MuonEtaRange range = (*el).second;
    float z = (*el).first->specificSurface().position().z();
    // first add next endcap layer (if compatible)
    MapEI plusOne(el); 
    plusOne++;
    MuonEtaRange tempR(range);
    MuonEtaRange secondOR(range);
    MapEI outerOne(plusOne);
    bool outerDoubleCheck = false;
    MapE outerELayers;
    if ( plusOne != layers.end()) {
        for ( MapEI l = plusOne; l != layers.end(); l++ ) {
          if ( (*l).second.isCompatible(tempR)) {
            outerELayers.insert(*l);
            if ( tempR.isInside((*l).second) ) break;
            if ((*l).second.isInside(tempR)) {
                  // split into 2 pieces
                  outerOne = l;
                  outerOne++;
                  if (tempR.max() > 0 ) {
                    secondOR = MuonEtaRange(tempR.max(),(*l).second.max());
                    tempR = MuonEtaRange((*l).second.min(),tempR.min());
                  }else {
                    secondOR = MuonEtaRange((*l).second.min(),tempR.min());
                    tempR = MuonEtaRange(tempR.max(),(*l).second.max());
                  }
                  outerDoubleCheck = true;
                  break;
             }
            tempR = tempR.subtract((*l).second);
          } //if ( (*l).second.isCompatible(tempR))
      }//for

      if (outerDoubleCheck) {
        for ( MapEI l = outerOne; l != layers.end(); l++ ) {
          if ( (*l).second.isCompatible(tempR)) {
            outerELayers.insert(*l);
            if ( tempR.isInside((*l).second) ) break;
            tempR = tempR.subtract((*l).second);
          } //if ( (*l).second.isCompatible(tempR))
        }//for

        for ( MapEI l = outerOne; l != layers.end(); l++ ) {
          if ( (*l).second.isCompatible(secondOR)) {
            outerELayers.insert(*l);
            if ( secondOR.isInside((*l).second) ) break;
            secondOR = secondOR.subtract((*l).second);
          } //if ( (*l).second.isCompatible(tempR))
        }//for
      }
    }//if end

    MapE allOuterELayers;
    for (MapEI iMEI = plusOne; iMEI!=layers.end(); iMEI++){
      if ((*iMEI).second.isCompatible(range)) allOuterELayers.insert(*iMEI);
    }
    // to avoid overlap
    int i = 0;
    bool hasOverlap = false; 
    MapB outerBLayers; 
    MapB allOuterBLayers;
    for (MapBI iMBI = theBarrelLayers.begin(); iMBI!=theBarrelLayers.end(); iMBI++){
      if ((*iMBI).second.isCompatible(tempR)) {
        float length = fabs((*iMBI).first->specificSurface().bounds().length()/2.);
        if (length > fabs(z)) {
           if ( (i==0) && (tempR.isInside((*iMBI).second)) ) hasOverlap = true;
           i++;
           outerBLayers.insert(*iMBI);
           if (tempR.isInside((*iMBI).second)) break;
           tempR = tempR.subtract((*iMBI).second);
         }
       }
    }

    for (MapBI iMBI = theBarrelLayers.begin(); iMBI!=theBarrelLayers.end(); iMBI++){
      float length = fabs((*iMBI).first->specificSurface().bounds().length()/2.);
      if (length < fabs(z)) continue; 
      if ((*iMBI).second.isCompatible(range)) allOuterBLayers.insert(*iMBI);
    }

    MapE innerELayers;
    MapE allInnerELayers;
    MapB innerBLayers;
    MapB allInnerBLayers;
    MuonEtaRange itempR(range);
    bool checkFurther = true;
    bool doubleCheck = false;
    MuonEtaRange secondR;
    float outRadius = 0;
    MapEI minusOne(el);
    if (el != layers.begin()) {
      minusOne--;
      outRadius = minusOne->first->specificSurface().outerRadius();
      MapEI innerOne;
      for (MapEI iMEI = minusOne; iMEI!=layers.begin(); iMEI--){
        if ( (*iMEI).second.isCompatible(itempR) ) {
          innerELayers.insert(*iMEI);

          if (itempR.isInside((*iMEI).second)) { checkFurther = false; break; }
          if ((*iMEI).second.isInside(itempR)) { 
                  // split into 2 pieces
                  doubleCheck = true; 
                  innerOne = iMEI; 
                  innerOne--; 
                  if (itempR.max() > 0 ) {
                    secondR = MuonEtaRange(itempR.max(),(*iMEI).second.max());
                    itempR = MuonEtaRange((*iMEI).second.min(),itempR.min());
                  }else {
                    itempR = MuonEtaRange(itempR.max(),(*iMEI).second.max());
                    secondR = MuonEtaRange((*iMEI).second.min(),itempR.min());
                  }
                  break; 
            }
          else itempR = itempR.subtract((*iMEI).second);  
        }//if ( (*iMEI).second.isCompatible(itempR) ) 
      }//for MapEI
      if (doubleCheck ) {

        for (MapEI iMEI = innerOne; iMEI!=layers.begin(); iMEI--){
          if ( (*iMEI).second.isCompatible(itempR) ) {
            innerELayers.insert(*iMEI);
            if (itempR.isInside((*iMEI).second)) { checkFurther = false; break; }
            else itempR = itempR.subtract((*iMEI).second);
          }//if ( (*iMEI).second.isCompatible(itempR) )
        }//for MapEI

        for (MapEI iMEI = innerOne; iMEI!=layers.begin(); iMEI--){
          if ( (*iMEI).second.isCompatible(secondR) ) {
            innerELayers.insert(*iMEI);
            if (secondR.isInside((*iMEI).second)) { checkFurther = false; break; }
            else secondR = secondR.subtract((*iMEI).second);
          }//if ( (*iMEI).second.isCompatible(itempR) )
        }//for MapEI
      }// if doubleCheck

      if (checkFurther && (*layers.begin()).second.isCompatible(itempR)) {
          innerELayers.insert(*layers.begin());
          itempR = itempR.subtract((*layers.begin()).second);
       }

      for (MapEI iMEI = minusOne; iMEI!=layers.begin(); iMEI--) {
        if ((*iMEI).second.isCompatible(range)) allInnerELayers.insert(*iMEI);
      }
      if ((*layers.begin()).second.isCompatible(range)) allInnerELayers.insert(*layers.begin());
    } 
    

    for (MapBI iMBI = theBarrelLayers.end(); iMBI!=theBarrelLayers.begin(); iMBI--) {
      if (iMBI == theBarrelLayers.end()) continue;
      float length = fabs((*iMBI).first->specificSurface().bounds().length()/2.);
      if (length > fabs(z)) continue;
      if ((*iMBI).second.isCompatible(range)) allInnerBLayers.insert(*iMBI);
    }
    if ((*theBarrelLayers.begin()).second.isCompatible(range)) allInnerBLayers.insert(*theBarrelLayers.begin());

    int k = 0;
    bool hasOverlap2 = false;
    bool hasInsideE = false;
    for (MapBI iMBI = theBarrelLayers.end(); iMBI!=theBarrelLayers.begin(); iMBI--) {
      if (iMBI == theBarrelLayers.end()) continue;
      float length = fabs((*iMBI).first->specificSurface().bounds().length()/2.);
      if (length > fabs(z)) continue;
      float radius = (*iMBI).first->specificSurface().radius();

      bool compatible = false;
      if (radius > outRadius) { 
             compatible = (*iMBI).second.isCompatible(range);
             if (compatible && outRadius > 40) hasInsideE = true;//CL: no general rule
      }
      else compatible = (*iMBI).second.isCompatible(itempR);
      if (!checkFurther && (radius < outRadius)) break;
      if (compatible) {
        if ((k==0) && (itempR.isInside((*iMBI).second)) && (radius < outRadius)) hasOverlap2 = true;
        if (radius < outRadius) k++;
        innerBLayers.insert(*iMBI);
        if (itempR.isInside((*iMBI).second) && (radius < outRadius)) break;
        itempR = itempR.subtract((*iMBI).second);
       }
    }
    
    if (el == layers.begin() && (*theBarrelLayers.begin()).second.isCompatible(itempR)) innerBLayers.insert(*theBarrelLayers.begin());
    
    ForwardDetLayer* mbp = (*el).first;
    if ( mbp->subDetector() == GeomDetEnumerators::CSC || mbp->subDetector() == GeomDetEnumerators::RPCEndcap ) {
      resultM.push_back(new MuonForwardNavigableLayer(mbp, 
                                                      innerBLayers, 
                                                      outerELayers, 
                                                      innerELayers, 
                                                      allInnerBLayers, 
                                                      allOuterELayers, 
                                                      allInnerELayers));
    }
    else if ( mbp->subDetector() == GeomDetEnumerators::PixelEndcap || mbp->subDetector() == GeomDetEnumerators::TEC || mbp->subDetector() ==  GeomDetEnumerators::TID ) {
      BDLC outerBarrelLayers;
      FDLC outerForwardLayers;
      BDLC allOuterBarrelLayers;
      FDLC allOuterForwardLayers;
      BDLC innerBarrelLayers;
      FDLC innerForwardLayers;
      BDLC allInnerBarrelLayers;
      FDLC allInnerForwardLayers;

     unsigned int j = 0;
     unsigned int l = 0;
     unsigned int m = 0;

     for (MapBI ib = outerBLayers.begin(); ib != outerBLayers.end(); ib++) {
       BarrelDetLayer* ibdl = (*ib).first;
       outerBarrelLayers.push_back(ibdl);
     }

     for (MapEI ie = outerELayers.begin(); ie != outerELayers.end(); ie++) {
       j++;
       if ( hasOverlap && j==outerELayers.size() ) break; 
       ForwardDetLayer* ifdl = (*ie).first;
       outerForwardLayers.push_back(ifdl);
     }

     for (MapBI ib = allOuterBLayers.begin(); ib != allOuterBLayers.end(); ib++) {
       BarrelDetLayer* ibdl = (*ib).first;
       allOuterBarrelLayers.push_back(ibdl);
     }

     for (MapEI ie = allOuterELayers.begin(); ie != allOuterELayers.end(); ie++) {
       ForwardDetLayer* ifdl = (*ie).first;
       allOuterForwardLayers.push_back(ifdl);
     }

     for (MapBI ib = innerBLayers.begin(); ib != innerBLayers.end(); ib++) {
       l++;
       if (hasOverlap2 && l==innerBLayers.size() ) continue;
       BarrelDetLayer* ibdl = (*ib).first;
       innerBarrelLayers.push_back(ibdl);
     }

     for (MapEI ie = innerELayers.begin(); ie != innerELayers.end(); ie++) {
       m++;
       if (hasInsideE && m==innerELayers.size()-2 ) continue;
       ForwardDetLayer* ifdl = (*ie).first;
       innerForwardLayers.push_back(ifdl);
     }

     for (MapBI ib = allInnerBLayers.begin(); ib != allInnerBLayers.end(); ib++) {
       BarrelDetLayer* ibdl = (*ib).first;
       allInnerBarrelLayers.push_back(ibdl);
     }

     for (MapEI ie = allInnerELayers.begin(); ie != allInnerELayers.end(); ie++) {
       ForwardDetLayer* ifdl = (*ie).first;
       allInnerForwardLayers.push_back(ifdl);
     }

     resultT.push_back(new SimpleForwardNavigableLayer(mbp, 
                                                       outerBarrelLayers,
                                                       allOuterBarrelLayers, 
                                                       innerBarrelLayers, 
                                                       allInnerBarrelLayers, 
                                                       outerForwardLayers,
                                                       allOuterForwardLayers, 
                                                       innerForwardLayers, 
                                                       allInnerForwardLayers,
                                                       theMagneticField, 5.));
    }

  }

}


//
// calculate the length of the barrel
//
float MuonTkNavigationSchool::barrelLength() const {

  float length = 0.0;
  for (MapBI i= theBarrelLayers.begin(); i != theBarrelLayers.end(); i++) {
    if ((*i).first->subDetector() != GeomDetEnumerators::PixelBarrel && (*i).first->subDetector() != GeomDetEnumerators::TIB && (*i).first->subDetector() != GeomDetEnumerators::TOB) continue;
    length = max(length,(*i).first->surface().bounds().length()/2.f);
  }

  return length;

}


//
// calculate pseudorapidity from r and z
//
float MuonTkNavigationSchool::calculateEta(float r, float z) const {

  if ( z > 0 ) return -log((tan(atan(r/z)/2.)));
  return log(-(tan(atan(r/z)/2.)));

}
