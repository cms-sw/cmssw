#include "RecoMuon/Navigation/interface/MuonTkNavigationSchool.h"

#include "RecoTracker/TkNavigation/interface/SimpleBarrelNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/SimpleForwardNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/DiskLessInnerRadius.h"
#include "RecoTracker/TkNavigation/interface/SymmetricLayerFinder.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "TrackingTools/DetLayers/src/DetBelowZ.h"
#include "TrackingTools/DetLayers/src/DetLessZ.h"
#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"
#include "RecoMuon/Navigation/interface/MuonBarrelNavigableLayer.h"
#include "RecoMuon/Navigation/interface/MuonForwardNavigableLayer.h"
#include "RecoMuon/Navigation/interface/MuonLayerSort.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "Utilities/General/interface/CMSexception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <functional>
#include <algorithm>
#include <map>
#include <cmath>


typedef std::vector<DetLayer*> LayerContainer;
/* Constructor */
MuonTkNavigationSchool::MuonTkNavigationSchool(const MuonDetLayerGeometry * muonGeom, const GeometricSearchTracker * trackerGeom, const MagneticField * field) : theMuonDetLayerGeometry(muonGeom), theGeometricSearchTracker(trackerGeom), theMagneticField(field), theBarrelLength(0) {
  
  // Get tracker barrel layers
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

  // establish the inwards links from Muon to Tracker
//  createInverseLinks(); 
}

MuonTkNavigationSchool::~MuonTkNavigationSchool() {

}
/* Operations as NavigationSchool */

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

void MuonTkNavigationSchool::addBarrelLayer(BarrelDetLayer* mbp) {

  BoundCylinder* bc = dynamic_cast<BoundCylinder*>(const_cast<BoundSurface*>(&(mbp->surface())));
  float radius = bc->radius();
  float length = bc->bounds().length()/2.;
  float eta_max = calculateEta(radius, length);
  float eta_min = -eta_max;
  edm::LogInfo("MuonTkNavigationSchool")<<"BarrelLayer eta: ("<<eta_min<<", "<<eta_max<<"). Radius "<<radius<<", Length "<<length;
  theBarrelLayers[mbp] = MuonEtaRange(eta_max, eta_min);

}


//
// create forwrad/backward layer maps
//
void MuonTkNavigationSchool::addEndcapLayer(ForwardDetLayer* mep) {

  BoundDisk* bd = dynamic_cast<BoundDisk*>(const_cast<BoundSurface*>(&(mep->surface())));
  float outRadius = bd->outerRadius();
  float inRadius = bd->innerRadius();
  float thick = bd->bounds().length()/2.;
  float z = bd->position().z();

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

for (MapBI bl  = theBarrelLayers.begin();
             bl != theBarrelLayers.end(); bl++) {

    MuonEtaRange range = (*bl).second;


    BoundCylinder* bc = dynamic_cast<BoundCylinder*>(const_cast<BoundSurface*>(&((*bl).first->surface())));
    float length = fabs(bc->bounds().length()/2.);
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
      BoundDisk* bd = dynamic_cast<BoundDisk*>(const_cast<BoundSurface*>(&((*el).first->surface())));
        float z = bd->position().z();
        if (fabs(z) < length) continue;
        allOuterBackward.insert(*el);
      }
    }

    //add the backward next layer with an eta criteria
    MapE outerBackward;
    for (MapEI el  = theBackwardLayers.begin();
               el != theBackwardLayers.end(); el++) {
      if ( (*el).second.isCompatible(range) ) {
      BoundDisk* bd = dynamic_cast<BoundDisk*>(const_cast<BoundSurface*>(&((*el).first->surface())));
        float z = bd->position().z();
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
      BoundDisk* bd = dynamic_cast<BoundDisk*>(const_cast<BoundSurface*>(&((*el).first->surface())));
        float z = bd->position().z();
        if (fabs(z) < length) continue;
        allOuterForward.insert(*el);
      }
    }

    // then add forward next layer with an eta criteria
    MapE outerForward;
    for (MapEI el  = theForwardLayers.begin();
               el != theForwardLayers.end(); el++) {
      if ( (*el).second.isCompatible(range) ) {
      BoundDisk* bd = dynamic_cast<BoundDisk*>(const_cast<BoundSurface*>(&((*el).first->surface())));
        float z = bd->position().z();
        if (fabs(z) < length) continue;
        outerForward.insert(*el);
        break;
      }
    }
    // first add next inner barrel layer
    MapBI minusOne(bl);
    MapB innerBarrel;
    MapB allInnerBarrel;

    if ( bl != theBarrelLayers.begin()) {
    minusOne--;
    innerBarrel.insert(*minusOne);
    // add all inner barrel layers
    for ( MapBI iMBI = minusOne; iMBI != theBarrelLayers.begin(); iMBI--){
        allInnerBarrel.insert(*iMBI);
      }
    allInnerBarrel.insert(*theBarrelLayers.begin());
    }
    // then add all compatible backward layers with an eta criteria
    MapE allInnerBackward;
    for (MapEI el  = theBackwardLayers.end();
               el != theBackwardLayers.begin(); el--) {
      if (el == theBackwardLayers.end()) continue;  //C.L @@: no -/+ for map iterator
      if ( (*el).second.isCompatible(range) ) {
      BoundDisk* bd = dynamic_cast<BoundDisk*>(const_cast<BoundSurface*>(&((*el).first->surface())));
        float z = bd->position().z();
        if (fabs(z) > length) continue;
        allInnerBackward.insert(*el);
      }
    }
    //add the backward next layer with an eta criteria
    MapE innerBackward;
    for (MapEI el  = theBackwardLayers.end();
               el != theBackwardLayers.begin(); el--) {
      if (el == theBackwardLayers.end()) continue; 
      if ( (*el).second.isCompatible(range) ) {
      BoundDisk* bd = dynamic_cast<BoundDisk*>(const_cast<BoundSurface*>(&((*el).first->surface())));
        float z = bd->position().z();
        if (fabs(z) > length) continue;
        innerBackward.insert(*el);
        break;
      }
    }

    MapEI el = theBackwardLayers.begin();
    if (el->second.isCompatible(range)) {
      BoundDisk* bd = dynamic_cast<BoundDisk*>(const_cast<BoundSurface*>(&((*el).first->surface())));
        float z = bd->position().z();
        if (fabs(z) < length)  {
          allInnerBackward.insert(*el);
          innerBackward.insert(*el);
        }
    }

    // then add all compatible forward layers with an eta criteria
    MapE allInnerForward;
    for (MapEI el  = theForwardLayers.end();
               el != theForwardLayers.begin(); el--) {
      if (el == theForwardLayers.end()) continue;  
      if ( (*el).second.isCompatible(range) ) {
      BoundDisk* bd = dynamic_cast<BoundDisk*>(const_cast<BoundSurface*>(&((*el).first->surface())));
        float z = bd->position().z();
        if (fabs(z) > length) continue;
        allInnerForward.insert(*el);
      }
    }

    // then add forward next layer with an eta criteria
    MapE innerForward;
    for (MapEI el  = theForwardLayers.end();
               el != theForwardLayers.begin(); el--) {
      if (el == theForwardLayers.end()) continue; 
      if ( (*el).second.isCompatible(range) ) {
      BoundDisk* bd = dynamic_cast<BoundDisk*>(const_cast<BoundSurface*>(&((*el).first->surface())));
        float z = bd->position().z();
        if (fabs(z) > length) continue;
        innerForward.insert(*el);
        break;
      }
    }
    el = theForwardLayers.begin();
    if (el->second.isCompatible(range)) {
      BoundDisk* bd = dynamic_cast<BoundDisk*>(const_cast<BoundSurface*>(&((*el).first->surface())));
        float z = bd->position().z();
        if (fabs(z) < length) {
          allInnerForward.insert(*el);
          innerForward.insert(*el);
        }
    }


    BarrelDetLayer* mbp = const_cast<BarrelDetLayer*>((*bl).first);
    if (mbp->module() == dt || mbp->module() == rpc)
    theMuonBarrelNLC.push_back(new MuonBarrelNavigableLayer(
                       mbp,
                       outerBarrel, innerBarrel, outerBackward, outerForward, innerBackward, innerForward, allOuterBarrel,allInnerBarrel, allOuterBackward,allOuterForward, allInnerBackward, allInnerForward));
   else if(mbp->module() == pixel || mbp->module() == silicon){
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
         BarrelDetLayer* ibdl = const_cast<BarrelDetLayer*>((*ib).first);
         outerBarrelLayers.push_back(ibdl);
        }
     for (MapBI ib = innerBarrel.begin(); ib != innerBarrel.end(); ib++) {
         BarrelDetLayer* ibdl = const_cast<BarrelDetLayer*>((*ib).first);
         innerBarrelLayers.push_back(ibdl);
        }
   
     for (MapBI ib = allOuterBarrel.begin(); ib != allOuterBarrel.end(); ib++) {
         BarrelDetLayer* ibdl = const_cast<BarrelDetLayer*>((*ib).first);
         allOuterBarrelLayers.push_back(ibdl);
        }
     for (MapBI ib = allInnerBarrel.begin(); ib != allInnerBarrel.end(); ib++) {
         BarrelDetLayer* ibdl = const_cast<BarrelDetLayer*>((*ib).first);
         allInnerBarrelLayers.push_back(ibdl);
        }

     for (MapEI ie = outerBackward.begin(); ie != outerBackward.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         outerBackwardLayers.push_back(ifdl);
        }
     for (MapEI ie = outerForward.begin(); ie != outerForward.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         outerForwardLayers.push_back(ifdl);
        }
     for (MapEI ie = allOuterBackward.begin(); ie != allOuterBackward.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         allOuterBackwardLayers.push_back(ifdl);
        }
     for (MapEI ie = allOuterForward.begin(); ie != allOuterForward.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         allOuterForwardLayers.push_back(ifdl);
        }
     for (MapEI ie = innerBackward.begin(); ie != innerBackward.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         innerBackwardLayers.push_back(ifdl);
        }
     for (MapEI ie = innerForward.begin(); ie != innerForward.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         innerForwardLayers.push_back(ifdl);
        }
     for (MapEI ie = allOuterBackward.begin(); ie != allOuterBackward.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         allOuterBackwardLayers.push_back(ifdl);
        }
     for (MapEI ie = allOuterForward.begin(); ie != allOuterForward.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         allOuterForwardLayers.push_back(ifdl);
        }


   theTkBarrelNLC.push_back(new SimpleBarrelNavigableLayer(mbp, outerBarrelLayers,innerBarrelLayers, allOuterBarrelLayers, allInnerBarrelLayers,outerBackwardLayers, outerForwardLayers, allOuterBackwardLayers, allOuterForwardLayers, innerBackwardLayers, innerForwardLayers, allInnerBackwardLayers, allInnerForwardLayers,
                                 theMagneticField, 5.));

    }

  }

}



void MuonTkNavigationSchool::linkEndcapLayers(const MapE& layers,
                                            std::vector<MuonForwardNavigableLayer*>& resultM, std::vector<SimpleForwardNavigableLayer*>& resultT) {

  for (MapEI el = layers.begin(); el != layers.end(); el++) {

    MuonEtaRange range = (*el).second;
    BoundDisk* bd = dynamic_cast<BoundDisk*>(const_cast<BoundSurface*>(&((*el).first->surface())));
    float z = bd->position().z();
    // first add next endcap layer (if compatible)
    MapEI plusOne(el); 
    plusOne++;
    MapB outerBLayers; 
    MapB allOuterBLayers;
    MuonEtaRange tempR(range);
    for (MapBI iMBI = theBarrelLayers.begin(); iMBI!=theBarrelLayers.end(); iMBI++){
      if ((*iMBI).second.isCompatible(range)) {
        BoundCylinder* bc = dynamic_cast<BoundCylinder*>(const_cast<BoundSurface*>(&((*iMBI).first->surface())));
        float length = fabs(bc->bounds().length()/2.);
        if (length > fabs(z)) {
           outerBLayers.insert(*iMBI);
           if (tempR.isInside((*iMBI).second)) break;
           tempR = (*iMBI).second.subtract(tempR);
         }
       }
    }

    for (MapBI iMBI = theBarrelLayers.begin(); iMBI!=theBarrelLayers.end(); iMBI++){
      BoundCylinder* bc = dynamic_cast<BoundCylinder*>(const_cast<BoundSurface*>(&((*iMBI).first->surface())));
      float length = fabs(bc->bounds().length()/2.);
      if (length < fabs(z)) continue; 
      if ((*iMBI).second.isCompatible(range)) allOuterBLayers.insert(*iMBI);
    }

    MapE outerELayers;
    if ( plusOne != layers.end() && (*plusOne).second.isCompatible(range) ) {
        outerELayers.insert(*plusOne);
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
            outerELayers.insert(*l);
          }
        }
      }
    }

    MapE allOuterELayers;
    for (MapEI iMEI = plusOne; iMEI!=layers.end(); iMEI++){
      if ((*iMEI).second.isCompatible(range)) allOuterELayers.insert(*iMEI);
    }

    MapE innerELayers;
    MapE allInnerELayers;

    if (el != layers.begin()) {
      MapEI minusOne(el);
      minusOne--;
      MuonEtaRange tempR(range);
      for (MapEI iMEI = minusOne; iMEI!=layers.begin(); iMEI--){
        if ( (*iMEI).second.isCompatible(tempR) ) {
          innerELayers.insert(*iMEI);
          if (tempR.isInside((*iMEI).second)) break;
          tempR = (*iMEI).second.subtract(tempR);  
        }
      }
      for (MapEI iMEI = minusOne; iMEI!=layers.begin(); iMEI--){
        if ((*iMEI).second.isCompatible(range)) allInnerELayers.insert(*iMEI);
      }
      if ((*layers.begin()).second.isCompatible(range)) allInnerELayers.insert(*layers.begin());
    }
    tempR = range;

    MapB innerBLayers;
    for (MapBI iMBI = theBarrelLayers.end(); iMBI!=theBarrelLayers.begin(); iMBI--){
      if (iMBI == theBarrelLayers.end()) continue;
      if ((*iMBI).second.isCompatible(range)) {
        innerBLayers.insert(*iMBI);
        if (tempR.isInside((*iMBI).second)) break;
        tempR = (*iMBI).second.subtract(tempR);

       }
    }

    MapB allInnerBLayers;
    for (MapBI iMBI = theBarrelLayers.end(); iMBI!=theBarrelLayers.begin(); iMBI--){
      if (iMBI == theBarrelLayers.end()) continue;
      BoundCylinder* bc = dynamic_cast<BoundCylinder*>(const_cast<BoundSurface*>(&((*iMBI).first->surface())));
      float length = fabs(bc->bounds().length()/2.);
      if (length > fabs(z)) continue;
      if ((*iMBI).second.isCompatible(range)) allInnerBLayers.insert(*iMBI);
    }
    if ((*theBarrelLayers.begin()).second.isCompatible(range)) allInnerBLayers.insert(*theBarrelLayers.begin());

    ForwardDetLayer* mbp = const_cast<ForwardDetLayer*>((*el).first);
    if (mbp->module() == csc || mbp->module() == rpc)
    resultM.push_back(new MuonForwardNavigableLayer(
                   mbp, innerBLayers, outerELayers, innerELayers, allInnerBLayers, allOuterELayers, allInnerELayers));

   else if(mbp->module() == pixel || mbp->module() == silicon){
      BDLC outerBarrelLayers;
      FDLC outerForwardLayers;
      BDLC allOuterBarrelLayers;
      FDLC allOuterForwardLayers;
      BDLC innerBarrelLayers;
      FDLC innerForwardLayers;
      BDLC allInnerBarrelLayers;
      FDLC allInnerForwardLayers;

     for (MapBI ib = outerBLayers.begin(); ib != outerBLayers.end(); ib++) {
         BarrelDetLayer* ibdl = const_cast<BarrelDetLayer*>((*ib).first);
         outerBarrelLayers.push_back(ibdl);
        }
     for (MapEI ie = outerELayers.begin(); ie != outerELayers.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         outerForwardLayers.push_back(ifdl);
        }

     for (MapBI ib = allOuterBLayers.begin(); ib != allOuterBLayers.end(); ib++) {
         BarrelDetLayer* ibdl = const_cast<BarrelDetLayer*>((*ib).first);
         allOuterBarrelLayers.push_back(ibdl);
        }
     for (MapEI ie = allOuterELayers.begin(); ie != allOuterELayers.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         allOuterForwardLayers.push_back(ifdl);
        }

     for (MapBI ib = innerBLayers.begin(); ib != innerBLayers.end(); ib++) {
         BarrelDetLayer* ibdl = const_cast<BarrelDetLayer*>((*ib).first);
         innerBarrelLayers.push_back(ibdl);
        }
     for (MapEI ie = innerELayers.begin(); ie != innerELayers.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         innerForwardLayers.push_back(ifdl);
        }

     for (MapBI ib = allInnerBLayers.begin(); ib != allInnerBLayers.end(); ib++) {
         BarrelDetLayer* ibdl = const_cast<BarrelDetLayer*>((*ib).first);
         allInnerBarrelLayers.push_back(ibdl);
        }

     for (MapEI ie = allInnerELayers.begin(); ie != allInnerELayers.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         allInnerForwardLayers.push_back(ifdl);
        }

    resultT.push_back(new SimpleForwardNavigableLayer(mbp, outerBarrelLayers,
               allOuterBarrelLayers, innerBarrelLayers, allInnerBarrelLayers, 
               outerForwardLayers,allOuterForwardLayers, innerForwardLayers, allInnerForwardLayers,theMagneticField, 5.));
    }

  }

}


float MuonTkNavigationSchool::barrelLength() {

  if ( theBarrelLength < 1.) {
  for (MapBI i= theBarrelLayers.begin(); i != theBarrelLayers.end(); i++) {
     if ((*i).first->module() !=pixel && (*i).first->module() != silicon) continue;
     theBarrelLength = max(theBarrelLength,(*i).first->surface().bounds().length()/2.f);
    }
  }

  return theBarrelLength;

}


void MuonTkNavigationSchool::createInverseLinks() const {

}


//
// calculate pseudorapidity from r and z
//
float MuonTkNavigationSchool::calculateEta(float r, float z) const {

  if ( z > 0 ) return -log((tan(atan(r/z)/2.)));
  return log(-(tan(atan(r/z)/2.)));

}

