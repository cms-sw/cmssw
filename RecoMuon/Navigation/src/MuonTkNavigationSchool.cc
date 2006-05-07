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
  vector<DetLayer*> fwd = muonGeom->allForwardLayers();
  for ( vector<DetLayer*>::const_iterator i = fwd.begin(); i != fwd.end(); i++ ) {
    ForwardDetLayer* mep = dynamic_cast<ForwardDetLayer*>(*i);
    if ( mep == 0 ) throw Genexception("Bad ForwardDetLayer");
    addEndcapLayer(mep);
  }

 // create outward links for all DetLayers
  linkBarrelLayers();
  linkEndcapLayers(theForwardLayers,theMuonForwardNLC, theTkForwardNLC);
  linkEndcapLayers(theBackwardLayers,theMuonBackwardNLC, theTkBackwardNLC);

  // establish the inwards links from Muon to Tracker
  createInverseLinks(); 
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
    theForwardLayers[mep] = MuonEtaRange(eta_max, eta_min);
  } else {
    float eta_max = calculateEta(outRadius, z+thick);
    float eta_min = calculateEta(inRadius, z-thick);
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

    BarrelDetLayer* mbp = const_cast<BarrelDetLayer*>((*bl).first);
    if (mbp->module() == dt || mbp->module() == rpc)
    theMuonBarrelNLC.push_back(new MuonBarrelNavigableLayer(
                       mbp,
                       outerBarrel, outerBackward, outerForward, allOuterBarrel,allOuterBackward,allOuterForward));

   else if(mbp->module() == pixel || mbp->module() == silicon){
      BDLC outerBarrelLayers;
      FDLC outerBackwardLayers;
      FDLC outerForwardLayers;
     for (MapBI ib = outerBarrel.begin(); ib != outerBarrel.end(); ib++) {
         BarrelDetLayer* ibdl = const_cast<BarrelDetLayer*>((*ib).first);
         outerBarrelLayers.push_back(ibdl);
        }
     for (MapEI ie = outerBackward.begin(); ie != outerBackward.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         outerBackwardLayers.push_back(ifdl);
        }
     for (MapEI ie = outerForward.begin(); ie != outerForward.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         outerForwardLayers.push_back(ifdl);
        }

   theTkBarrelNLC.push_back(new SimpleBarrelNavigableLayer(mbp, outerBarrelLayers,
                                 outerBackwardLayers,
                                 outerForwardLayers,theMagneticField, 5.));

    }
  }

}

void MuonTkNavigationSchool::linkEndcapLayers(const MapE& layers,
                                            std::vector<MuonForwardNavigableLayer*>& resultM, std::vector<SimpleForwardNavigableLayer*>& resultT) {

  for (MapEI el = layers.begin(); el != layers.end(); el++) {

    MuonEtaRange range = (*el).second;
    // first add next endcap layer (if compatible)
    MapEI plusOne(el); 
    plusOne++;
    MapB outerBLayers; //FIXME: should be filled!

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

    MapE allOuterLayers;
    for (MapEI iMEI = plusOne; iMEI!=layers.end(); iMEI++){
      if ((*iMEI).second.isCompatible(range)) allOuterLayers.insert(*iMEI);
    }
    ForwardDetLayer* mbp = const_cast<ForwardDetLayer*>((*el).first);  
    if (mbp->module() == csc || mbp->module() == rpc)
    resultM.push_back(new MuonForwardNavigableLayer(
                   mbp, outerELayers, allOuterLayers));

   else if(mbp->module() == pixel || mbp->module() == silicon){
      BDLC outerBarrelLayers;
      FDLC outerForwardLayers;
     for (MapBI ib = outerBLayers.begin(); ib != outerBLayers.end(); ib++) {
         BarrelDetLayer* ibdl = const_cast<BarrelDetLayer*>((*ib).first);
         outerBarrelLayers.push_back(ibdl);
        }
     for (MapEI ie = outerELayers.begin(); ie != outerELayers.end(); ie++) {
         ForwardDetLayer* ifdl = const_cast<ForwardDetLayer*>((*ie).first);
         outerForwardLayers.push_back(ifdl);
        }

    resultT.push_back(new SimpleForwardNavigableLayer(mbp, outerBarrelLayers,
                                 outerForwardLayers,theMagneticField, 5.));
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
    if (mbnl != 0 ) {
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
//if tracker=======================================================
   SimpleBarrelNavigableLayer* sbnl =
      dynamic_cast<SimpleBarrelNavigableLayer*>(((*bli).first)->navigableLayer());
    if (sbnl != 0 ) {
        DLC reachedLC = (*bli).first->nextLayers(alongMomentum);
        for ( DLI i = reachedLC.begin(); i != reachedLC.end(); i++) {
           const DetLayer *abdl(*i);  
           reachedBarrelLayersMap[abdl].insert(*bli);
         }
     } 
//=============================================================
  }

  // collect all reacheable layer starting from a backward layer
  for ( MapEI eli  = theBackwardLayers.begin(); 
              eli != theBackwardLayers.end(); eli++ ) {
// if muon
   MuonForwardNavigableLayer* mfnl =
      dynamic_cast<MuonForwardNavigableLayer*>(((*eli).first)->navigableLayer());
   if (mfnl != 0) {
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
//if tracker==================================================
   SimpleForwardNavigableLayer* sfnl =
      dynamic_cast<SimpleForwardNavigableLayer*>(((*eli).first)->navigableLayer());
   if (sfnl != 0) {
      DLC reachedLC = (*eli).first->nextLayers( alongMomentum);
      for ( DLI i = reachedLC.begin(); i != reachedLC.end(); i++) {
        const DetLayer * afl(*i); 
        reachedForwardLayersMap[afl].insert(*eli);
      }
    }
//============================================================
  }

  for ( MapEI eli  = theForwardLayers.begin(); 
              eli != theForwardLayers.end(); eli++ ) {

   MuonForwardNavigableLayer* mfnl =
      dynamic_cast<MuonForwardNavigableLayer*>(((*eli).first)->navigableLayer());
   if (mfnl != 0) {
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
//if tracker==================================================
   SimpleForwardNavigableLayer* sfnl =
      dynamic_cast<SimpleForwardNavigableLayer*>(((*eli).first)->navigableLayer());
   if (sfnl != 0) {
      DLC reachedLC = (*eli).first->nextLayers( alongMomentum);
      for ( DLI i = reachedLC.begin(); i != reachedLC.end(); i++) {
        const DetLayer * afl(*i);
        reachedForwardLayersMap[afl].insert( *eli);
      }
    }
//============================================================

  }

  // now set inverse link for barrel layers
  for ( MapBI bli  = theBarrelLayers.begin(); 
              bli != theBarrelLayers.end(); bli++ ) {
    MuonBarrelNavigableLayer* mbnl =
      dynamic_cast<MuonBarrelNavigableLayer*>(((*bli).first)->navigableLayer());
    if (mbnl != 0 ) {
    mbnl->setInwardLinks(reachedBarrelLayersMap[(*bli).first]);
    mbnl->setInwardCompatibleLinks(compatibleBarrelLayersMap[(*bli).first]);
    }
   
    SimpleBarrelNavigableLayer* sbnl =
      dynamic_cast<SimpleBarrelNavigableLayer*>(((*bli).first)->navigableLayer());
    if (sbnl != 0 ) {
      MapB reachedBMap= reachedBarrelLayersMap[(*bli).first];
      BDLC reachedBarrelLayers;
      FDLC reachedForwardLayers; //FIXME: fill it!
      for (MapBI ib = reachedBMap.begin(); ib != reachedBMap.end(); ib++) {
           BarrelDetLayer* ibdl = const_cast<BarrelDetLayer*>((*ib).first);
           reachedBarrelLayers.push_back(ibdl);
         }
    sbnl->setInwardLinks(reachedBarrelLayers, reachedForwardLayers);
    }
 
  }
//BACKWARD
  for ( MapEI eli  = theBackwardLayers.begin(); 
              eli != theBackwardLayers.end(); eli++ ) {
    MuonForwardNavigableLayer* mfnl =      
      dynamic_cast<MuonForwardNavigableLayer*>(((*eli).first)->navigableLayer());
    if (mfnl!= 0 ) {
    // for backward next layers
      mfnl->setInwardLinks(reachedBarrelLayersMap[(*eli).first],
                         reachedForwardLayersMap[(*eli).first]);
  // for backward compatible layers
      mfnl->setInwardCompatibleLinks(compatibleBarrelLayersMap[(*eli).first],
                         compatibleForwardLayersMap[(*eli).first]);
    }
    SimpleForwardNavigableLayer* sfnl =
      dynamic_cast<SimpleForwardNavigableLayer*>(((*eli).first)->navigableLayer());
    if (sfnl != 0 ) {
      MapB reachedBMap= reachedBarrelLayersMap[(*eli).first];
      MapE reachedEMap= reachedForwardLayersMap[(*eli).first];
      BDLC reachedBarrelLayers;
      FDLC reachedForwardLayers; 
      for (MapBI ib = reachedBMap.begin(); ib != reachedBMap.end(); ib++) {
           BarrelDetLayer* ibdl = const_cast<BarrelDetLayer*>((*ib).first);
           reachedBarrelLayers.push_back(ibdl);
         }
      for (MapEI ie = reachedEMap.begin(); ie != reachedEMap.end(); ie++) {
           ForwardDetLayer* iedl = const_cast<ForwardDetLayer*>((*ie).first);
           reachedForwardLayers.push_back(iedl);
         }

    sfnl->setInwardLinks(reachedBarrelLayers, reachedForwardLayers);
    }

  }
//FORWARD
  for ( MapEI eli  = theForwardLayers.begin(); 
              eli != theForwardLayers.end(); eli++ ) {
    MuonForwardNavigableLayer* mfnl = 
      dynamic_cast<MuonForwardNavigableLayer*>(((*eli).first)->navigableLayer());
   if (mfnl != 0 ) {
  // and for forward next layers
    mfnl->setInwardLinks(reachedBarrelLayersMap[(*eli).first],
                         reachedForwardLayersMap[(*eli).first]);
  // and for forward compatible layers
    mfnl->setInwardCompatibleLinks(compatibleBarrelLayersMap[(*eli).first],
                         compatibleForwardLayersMap[(*eli).first]);
   }
  SimpleForwardNavigableLayer* sfnl =
  dynamic_cast<SimpleForwardNavigableLayer*>(((*eli).first)->navigableLayer());
    if (sfnl != 0 ) {
      MapB reachedBMap= reachedBarrelLayersMap[(*eli).first];
      MapE reachedEMap= reachedForwardLayersMap[(*eli).first];
      BDLC reachedBarrelLayers;
      FDLC reachedForwardLayers;
      for (MapBI ib = reachedBMap.begin(); ib != reachedBMap.end(); ib++) {
           BarrelDetLayer* ibdl = const_cast<BarrelDetLayer*>((*ib).first);
           reachedBarrelLayers.push_back(ibdl);
         }
      for (MapEI ie = reachedEMap.begin(); ie != reachedEMap.end(); ie++) {
           ForwardDetLayer* iedl = const_cast<ForwardDetLayer*>((*ie).first);
           reachedForwardLayers.push_back(iedl);
         }

    sfnl->setInwardLinks(reachedBarrelLayers, reachedForwardLayers);
    }


  }

  	
}


//
// calculate pseudorapidity from r and z
//
float MuonTkNavigationSchool::calculateEta(float r, float z) const {

  if ( z > 0 ) return -log((tan(atan(r/z)/2.)));
  return log(-(tan(atan(r/z)/2.)));

}

