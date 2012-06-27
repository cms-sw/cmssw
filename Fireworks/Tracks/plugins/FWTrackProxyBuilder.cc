// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWTrackProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 25 14:42:13 EST 2008
// $Id: FWTrackProxyBuilder.cc,v 1.13 2010/11/11 20:25:29 amraktad Exp $
//

// system include files
#include "TEveTrack.h"
#include "TEveGeoShape.h"
#include "TEveVector.h"
#include "TEveStraightLineSet.h"
#include "TEveManager.h"
// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/estimate_field.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/fwLog.h"


#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"


class FWTrackProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track> {
   
public:
   FWTrackProxyBuilder();
   virtual ~FWTrackProxyBuilder();
   
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWTrackProxyBuilder(const FWTrackProxyBuilder&); // stop default
   
   const FWTrackProxyBuilder& operator=(const FWTrackProxyBuilder&); // stop default
   
   void build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*);
   void addRecHitInfo(const reco::Track& iData, TEveElement& oItemHolder, bool drawRecHits, TEveTrack* trk) ;
   
   virtual void setItem(const FWEventItem* iItem)
   {
      FWProxyBuilderBase::setItem(iItem);
      if (iItem)
         iItem->getConfig()->assertParam("Fit RecHits", false);
         iItem->getConfig()->assertParam("Draw RecHits", false);
   }
};

FWTrackProxyBuilder::FWTrackProxyBuilder()
{
}

FWTrackProxyBuilder::~FWTrackProxyBuilder()
{
}

void
FWTrackProxyBuilder::build( const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder , const FWViewContext*) 
{
   // printf("Track [%d]========================================= \n", iIndex);
   
   if( context().getField()->getSource() == FWMagField::kNone ) {
      if( fabs( iData.eta() ) < 2.0 && iData.pt() > 0.5 && iData.pt() < 30 ) {
         double estimate = fw::estimate_field( iData, true );
         if( estimate >= 0 ) context().getField()->guessField( estimate );
      }
   }
   
   TEveTrack* trk;
   bool fitRecHits = item()->getConfig()->value<bool>("Fit RecHits");
   bool drawRecHits = item()->getConfig()->value<bool>("Draw RecHits");
   
   if (fitRecHits)
   {
      TEveRecTrack t;
      t.fBeta = 1.;
      t.fSign = iData.charge();
      t.fP.Set(iData.px(), iData.py(), iData.pz());
      t.fV.Set(iData.vx(), iData.vy(), iData.vz());
      // printf("rc->fSign = %d; \n", t.fSign );
      // printf("rc->fP.Set( %f, %f, %f); \n",  iData.px(), iData.py(), iData.pz());
      // printf("rc->fV.Set(%f, %f, %f); \n",  iData.vx(), iData.vy(), iData.vz());
      
      trk = new TEveTrack( &t, context().getTrackerTrackPropagator()); 
   }
   else
   {
      TEveTrackPropagator* propagator = ( !iData.extra().isAvailable() ) ?  context().getTrackerTrackPropagator() : context().getTrackPropagator();
      trk = fireworks::prepareTrack( iData, propagator );
   }
   
   if (drawRecHits || fitRecHits)
      addRecHitInfo(iData, oItemHolder,  drawRecHits, fitRecHits ? trk :  0);  
   
   trk->MakeTrack();
   setupAddElement(trk, &oItemHolder);
}

//------------------------------------------------------------------
void FWTrackProxyBuilder::addRecHitInfo(const reco::Track& iData, TEveElement& oItemHolder, bool drawRecHits, TEveTrack* trk) 
{
   TEveStraightLineSet *scposition = 0;
   TEvePointSet* pointSet = 0;
   
   if (drawRecHits)
   {
      pointSet = new TEvePointSet();   
      scposition =  new TEveStraightLineSet;   
   }

   for( trackingRecHit_iterator it = iData.recHitsBegin(), itEnd = iData.recHitsEnd(); it != itEnd; ++it )
   {
      unsigned int rawid = (*it)->geographicalId();      
      unsigned int subdet = (unsigned int)(*it)->geographicalId().subdetId();
      const float* pars = item()->getGeom()->getParameters( rawid );
      
      // pixel
      if( ( subdet == PixelSubdetector::PixelBarrel ) || ( subdet == PixelSubdetector::PixelEndcap ))
      {
         if( const SiPixelRecHit* pixel = dynamic_cast<const SiPixelRecHit*>( &(**it)))
         {
            const SiPixelCluster& c = *( pixel->cluster());
            
            double row = c.minPixelRow();
            double col = c.minPixelCol();
            float lx = 0.;
            float ly = 0.;
            
            int nrows = (int)pars[0];
            int ncols = (int)pars[1];
            lx = fireworks::pixelLocalX( row, nrows );
            ly = fireworks::pixelLocalY( col, ncols );
            
            fwLog( fwlog::kDebug )
            << ", row: " << row << ", col: " << col 
            << ", lx: " << lx << ", ly: " << ly ;
				
            float local[3] = { lx, ly, 0. };
            float global[3];
            item()->getGeom()->localToGlobal( rawid, local, global );
            
            if (drawRecHits) pointSet->SetNextPoint(global[0], global[1], global[2]);
            if (trk)
            {
               trk->AddPathMark( TEvePathMark( TEvePathMark::kDaughter, TEveVector(global[0], global[1], global[2])));            
            // printf("addPixeHit(%f, %f, %f)\n", global[0], global[1], global[2]);
            }
         }
      }
      const SiStripCluster *cluster = fireworks::extractClusterFromTrackingRecHit( &(*(*it)) );
      if( cluster )
      {
         short firststrip = cluster->firstStrip();
         float localTop[3] = { 0.0, 0.0, 0.0 };
         float localBottom[3] = { 0.0, 0.0, 0.0 };
         fireworks::localSiStrip( firststrip, localTop, localBottom, pars, rawid );
         float globalTop[3];
         float globalBottom[3];
         item()->getGeom()->localToGlobal( rawid, localTop, globalTop, localBottom, globalBottom );
         
         if (drawRecHits) scposition->AddLine( globalTop[0], globalTop[1], globalTop[2],globalBottom[0], globalBottom[1], globalBottom[2] );
                             
         if (trk) {
            TEveVectorD stripVec( -globalTop[0] +globalBottom[0], -globalTop[1] +globalBottom[1],  -globalTop[2] +globalBottom[2]);
            TEveVectorD stripPos(globalTop[0], globalTop[1],  globalTop[2]);
            TEveVectorD p;
            trk->AddPathMark( TEvePathMark( TEvePathMark::kLineSegment, stripPos, p, stripVec));
            // printf("addStripMarker(track, lineset, %f, %f, %f,  %f, %f, %f);\n ", stripPos.fX, stripPos.fY, stripPos.fZ, stripVec.fX, stripVec.fY, stripVec.fZ);        
         }
      }
   }
   
   if (drawRecHits) 
   {
      setupAddElement(pointSet, &oItemHolder);
      setupAddElement(scposition, &oItemHolder);     
      scposition->SetMainColor(kRed);
      pointSet->SetMainColor(kRed);
      pointSet->SetMarkerStyle(2);
      scposition->SetPickable(false);
   }
   
   // debug with Eve
   if (0) {
      trk->SetRnrPoints(true);
      gEve->AddToListTree(trk, true);   
   }
}

REGISTER_FWPROXYBUILDER(FWTrackProxyBuilder, reco::Track, "Tracks", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);

/* 
 
 if( iData.extra().isAvailable() )
 {
 if (iData.innerOk()) {
 const reco::TrackBase::Point  &v = iData.innerPosition();
 float perpD = TMath::Sqrt((v.x() - iData.vx())*(v.x() - iData.vx()) + (v.y()  - iData.vy())*(v.y()  - iData.vy()));
 printf("tracl inner (%f, %f, %f ) R: %f\n",  v.x(), v.y(), v.z(), perpD);
 }
 if (iData.outerOk()) {
 const reco::TrackBase::Point  &v = iData.outerPosition();
 
 float perpD = TMath::Sqrt((v.x()  - iData.vx())*(v.x()  - iData.vx()) + (v.y()  - iData.vy())*(v.y()  - iData.vy()));
 printf("tracl outer (%f, %f, %f ) R:%f\n",  v.x(), v.y(), v.z(), perpD);
 }
 }
 */   


// need shape for plane normal
/*
 TEveGeoShape* shape = item()->getGeom()->getEveShape( rawid );
 setupAddElement(shape,  &oItemHolder);
 shape->SetMainTransparency(90);
 
 TEveTrans& trans = shape->RefMainTrans();
 TEveVector stripNormal(trans.GetBaseVec(3).x(), trans.GetBaseVec(3).y(), trans.GetBaseVec(3).z());
 */

