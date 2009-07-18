#include "TEveElement.h"
#include "TEveGeoNode.h"
#include "TEveManager.h"
#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"
#include "TEveTrack.h"
#include "TEveBoxSet.h"
#include "TEveTrackPropagator.h"
#include "TGLViewer.h"
#include "TGLUtil.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Fireworks/Core/interface/FWDetailView.h"

#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWDetailView.h"

class FWTrackDetailView : public FWDetailView<reco::Track> {

public:
   FWTrackDetailView();
   virtual ~FWTrackDetailView();

   virtual TEveElement* build (const FWModelId &id,const reco::Track*);

protected:
   void getCenter( Double_t* vars )
   {
      vars[0] = rotationCenter()[0];
      vars[1] = rotationCenter()[1];
      vars[2] = rotationCenter()[2];
   }

private:
   FWTrackDetailView(const FWTrackDetailView&); // stop default
   const FWTrackDetailView& operator=(const FWTrackDetailView&); // stop default

   // ---------- member data --------------------------------
   void resetCenter() {
      rotationCenter()[0] = 0;
      rotationCenter()[1] = 0;
      rotationCenter()[2] = 0;
   }

};


FWTrackDetailView::FWTrackDetailView ()
{

}

FWTrackDetailView::~FWTrackDetailView ()
{

}

TEveElement* FWTrackDetailView::build (const FWModelId &id, const reco::Track* iTrack)
{
   if( 0 == iTrack) { return 0; }
   const FWEventItem* item = id.item();
   // printf("calling ElectronDetailView::buildRhoZ\n");
   TEveElementList* tList =  new TEveElementList(item->name().c_str(),"Supercluster RhoZ",true);
   tList->SetMainColor(item->defaultDisplayProperties().color());

   TEveTrackPropagator* rnrStyle = new TEveTrackPropagator();
   //units are Tesla
   rnrStyle->SetMagField( -4.0);
   //get this from geometry, units are CM
   rnrStyle->SetMaxR(120.0);
   rnrStyle->SetMaxZ(300.0);

   //  Original Commented out here
   //  TEveTrackPropagator* rnrStyle = tList->GetPropagator();

   int index=0;
   //cout <<"----"<<endl;
   TEveRecTrack t;

   t.fBeta = 1.;
   t.fP = TEveVector(iTrack->px(),
                     iTrack->py(),
                     iTrack->pz());
   t.fV = TEveVector(iTrack->vx(),
                     iTrack->vy(),
                     iTrack->vz());
   t.fSign = iTrack->charge();

   TEveElementList* trkList = new TEveElementList(Form("track%d",index));
   gEve->AddElement(trkList,tList);
   TEveTrack* trk = new TEveTrack(&t,rnrStyle);
   trk->SetMainColor(item->defaultDisplayProperties().color());
   trk->MakeTrack();
   trkList->AddElement(trk);

   /* The addition of RecHits is in a try-catch block to avoid exceptions
    * occuring when TrackExtras and/or RecHits aren't available.
    */

   try {

      //      HitPattern pattern = (*it).HitPattern();

      Int_t nHits = iTrack->recHitsSize();
      // const reco::HitPattern& p = (*it).hitPattern();
      // If we have muon tracks, then this is going to be bad

      // Declare point set with size of the overall number of recHits
      TEvePointSet *recHitsMarker = new TEvePointSet(nHits);
      // Make a TEveStraightLineSet for the line implementation of recHits
      TEveStraightLineSet *recHitsLines = new TEveStraightLineSet();
      Int_t globalRecHitIndex = 0;
      Double_t localRecHitPoint[3];
      Double_t globalRecHitPoint[3];

      for(trackingRecHit_iterator recIt = iTrack->recHitsBegin(); recIt != iTrack->recHitsEnd(); ++recIt){
         if((*recIt)->isValid()){
            DetId detid = (*recIt)->geographicalId();
            LocalPoint lp = (*recIt)->localPosition();
            // localError we don't use just yet.
            //	  LocalError le = (*recIt)->localPositionError();
            // Here's the local->global transformation matrix
            const TGeoHMatrix* matrix = item->getGeom()->getMatrix( detid );
            // And here are the shapes of the hit modules
            TEveGeoShape* shape = item->getGeom()->getShape( detid );
            if(0!=shape) {
               shape->SetMainTransparency(50);
               shape->SetMainColor(tList->GetMainColor());
               trkList->AddElement(shape);
            }

            // set local position to a more digestable format
            localRecHitPoint[0] = lp.x();
            localRecHitPoint[1] = lp.y();
            localRecHitPoint[2] = 0;

            // reset global position
            globalRecHitPoint[0] = 0;
            globalRecHitPoint[1] = 0;
            globalRecHitPoint[2] = 0;


            /*  Do segments for the recHits.  Right now they're going to be of arbitrary length.
               Eventually, we'll have detector geom in here, but for now, we'll just use 5cm
               segments in the local y.  These transform nicely.
             */

            Double_t localRecHitInnerPoint[3];
            Double_t localRecHitOuterPoint[3];
            Double_t globalRecHitInnerPoint[3];
            Double_t globalRecHitOuterPoint[3];

            localRecHitInnerPoint[0] = 0;
            localRecHitInnerPoint[1] = 2.5;
            localRecHitInnerPoint[2] = 0;

            localRecHitOuterPoint[0] = 0;
            localRecHitOuterPoint[1] = -2.5;
            localRecHitOuterPoint[2] = 0;

            // Transform from local to global, if possible
            if ( matrix ) {
               matrix->LocalToMaster(localRecHitPoint, globalRecHitPoint);
               matrix->LocalToMaster(localRecHitInnerPoint, globalRecHitInnerPoint);
               matrix->LocalToMaster(localRecHitOuterPoint, globalRecHitOuterPoint);
               // Now we add the point to the TEvePointSet
               recHitsMarker->SetPoint(globalRecHitIndex, globalRecHitPoint[0],
                                       globalRecHitPoint[1], globalRecHitPoint[2]);
               recHitsLines->AddLine(globalRecHitInnerPoint[0], globalRecHitInnerPoint[1], globalRecHitInnerPoint[2],
                                     globalRecHitOuterPoint[0], globalRecHitOuterPoint[1], globalRecHitOuterPoint[2]);
               globalRecHitIndex++;
            }

         } // if the hit isValid().
      } // For Loop Over Rec hits (recIt)

      // This saved just in case we want to use points for recHits.
      //recHitsMarker->SetMainColor(iItem->defaultDisplayProperties().color());
      //recHitsMarker->SetMarkerSize(1);
      //trkList->AddElement(recHitsMarker);

      recHitsLines->SetMainColor(item->defaultDisplayProperties().color());
      trkList->AddElement(recHitsLines);
   }
   catch (...) {
      //      std::cout << "Sorry, don't have the recHits for this event." << std::endl;
   }

   viewer()->SetGuideState(TGLUtil::kAxesOrigin, kTRUE, kFALSE, 0);
   return tList;
}

REGISTER_FWDETAILVIEW(FWTrackDetailView);

