// -*- C++ -*-
//
// Package:     Tracks
// Class  :     TrackUtils
//

// system include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveStraightLineSet.h"
#include "TEveVSDStructs.h"
#include "TEveGeoNode.h"

// user include files
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "FWCore/Common/interface/EventBase.h"

namespace fireworks {

static const double MICRON = 1./1000./10.;
static const double PITCHX = 100*MICRON;
static const double PITCHY = 150*MICRON;
static const int BIG_PIX_PER_ROC_Y = 2;  // in y direction, cols
static const int COLS_PER_ROC      = 52; // Num of Rows per ROC
static const int ROWS_PER_ROC      = 80; // Num of cols per ROC
static const int BIG_PIX_PER_ROC_X = 1;  // in x direction, rows

// -- Si module names for printout
static const std::string subdets[7] = { "UNKNOWN", "PXB", "PXF", "TIB", "TID", "TOB", "TEC" };

TEveTrack*
prepareTrack( const reco::Track& track,
              TEveTrackPropagator* propagator,
              const std::vector<TEveVector>& extraRefPoints )
{
   // To make use of all available information, we have to order states
   // properly first. Propagator should take care of y=0 transition.

   std::vector<State> refStates;
   TEveVector trackMomentum(track.px(), track.py(), track.pz());
   refStates.push_back(State(TEveVector(track.vx(), track.vy(), track.vz()),
                             trackMomentum));
   if( track.extra().isAvailable() ) {
      if (track.innerOk()) {
         const reco::TrackBase::Point  &v = track.innerPosition();
         const reco::TrackBase::Vector &p = track.innerMomentum();
         refStates.push_back(State(TEveVector(v.x(), v.y(), v.z()), TEveVector(p.x(), p.y(), p.z())));
      }
      if (track.outerOk()) {
         const reco::TrackBase::Point  &v = track.outerPosition();
         const reco::TrackBase::Vector &p = track.outerMomentum();
         refStates.push_back(State(TEveVector(v.x(), v.y(), v.z()), TEveVector(p.x(), p.y(), p.z())));
      }
   }
   for( std::vector<TEveVector>::const_iterator point = extraRefPoints.begin(), pointEnd = extraRefPoints.end();
        point != pointEnd; ++point )
      refStates.push_back(State(*point));
   if( track.pt()>1 )
      std::sort( refStates.begin(), refStates.end(), StateOrdering(trackMomentum) );

   // * if the first state has non-zero momentum use it as a starting point
   //   and all other points as PathMarks to follow
   // * if the first state has only position, try the last state. If it has
   //   momentum we propagate backword, if not, we look for the first one
   //   on left that has momentum and ignore all earlier.
   //

   TEveRecTrack t;
   t.fBeta = 1.;
   t.fSign = track.charge();

   if( refStates.front().valid ) {
      t.fV = refStates.front().position;
      t.fP = refStates.front().momentum;
      TEveTrack* trk = new TEveTrack( &t, propagator );
      for( unsigned int i(1); i<refStates.size()-1; ++i) {
         if( refStates[i].valid )
            trk->AddPathMark( TEvePathMark( TEvePathMark::kReference, refStates[i].position, refStates[i].momentum ) );
         else
            trk->AddPathMark( TEvePathMark( TEvePathMark::kDaughter, refStates[i].position ) );
      }
      if( refStates.size()>1 ) {
         trk->AddPathMark( TEvePathMark( TEvePathMark::kDecay, refStates.back().position ) );
      }
      return trk;
   }

   if( refStates.back().valid ) {
      t.fSign = (-1)*track.charge();
      t.fV = refStates.back().position;
      t.fP = refStates.back().momentum * (-1.0f);
      TEveTrack* trk = new TEveTrack( &t, propagator );
      unsigned int i( refStates.size()-1 );
      for(; i>0; --i) {
         if ( refStates[i].valid )
            trk->AddPathMark( TEvePathMark( TEvePathMark::kReference, refStates[i].position, refStates[i].momentum*(-1.0f) ) );
         else
            trk->AddPathMark( TEvePathMark( TEvePathMark::kDaughter, refStates[i].position ) );
      }
      if ( refStates.size()>1 ) {
         trk->AddPathMark( TEvePathMark( TEvePathMark::kDecay, refStates.front().position ) );
      }
      return trk;
   }

   unsigned int i(0);
   while( i<refStates.size() && !refStates[i].valid ) ++i;
   assert( i < refStates.size() );

   t.fV = refStates[i].position;
   t.fP = refStates[i].momentum;
   TEveTrack* trk = new TEveTrack( &t, propagator );
   for( unsigned int j(i+1); j<refStates.size()-1; ++j ) {
      if( refStates[i].valid )
         trk->AddPathMark( TEvePathMark( TEvePathMark::kReference, refStates[i].position, refStates[i].momentum ) );
      else
         trk->AddPathMark( TEvePathMark( TEvePathMark::kDaughter, refStates[i].position ) );
   }
   if ( i < refStates.size() ) {
      trk->AddPathMark( TEvePathMark( TEvePathMark::kDecay, refStates.back().position ) );
   }
   return trk;
}

// Transform measurement to local coordinates in X dimension
float
pixelLocalX( const double mpx, const int nrows )
{
  // Calculate the edge of the active sensor with respect to the center,
  // that is simply the half-size.       
  // Take into account large pixels
  const double xoffset = -( nrows + BIG_PIX_PER_ROC_X * nrows / ROWS_PER_ROC ) / 2. * PITCHX;

  // Measurement to local transformation for X coordinate
  // X coordinate is in the ROC row number direction
  // Copy from RectangularPixelTopology::localX implementation
   int binoffx = int(mpx);             // truncate to int
   double fractionX = mpx - binoffx;   // find the fraction
   double local_PITCHX = PITCHX;       // defaultpitch
   if( binoffx > 80 ) {                // ROC 1 - handles x on edge cluster
      binoffx = binoffx + 2;
   } else if( binoffx == 80 ) {        // ROC 1
      binoffx = binoffx+1;
      local_PITCHX = 2 * PITCHX;
   } else if( binoffx == 79 ) {        // ROC 0
      binoffx = binoffx + 0;
      local_PITCHX = 2 * PITCHX;
   } else if( binoffx >= 0 ) {         // ROC 0
      binoffx = binoffx + 0;
   }

   // The final position in local coordinates
   double lpX = double( binoffx * PITCHX ) + fractionX * local_PITCHX + xoffset;

   return lpX;
}

// Transform measurement to local coordinates in Y dimension
float
pixelLocalY( const double mpy, const int ncols )
{
  // Calculate the edge of the active sensor with respect to the center,
  // that is simply the half-size.       
  // Take into account large pixels
  double yoffset = -( ncols + BIG_PIX_PER_ROC_Y * ncols / COLS_PER_ROC ) / 2. * PITCHY;

  // Measurement to local transformation for Y coordinate
  // Y is in the ROC column number direction
  // Copy from RectangularPixelTopology::localY implementation
  int binoffy = int( mpy );           // truncate to int
  double fractionY = mpy - binoffy;   // find the fraction
  double local_PITCHY = PITCHY;       // defaultpitch

  if( binoffy>416 ) {                 // ROC 8, not real ROC
    binoffy = binoffy+17;
  } else if( binoffy == 416 ) {       // ROC 8
    binoffy = binoffy + 16;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy == 415 ) {       // ROC 7, last big pixel
    binoffy = binoffy + 15;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy > 364 ) {        // ROC 7
    binoffy = binoffy + 15;
  } else if( binoffy == 364 ) {       // ROC 7
    binoffy = binoffy + 14;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy == 363 ) {       // ROC 6
    binoffy = binoffy + 13;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy > 312 ) {        // ROC 6
    binoffy = binoffy + 13;
  } else if( binoffy == 312 ) {       // ROC 6
    binoffy = binoffy + 12;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy == 311 ) {       // ROC 5
    binoffy = binoffy + 11;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy > 260 ) {        // ROC 5
    binoffy = binoffy + 11;
  } else if( binoffy == 260 ) {       // ROC 5
    binoffy = binoffy + 10;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy == 259 ) {       // ROC 4
    binoffy = binoffy + 9;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy > 208 ) {        // ROC 4
    binoffy = binoffy + 9;
  } else if(binoffy == 208 ) {        // ROC 4
    binoffy = binoffy + 8;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy == 207 ) {       // ROC 3
    binoffy = binoffy + 7;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy > 156 ) {        // ROC 3
    binoffy = binoffy + 7;
  } else if( binoffy == 156 ) {       // ROC 3
    binoffy = binoffy + 6;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy == 155 ) {       // ROC 2
    binoffy = binoffy + 5;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy > 104 ) {        // ROC 2
    binoffy = binoffy + 5;
  } else if( binoffy == 104 ) {       // ROC 2
    binoffy = binoffy + 4;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy == 103 ) {       // ROC 1
    binoffy = binoffy + 3;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy > 52 ) {         // ROC 1
    binoffy = binoffy + 3;
  } else if( binoffy == 52 ) {        // ROC 1
    binoffy = binoffy + 2;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy == 51 ) {        // ROC 0
    binoffy = binoffy + 1;
    local_PITCHY = 2 * PITCHY;
  } else if( binoffy > 0 ) {          // ROC 0
    binoffy=binoffy + 1;
  } else if( binoffy == 0 ) {         // ROC 0
    binoffy = binoffy + 0;
    local_PITCHY = 2 * PITCHY;
  }

  // The final position in local coordinates
  double lpY = double( binoffy * PITCHY ) + fractionY * local_PITCHY + yoffset;

  return lpY;
}

//
// Returns strip geometry in local coordinates of a detunit.
// The strip is a line from a localTop to a localBottom point.
void localSiStrip( short strip, float* localTop, float* localBottom, const float* pars, unsigned int id )
{
  Float_t topology = pars[0];
  Float_t halfStripLength = pars[2] * 0.5;
  
  Double_t localCenter[3] = { 0.0, 0.0, 0.0 };
  localTop[1] = halfStripLength;
  localBottom[1] = -halfStripLength;
  
  if( topology == 1 ) // RadialStripTopology
  {
    // stripAngle = phiOfOneEdge + strip * angularWidth
    // localY = originToIntersection * tan( stripAngle )
    Float_t stripAngle = tan( pars[5] + strip * pars[6] );
    Float_t delta = halfStripLength * stripAngle;
    localCenter[0] = pars[4] * stripAngle;
    localTop[0] = localCenter[0] + delta;
    localBottom[0] = localCenter[0] - delta;
  }
  else if( topology == 2 ) // RectangularStripTopology
  {
    // offset = -numberOfStrips/2. * pitch
    // localY = strip * pitch + offset
    Float_t offset = -pars[1] * 0.5 * pars[3];
    localCenter[0] = strip * pars[3] + offset;
    localTop[0] = localCenter[0];
    localBottom[0] = localCenter[0];
  }
  else if( topology == 3 ) // TrapezoidalStripTopology
  {
    fwLog( fwlog::kError )
      << "did not expect TrapezoidalStripTopology of "
      << id << std::endl;
  }
  else if( pars[0] == 0 ) // StripTopology
  {
    fwLog( fwlog::kError )
      << "did not find StripTopology of "
      << id << std::endl;
  }
}

//______________________________________________________________________________

void
setupAddElement(TEveElement* el, TEveElement* parent, const FWEventItem* item, bool master, bool color)
{
   if (master)
   {
      el->CSCTakeAnyParentAsMaster();
      el->SetPickable(true);
   }

   if (color)
   {
      el->CSCApplyMainColorToMatchingChildren();
      el->CSCApplyMainTransparencyToMatchingChildren();
      el->SetMainColor(item->defaultDisplayProperties().color());
      assert((item->defaultDisplayProperties().transparency() >= 0)
             && (item->defaultDisplayProperties().transparency() <= 100));
      el->SetMainTransparency(item->defaultDisplayProperties().transparency());
   }
   parent->AddElement(el);
}

//______________________________________________________________________________

const SiStripCluster* extractClusterFromTrackingRecHit( const TrackingRecHit* rechit )
{
   const SiStripCluster* cluster = 0;

   if( const SiStripRecHit2D* hit2D = dynamic_cast<const SiStripRecHit2D*>( rechit ))
   {     
      fwLog( fwlog::kDebug ) << "hit 2D ";
      
	 cluster = hit2D->cluster().get();
   }
   if( cluster == 0 )
   {
     if( const SiStripRecHit1D* hit1D = dynamic_cast<const SiStripRecHit1D*>( rechit ))
     {
        fwLog( fwlog::kDebug ) << "hit 1D ";

	   cluster = hit1D->cluster().get();
     }
   }
   return cluster;
}

void
addSiStripClusters( const FWEventItem* iItem, const reco::Track &t, class TEveElement *tList, bool addNearbyClusters, bool master ) 
{
   // master is true if the product is for proxy builder
   const FWGeometry *geom = iItem->getGeom();

   const edmNew::DetSetVector<SiStripCluster> * allClusters = 0;
   if( addNearbyClusters )
   {
      for( trackingRecHit_iterator it = t.recHitsBegin(), itEnd = t.recHitsEnd(); it != itEnd; ++it )
      {
         if( typeid( **it ) == typeid( SiStripRecHit2D ))
         {
            const SiStripRecHit2D &hit = static_cast<const SiStripRecHit2D &>( **it );
            if( hit.cluster().isNonnull() && hit.cluster().isAvailable())
	    {
               edm::Handle<edmNew::DetSetVector<SiStripCluster> > allClustersHandle;
               iItem->getEvent()->get(hit.cluster().id(), allClustersHandle);
               allClusters = allClustersHandle.product();
               break;
            }
         }
         else if( typeid( **it ) == typeid( SiStripRecHit1D ))
         {
            const SiStripRecHit1D &hit = static_cast<const SiStripRecHit1D &>( **it );
            if( hit.cluster().isNonnull() && hit.cluster().isAvailable())
	    {
               edm::Handle<edmNew::DetSetVector<SiStripCluster> > allClustersHandle;
               iItem->getEvent()->get(hit.cluster().id(), allClustersHandle);
               allClusters = allClustersHandle.product();
               break;
            }
         }
      }
   }

   for( trackingRecHit_iterator it = t.recHitsBegin(), itEnd = t.recHitsEnd(); it != itEnd; ++it )
   {
      unsigned int rawid = (*it)->geographicalId();
      if( ! geom->contains( rawid ))
      {
	 fwLog( fwlog::kError )
	   << "failed to get geometry of SiStripCluster with detid: " 
	   << rawid << std::endl;
	 
	 continue;
      }
	
      const float* pars = geom->getParameters( rawid );
      
      // -- get phi from SiStripHit
      auto rechitRef = *it;
      const TrackingRecHit *rechit = &( *rechitRef );
      const SiStripCluster *cluster = extractClusterFromTrackingRecHit( rechit );

      if( cluster )
      {
         if( allClusters != 0 )
         {
            const edmNew::DetSet<SiStripCluster> & clustersOnThisDet = (*allClusters)[rechit->geographicalId().rawId()];

            for( edmNew::DetSet<SiStripCluster>::const_iterator itc = clustersOnThisDet.begin(), edc = clustersOnThisDet.end(); itc != edc; ++itc )
            {

               TEveStraightLineSet *scposition = new TEveStraightLineSet;
               scposition->SetDepthTest( false );
	       scposition->SetPickable( kTRUE );

	       short firststrip = itc->firstStrip();

	       if( &*itc == cluster )
	       {
		  scposition->SetTitle( Form( "Exact SiStripCluster from TrackingRecHit, first strip %d", firststrip ));
		  scposition->SetLineColor( kGreen );
	       }
	       else
	       {
		  scposition->SetTitle( Form( "SiStripCluster, first strip %d", firststrip ));
		  scposition->SetLineColor( kRed );
	       }
	       
	       float localTop[3] = { 0.0, 0.0, 0.0 };
	       float localBottom[3] = { 0.0, 0.0, 0.0 };

	       fireworks::localSiStrip( firststrip, localTop, localBottom, pars, rawid );

	       float globalTop[3];
	       float globalBottom[3];
	       geom->localToGlobal( rawid, localTop, globalTop, localBottom, globalBottom );
  
	       scposition->AddLine( globalTop[0], globalTop[1], globalTop[2],
				    globalBottom[0], globalBottom[1], globalBottom[2] );
	       
	       setupAddElement( scposition, tList, iItem, master, false );
            }
         }
         else
         {
	    short firststrip = cluster->firstStrip();
	    TEveStraightLineSet *scposition = new TEveStraightLineSet;
            scposition->SetDepthTest( false );
	    scposition->SetPickable( kTRUE );
	    scposition->SetTitle( Form( "SiStripCluster, first strip %d", firststrip ));
	    
	    float localTop[3] = { 0.0, 0.0, 0.0 };
	    float localBottom[3] = { 0.0, 0.0, 0.0 };

	    fireworks::localSiStrip( firststrip, localTop, localBottom, pars, rawid );

	    float globalTop[3];
	    float globalBottom[3];
	    geom->localToGlobal( rawid, localTop, globalTop, localBottom, globalBottom );
  
	    scposition->AddLine( globalTop[0], globalTop[1], globalTop[2],
				 globalBottom[0], globalBottom[1], globalBottom[2] );
	       
            setupAddElement( scposition, tList, iItem, master, true );
         }		
      }
      else if( !rechit->isValid() && ( rawid != 0 )) // lost hit
      {
         if( allClusters != 0 )
	 {
            edmNew::DetSetVector<SiStripCluster>::const_iterator itds = allClusters->find( rawid );
            if( itds != allClusters->end())
            {
               const edmNew::DetSet<SiStripCluster> & clustersOnThisDet = *itds;
               for( edmNew::DetSet<SiStripCluster>::const_iterator itc = clustersOnThisDet.begin(), edc = clustersOnThisDet.end(); itc != edc; ++itc )
               {
		  short firststrip = itc->firstStrip();

                  TEveStraightLineSet *scposition = new TEveStraightLineSet;
                  scposition->SetDepthTest( false );
		  scposition->SetPickable( kTRUE );
		  scposition->SetTitle( Form( "Lost SiStripCluster, first strip %d", firststrip ));

		  float localTop[3] = { 0.0, 0.0, 0.0 };
		  float localBottom[3] = { 0.0, 0.0, 0.0 };

		  fireworks::localSiStrip( firststrip, localTop, localBottom, pars, rawid );

		  float globalTop[3];
		  float globalBottom[3];
		  geom->localToGlobal( rawid, localTop, globalTop, localBottom, globalBottom );
  
		  scposition->AddLine( globalTop[0], globalTop[1], globalTop[2],
				       globalBottom[0], globalBottom[1], globalBottom[2] );


                  setupAddElement( scposition, tList, iItem, master, false );
                  scposition->SetLineColor( kRed );
               }
            }
         }
      }
      else
      {
	 fwLog( fwlog::kDebug )
	    << "*ANOTHER* option possible: valid=" << rechit->isValid()
	    << ", rawid=" << rawid << std::endl;
      }
   }
}

//______________________________________________________________________________

void
pushNearbyPixelHits( std::vector<TVector3> &pixelPoints, const FWEventItem &iItem, const reco::Track &t )
{
   const edmNew::DetSetVector<SiPixelCluster> * allClusters = 0;
   for( trackingRecHit_iterator it = t.recHitsBegin(), itEnd = t.recHitsEnd(); it != itEnd; ++it)
   {
      if( typeid(**it) == typeid( SiPixelRecHit ))
      {
         const SiPixelRecHit &hit = static_cast<const SiPixelRecHit &>(**it);
         if( hit.cluster().isNonnull() && hit.cluster().isAvailable())
	 {
            edm::Handle<edmNew::DetSetVector<SiPixelCluster> > allClustersHandle;
            iItem.getEvent()->get(hit.cluster().id(), allClustersHandle);
            allClusters = allClustersHandle.product();
            break;
	 }
      }
   }
   if( allClusters == 0 ) return;

   const FWGeometry *geom = iItem.getGeom();

   for( trackingRecHit_iterator it = t.recHitsBegin(), itEnd = t.recHitsEnd(); it != itEnd; ++it )
   {
      const TrackingRecHit* rh = &(**it);

      DetId id = (*it)->geographicalId();
      if( ! geom->contains( id ))
      {
	 fwLog( fwlog::kError )
	    << "failed to get geometry of Tracker Det with raw id: " 
	    << id.rawId() << std::endl;

	continue;
      }

      // -- in which detector are we?
      unsigned int subdet = (unsigned int)id.subdetId();
      if(( subdet != PixelSubdetector::PixelBarrel ) && ( subdet != PixelSubdetector::PixelEndcap )) continue;

      const SiPixelCluster* hitCluster = 0;
      if( const SiPixelRecHit* pixel = dynamic_cast<const SiPixelRecHit*>( rh ))
	 hitCluster = pixel->cluster().get();
      edmNew::DetSetVector<SiPixelCluster>::const_iterator itds = allClusters->find(id.rawId());
      if( itds != allClusters->end())
      {
         const edmNew::DetSet<SiPixelCluster> & clustersOnThisDet = *itds;
	 for( edmNew::DetSet<SiPixelCluster>::const_iterator itc = clustersOnThisDet.begin(), edc = clustersOnThisDet.end(); itc != edc; ++itc )
	 {
	   if( &*itc != hitCluster )
	      pushPixelCluster( pixelPoints, *geom, id, *itc, geom->getParameters( id ));
         }
      }
   }
}

//______________________________________________________________________________

void
pushPixelHits( std::vector<TVector3> &pixelPoints, const FWEventItem &iItem, const reco::Track &t )
{		
   /*
    * -- return for each Pixel Hit a 3D point
    */
   const FWGeometry *geom = iItem.getGeom();
   
   double dz = t.dz();
   double vz = t.vz();
   double etaT = t.eta();
   
   fwLog( fwlog::kDebug ) << "Track eta: " << etaT << ", vz: " << vz << ", dz: " << dz
			  << std::endl;
		
   int cnt = 0;
   for( trackingRecHit_iterator it = t.recHitsBegin(), itEnd = t.recHitsEnd(); it != itEnd; ++it )
   {
      const TrackingRecHit* rh = &(**it);			
      // -- get position of center of wafer, assuming (0,0,0) is the center
      DetId id = (*it)->geographicalId();
      if( ! geom->contains( id ))
      {
	 fwLog( fwlog::kError )
	    << "failed to get geometry of Tracker Det with raw id: " 
	    << id.rawId() << std::endl;

	continue;
      }

      // -- in which detector are we?			
      unsigned int subdet = (unsigned int)id.subdetId();
			
      if(( subdet == PixelSubdetector::PixelBarrel ) || ( subdet == PixelSubdetector::PixelEndcap ))
      {
	 fwLog( fwlog::kDebug ) << cnt++ << " -- "
				<< subdets[subdet];
								
         if( const SiPixelRecHit* pixel = dynamic_cast<const SiPixelRecHit*>( rh ))
	 {
	    const SiPixelCluster& c = *( pixel->cluster());
	    pushPixelCluster( pixelPoints, *geom, id, c, geom->getParameters( id ));
	 } 
      }
   }
}
  
void
pushPixelCluster( std::vector<TVector3> &pixelPoints, const FWGeometry &geom, DetId id, const SiPixelCluster &c, const float* pars )
{
   double row = c.minPixelRow();
   double col = c.minPixelCol();
   float lx = 0.;
   float ly = 0.;
   
   int nrows = (int)pars[0];
   int ncols = (int)pars[1];
   lx = pixelLocalX( row, nrows );
   ly = pixelLocalY( col, ncols );

   fwLog( fwlog::kDebug )
      << ", row: " << row << ", col: " << col 
      << ", lx: " << lx << ", ly: " << ly ;
				
   float local[3] = { lx, ly, 0. };
   float global[3];
   geom.localToGlobal( id, local, global );
   TVector3 pb( global[0], global[1], global[2] );
   pixelPoints.push_back( pb );
				
   fwLog( fwlog::kDebug )
      << " x: " << pb.X()
      << ", y: " << pb.Y()
      << " z: " << pb.Z()
      << " eta: " << pb.Eta()
      << ", phi: " << pb.Phi()
      << " rho: " << pb.Pt() << std::endl;
}
	
//______________________________________________________________________________

std::string
info(const DetId& id) {
   std::ostringstream oss;
   
   oss << "DetId: " << id.rawId() << "\n";
   
   switch ( id.det() ) {
    
      case DetId::Tracker:
         switch ( id.subdetId() ) {
            case StripSubdetector::TIB:
	    {
	       oss <<"TIB "<<TIBDetId(id).layer();
	    }
	    break;
            case StripSubdetector::TOB:
	    {
	       oss <<"TOB "<<TOBDetId(id).layer();
	    }
	    break;
            case StripSubdetector::TEC:
	    {
	       oss <<"TEC "<<TECDetId(id).wheel();
	    }
	    break;
            case StripSubdetector::TID:
	    {
	       oss <<"TID "<<TIDDetId(id).wheel();
	    }
	    break;
            case (int) PixelSubdetector::PixelBarrel:
	    {
	       oss <<"PixBarrel "<< PXBDetId(id).layer();
	    }
	    break;
            case (int) PixelSubdetector::PixelEndcap:
	    {
	       oss <<"PixEndcap "<< PXBDetId(id).layer();
	    }
	    break;
	 }
	 break;

      case DetId::Muon:
         switch ( id.subdetId() ) {
            case MuonSubdetId::DT:
	    { 
	       DTChamberId detId(id.rawId());
	       oss << "DT chamber (wheel, station, sector): "
		   << detId.wheel() << ", "
		   << detId.station() << ", "
		   << detId.sector();
	    }
	    break;
            case MuonSubdetId::CSC:
	    {
	       CSCDetId detId(id.rawId());
	       oss << "CSC chamber (endcap, station, ring, chamber, layer): "
		   << detId.endcap() << ", "
		   << detId.station() << ", "
		   << detId.ring() << ", "
		   << detId.chamber() << ", "
		   << detId.layer();
	    }
	    break;
            case MuonSubdetId::RPC:
	    { 
	       RPCDetId detId(id.rawId());
	       oss << "RPC chamber ";
	       switch ( detId.region() ) {
                  case 0:
                     oss << "/ barrel / (wheel, station, sector, layer, subsector, roll): "
                         << detId.ring() << ", "
                         << detId.station() << ", "
                         << detId.sector() << ", "
                         << detId.layer() << ", "
                         << detId.subsector() << ", "
                         << detId.roll();
                     break;
                  case 1:
                     oss << "/ forward endcap / (wheel, station, sector, layer, subsector, roll): "
                         << detId.ring() << ", "
                         << detId.station() << ", "
                         << detId.sector() << ", "
                         << detId.layer() << ", "
                         << detId.subsector() << ", "
                         << detId.roll();
                     break;
                  case -1:
                     oss << "/ backward endcap / (wheel, station, sector, layer, subsector, roll): "
                         << detId.ring() << ", "
                         << detId.station() << ", "
                         << detId.sector() << ", "
                         << detId.layer() << ", "
                         << detId.subsector() << ", "
                         << detId.roll();
                     break;
	       }
	    }
	    break;
            case MuonSubdetId::GEM:
	    {
	       GEMDetId detId(id.rawId());
	       oss << "GEM chamber (region, station, ring, chamber, layer): "
		   << detId.region() << ", "
		   << detId.station() << ", "
		   << detId.ring() << ", "
		   << detId.chamber() << ", "
		   << detId.layer();
	    }
	    break;
            case MuonSubdetId::ME0:
	    {
	       ME0DetId detId(id.rawId());
	       oss << "ME0 chamber (region, chamber, layer): "
		   << detId.region() << ", "
		   << detId.chamber() << ", "
		   << detId.layer();
	    }
	    break;
	 }
	 break;
    
      case DetId::Calo:
      {
         CaloTowerDetId detId( id.rawId() );
         oss << "CaloTower (ieta, iphi): "
             << detId.ieta() << ", "
             << detId.iphi();
      }
      break;
    
      case DetId::Ecal:
         switch ( id.subdetId() ) {
            case EcalBarrel:
	    {
	       EBDetId detId(id);
	       oss << "EcalBarrel (ieta, iphi, tower_ieta, tower_iphi): "
		   << detId.ieta() << ", "
		   << detId.iphi() << ", "
		   << detId.tower_ieta() << ", "
		   << detId.tower_iphi();
	    }
	    break;
            case EcalEndcap:
	    {
	       EEDetId detId(id);
	       oss << "EcalEndcap (ix, iy, SuperCrystal, crystal, quadrant): "
		   << detId.ix() << ", "
		   << detId.iy() << ", "
		   << detId.isc() << ", "
		   << detId.ic() << ", "
		   << detId.iquadrant();
	    }
	    break;
            case EcalPreshower:
               oss << "EcalPreshower";
               break;
            case EcalTriggerTower:
               oss << "EcalTriggerTower";
               break;
            case EcalLaserPnDiode:
               oss << "EcalLaserPnDiode";
               break;
	 }
	 break;
      
      case DetId::Hcal:
      {
         HcalDetId detId(id);
         switch ( detId.subdet() ) {
	    case HcalEmpty:
	       oss << "HcalEmpty ";
	       break;
	    case HcalBarrel:
	       oss << "HcalBarrel ";
	       break;
	    case HcalEndcap:
	       oss << "HcalEndcap ";
	       break;
	    case HcalOuter:
	       oss << "HcalOuter ";
	       break;
	    case HcalForward:
	       oss << "HcalForward ";
	       break;
	    case HcalTriggerTower:
	       oss << "HcalTriggerTower ";
	       break;
	    case HcalOther:
	       oss << "HcalOther ";
	       break;
         }
         oss << "(ieta, iphi, depth):"
             << detId.ieta() << ", "
             << detId.iphi() << ", "
             << detId.depth();
      }
      break;
      default :;
   }
   return oss.str();
}
 
std::string
info(const std::set<DetId>& idSet) {
   std::string text;
   for(std::set<DetId>::const_iterator id = idSet.begin(), idEnd = idSet.end(); id != idEnd; ++id)
   {
      text += info(*id);
      text += "\n";
   }
   return text;
}

std::string
info(const std::vector<DetId>& idSet) {
   std::string text;
   for(std::vector<DetId>::const_iterator id = idSet.begin(), idEnd = idSet.end(); id != idEnd; ++id)
   {
      text += info(*id);
      text += "\n";
   }
   return text;
}   
}
