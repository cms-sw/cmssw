// -*- C++ -*-
//
// Package:     Tracks
// Class  :     TrackUtils
// $Id: TrackUtils.cc,v 1.31 2010/06/21 18:45:19 matevz Exp $
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
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/fwLog.h"

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

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

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

// -- SiStrip module mini geometry:
// -- end cap nModules: 24, 24, 40, 56, 40, 56, 80
// -- end cap nStrips: 768, 768, 512, 512, 768, 512, 512
// -- barrel dStrip: 80, 80, 120, 120, 183, 183, 183, 183, 122, 122
	
// -- end cap SiStrip module geometry
static const double TWOPI = 6.28318531;
static const double dpEStrips[7] = { TWOPI/24/768, TWOPI/24/768, TWOPI/40/512, TWOPI/56/512, TWOPI/40/768, TWOPI/56/512, TWOPI/80/512 };
static const int    nEStrips[7]  = { 768, 768, 512, 512, 768, 512, 512 };
static const double hEStrips[7]  = { 8.52, /* 11.09,*/ 8.82, 11.07, 11.52, 8.12+6.32, 9.61+8.49, 10.69+9.08 };

// -- barrel SiStrip module geometry
static const double dpBStrips[10] = { 80.*MICRON, 80.*MICRON, 120.*MICRON, 120.*MICRON, 183.*MICRON, 183.*MICRON, 183.*MICRON, 183.*MICRON, 122.*MICRON, 122.*MICRON };
static const int    nBStrips[10]  = { 768, 768, 512, 512, 768, 768, 512, 512, 512, 512 };
static const double hBStrips[10]  = { 11.69, 11.69, 11.69, 11.69, 2*9.16, 2*9.16, 2*9.16, 2*9.16, 2*9.16, 2*9.16 };

static int PRINT = 0;

TEveTrack*
prepareTrack(const reco::Track& track,
             TEveTrackPropagator* propagator,
             const std::vector<TEveVector>& extraRefPoints)
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


//______________________________________________________________________________

void
pixelLocalXY( const double mpx, const double mpy, const DetId& id, double& lpx, double& lpy ) {
   int nrows = 0;
   int ncols = 0;
   unsigned int subdet = id.subdetId();
   if( subdet == PixelSubdetector::PixelBarrel ) {
      PXBDetId pxbDet = id;
      int layer = pxbDet.layer();
      int ladder = pxbDet.ladder();
      nrows = 160;
      ncols = 416;
      switch( layer ) {
         case 1:
            if (ladder==5 || ladder==6 || ladder==15 || ladder==16) nrows = 80;
            break;
         case 2:
            if (ladder==8 || ladder==9 || ladder==24 || ladder==25) nrows = 80;
            break;
         case 3:
            if (ladder==11 || ladder==12 || ladder==33 || ladder==34) nrows = 80;
            break;
         default:
            // wrong DetId
            return;
      }
   } else if( subdet == PixelSubdetector::PixelEndcap ) {
      PXFDetId pxfDet = id;
      int module = pxfDet.module();
      int panel = pxfDet.panel();
      if( module==1 && panel==1 ) {
         nrows = 80;
         ncols = 104;
      } else if ((module==1 && panel==2) || (module==2 && panel==1)) {
         nrows = 160;
         ncols = 156;
      } else if ((module==2 && panel==2) || (module==3 && panel==1)) {
         nrows = 160;
         ncols = 208;
      } else if (module==3 && panel==2) {
         nrows = 160;
         ncols = 260;
      } else if (module==4 && panel==1) {
         nrows = 80;
         ncols = 260;
      } else {
         // wrong DetId
         return;
      }
   } else {
      // wrong DetId
      return;
   }
   lpx = pixelLocalX( mpx, nrows );
   lpy = pixelLocalY( mpy, ncols );
   return;
}

//______________________________________________________________________________

double
pixelLocalX( const double mpx, const int nrows ) {
   const double xoffset = -(nrows + BIG_PIX_PER_ROC_X*nrows/ROWS_PER_ROC)/2. * PITCHX;

   int binoffx = int(mpx);             // truncate to int
   double fractionX = mpx - binoffx;   // find the fraction
   double local_PITCHX = PITCHX;       // defaultpitch
   if( binoffx>80 ) {              // ROC 1 - handles x on edge cluster
      binoffx = binoffx+2;
   } else if( binoffx == 80 ) {    // ROC 1
      binoffx = binoffx+1;
      local_PITCHX = 2 * PITCHX;
   } else if( binoffx == 79 ) {    // ROC 0
      binoffx = binoffx+0;
      local_PITCHX = 2 * PITCHX;
   } else if( binoffx >= 0 ) {     // ROC 0
      binoffx = binoffx+0;
   }

   // The final position in local coordinates
   double lpX = double( binoffx*PITCHX ) + fractionX*local_PITCHX + xoffset;

   return lpX;
}

//______________________________________________________________________________

double
pixelLocalY( const double mpy, const int ncols ) {
   double yoffset = -(ncols + BIG_PIX_PER_ROC_Y*ncols/COLS_PER_ROC)/2. * PITCHY;

   int binoffy = int(mpy);             // truncate to int
   double fractionY = mpy - binoffy;   // find the fraction
   double local_PITCHY = PITCHY;       // defaultpitch

   if( binoffy>416 ) {                 // ROC 8, not real ROC
      binoffy = binoffy+17;
   } else if( binoffy == 416 ) {       // ROC 8
      binoffy = binoffy+16;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy == 415 ) {       // ROC 7, last big pixel
      binoffy = binoffy+15;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy > 364 ) {        // ROC 7
      binoffy = binoffy+15;
   } else if( binoffy == 364 ) {       // ROC 7
      binoffy = binoffy+14;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy == 363 ) {       // ROC 6
      binoffy = binoffy+13;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy>312 ) {          // ROC 6
      binoffy = binoffy+13;
   } else if( binoffy == 312 ) {       // ROC 6
      binoffy = binoffy+12;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy == 311 ) {       // ROC 5
      binoffy = binoffy+11;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy > 260 ) {        // ROC 5
      binoffy = binoffy+11;
   } else if( binoffy == 260 ) {       // ROC 5
      binoffy = binoffy+10;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy == 259 ) {       // ROC 4
      binoffy = binoffy+9;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy > 208 ) {        // ROC 4
      binoffy = binoffy+9;
   } else if(binoffy == 208 ) {        // ROC 4
      binoffy = binoffy+8;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy == 207 ) {       // ROC 3
      binoffy = binoffy+7;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy > 156 ) {        // ROC 3
      binoffy = binoffy+7;
   } else if( binoffy == 156 ) {       // ROC 3
      binoffy = binoffy+6;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy == 155 ) {       // ROC 2
      binoffy = binoffy+5;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy > 104 ) {        // ROC 2
      binoffy = binoffy+5;
   } else if( binoffy == 104 ) {       // ROC 2
      binoffy = binoffy+4;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy == 103 ) {       // ROC 1
      binoffy = binoffy+3;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy > 52 ) {         // ROC 1
      binoffy = binoffy+3;
   } else if( binoffy == 52 ) {        // ROC 1
      binoffy = binoffy+2;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy == 51 ) {        // ROC 0
      binoffy = binoffy+1;
      local_PITCHY = 2 * PITCHY;
   } else if( binoffy > 0 ) {          // ROC 0
      binoffy=binoffy+1;
   } else if( binoffy == 0 ) {         // ROC 0
      binoffy = binoffy+0;
      local_PITCHY = 2 * PITCHY;
   }

   // The final position in local coordinates
   double lpY = double( binoffy*PITCHY ) + fractionY*local_PITCHY + yoffset;

   return lpY;
}

void localSiPixel( TVector3& point, double row, double col, 
                   DetId id, const FWEventItem* iItem )
{
   const DetIdToMatrix *detIdToGeo = iItem->getGeom();
   const TGeoHMatrix *m = detIdToGeo->getMatrix(id);
   double lx = 0.;
   double ly = 0.;
   pixelLocalXY( row, col, id, lx, ly );
   if( PRINT )
      std::cout<<"SiPixelCluster, row=" << row
               << ", col=" << col 
               << ", lx=" << lx << ", ly=" << ly 
               << std::endl;
   double local[3] = { lx, ly, 0. };
   double global[3] = { 0., 0., 0. };
   m->LocalToMaster( local, global );
   point.SetXYZ( global[0], global[1], global[2] );
}

void localSiStrip( TVector3& point, TVector3& pointA, TVector3& pointB, 
                   double bc, DetId id, const FWEventItem* iItem )
{
   const DetIdToMatrix *detIdToGeo = iItem->getGeom();
   const TGeoHMatrix *m = detIdToGeo->getMatrix(id);

   // -- calc phi, eta, rho of detector
   double local[3] = { 0., 0., 0. };
   double global[3];
   m->LocalToMaster( local, global );
   point.SetXYZ( global[0], global[1], global[2] );
		
   double rhoDet = point.Pt();
   double zDet = point.Z();
   double phiDet = point.Phi();
		
   unsigned int subdet = (unsigned int)id.subdetId();
		
   if( PRINT ) std::cout << subdets[subdet];
		
   double phi = 0.;
   int rNumber = 0;
   bool stereoDet = 0;
   if( subdet == SiStripDetId::TID ) {
      TIDDetId tidDet = id;
      rNumber = tidDet.ringNumber()-1;
      stereoDet = tidDet.isStereo();
      if( PRINT )
         std::cout << "-" << tidDet.isStereo()
                   << "-" << tidDet.isRPhi()
                   << "-" << tidDet.isBackRing()
                   << "-" << rNumber
                   << "-" << tidDet.moduleNumber()
                   << "-" << tidDet.diskNumber();
   } else if( subdet == SiStripDetId::TEC ) {
      TECDetId tecDet = id;
      rNumber = tecDet.ringNumber()-1;
      stereoDet = tecDet.isStereo();
      if( PRINT ) std::cout << "-" << tecDet.isStereo()
			    << "-" << tecDet.isRPhi()
			    << "-" << tecDet.isBackPetal()
			    << "-" << rNumber
			    << "-" << tecDet.moduleNumber()
			    << "-" << tecDet.wheelNumber();
   } else if( subdet == SiStripDetId::TIB ) {
      TIBDetId tibDet = id;
      rNumber = tibDet.layerNumber()-1;
      stereoDet = tibDet.isStereo();
      if( PRINT ) std::cout << "-" << tibDet.isStereo()
			    << "-" << tibDet.isRPhi()
			    << "-" << tibDet.isDoubleSide()
			    << "-" << rNumber
			    << "-" << tibDet.moduleNumber()
			    << "-" << tibDet.stringNumber();
   } else if( subdet == SiStripDetId::TOB ) {
      TOBDetId tobDet = id;
      rNumber = tobDet.layerNumber()+3;
      stereoDet = tobDet.isStereo();
      if( PRINT ) std::cout << "-" << tobDet.isStereo()
			    << "-" << tobDet.isRPhi()
			    << "-" << tobDet.isDoubleSide()
			    << "-" << rNumber
			    << "-" << tobDet.moduleNumber()
			    << "-" << tobDet.rodNumber();
   }
		
   if( PRINT ) std::cout << " rhoDet: " << rhoDet << " zDet: " << zDet << " phiDet: " << phiDet;

   // -- here we have rNumber, 
   // -- and use the mini geometry to calculate strip position as function of cluster barycenter bc

   if( (subdet == SiStripDetId::TID) || (subdet == SiStripDetId::TEC) ) {
      // -- get orientation of detector
      local[0] = 1.;
      local[1] = 0.;
      local[2] = 0.;
      m->LocalToMaster( local, global );
      TVector3 pp( global[0], global[1], global[2] );
      double dPhiDet = ( pp.Phi()-phiDet ) > 0 ? 1. : -1.;
      //LATB this does not quite work for stereo layers	
      bc = bc - nEStrips[rNumber]/2.;
      double dPhi = bc*dpEStrips[rNumber] * dPhiDet;
      if( PRINT ) std::cout << " bc: "<< bc << ", dPhi: " << dPhi;
      phi = phiDet + dPhi;
			
      double z = zDet;
      double rho = rhoDet; // +- stripLength/2
      double tanLambda = z/rho;
      double eta = log(tanLambda + sqrt(1+tanLambda*tanLambda));

      point.SetPtEtaPhi(rho, eta, phi);
      rho = rhoDet-hEStrips[rNumber]/2.;
      tanLambda = z/rho;
      eta = log(tanLambda + sqrt(1+tanLambda*tanLambda));
      pointA.SetPtEtaPhi(rho, eta, phi);
      rho = rhoDet+hEStrips[rNumber]/2.;
      tanLambda = z/rho;
      eta = log(tanLambda + sqrt(1+tanLambda*tanLambda));
      pointB.SetPtEtaPhi(rho, eta, phi);
   } else {

      // -- barrel
      bc = bc - nBStrips[rNumber]/2.;
      double dx = bc*dpBStrips[rNumber];
			
      // mysterious shifts for TOB
			
      if (rNumber == 4) dx = dx + 2.3444;
      if (rNumber == 5) dx = dx + 2.3444;
      if (rNumber == 8) dx = dx - 1.5595;
      if (rNumber == 9) dx = dx - 1.5595;
		
      local[0] = dx;
      local[1] = 0.;
      local[2] = 0.;
      m->LocalToMaster(local, global);
      point.SetXYZ(global[0], global[1], global[2]);  // z +- stripLength/2
      local[0] = dx;
      local[1] = hBStrips[rNumber]/2.;
      local[2] = 0.;
      m->LocalToMaster(local, global);
      pointA.SetXYZ(global[0], global[1], global[2]);  // z +- stripLength/2
      local[0] = dx;
      local[1] = -hBStrips[rNumber]/2.;
      local[2] = 0.;
      m->LocalToMaster(local, global);
      pointB.SetXYZ(global[0], global[1], global[2]);  // z +- stripLength/2
			
      if (PRINT) std::cout << " bc: "<< bc  << ", dx: " << dx;

      phi = point.Phi();
		
   }
   if (PRINT) std::cout << std::endl;
		
   return;
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

const SiStripCluster* extractClusterFromTrackingRecHit(const TrackingRecHit* rh)
{
   const SiStripCluster* clus = 0;

   {
      const SiStripRecHit2D* hit2d = dynamic_cast<const SiStripRecHit2D*>(rh);
      if (hit2d)
      {
         if (PRINT) std::cout << " hit 2D ";

         if (hit2d->cluster().isNonnull())
         {
            clus = hit2d->cluster().get();
         }
         else if (hit2d->cluster_regional().isNonnull())
         {
            clus = hit2d->cluster_regional().get();
         }
         else
         {
            if (PRINT) std::cout << " no cluster found!";
         }
      }
   }
   if (clus == 0)
   {
      const SiStripRecHit1D* hit1d = dynamic_cast<const SiStripRecHit1D*>(rh);
      if (hit1d)
      {
         if (PRINT) std::cout << " hit 1D ";

         if (hit1d->cluster().isNonnull())
         {
            clus = hit1d->cluster().get();
         }
         else if (hit1d->cluster_regional().isNonnull())
         {
            clus = hit1d->cluster_regional().get();
         }
         else
         {
            if (PRINT) std::cout << " no cluster found!";
         }
      }
   }
   return clus;
}

void
addSiStripClusters(const FWEventItem* iItem, const reco::Track &t, class TEveElement *tList, bool addNearbyClusters , bool master) 
{
   // master is true if the product is for proxy builder
   const char* title = "TrackHits";

   const edmNew::DetSetVector<SiStripCluster> * allClusters = 0;
   if (addNearbyClusters)
   {
      for (trackingRecHit_iterator it = t.recHitsBegin(), itEnd = t.recHitsEnd(); it != itEnd; ++it )
      {
         if (typeid(**it) == typeid(SiStripRecHit2D))
         {
            const SiStripRecHit2D &hit = static_cast<const SiStripRecHit2D &>(**it);
            if( hit.cluster().isNonnull() && hit.cluster().isAvailable() ) {
               allClusters = hit.cluster().product();
               break;
            }
         }
         else if (typeid(**it) == typeid(SiStripRecHit1D))
         {
            const SiStripRecHit1D &hit = static_cast<const SiStripRecHit1D &>(**it);
            if( hit.cluster().isNonnull() && hit.cluster().isAvailable() ) {
               allClusters = hit.cluster().product();
               break;
            }
         }
      }
   }

   for (trackingRecHit_iterator it = t.recHitsBegin(), itEnd = t.recHitsEnd(); it != itEnd; ++it)
   {
      // -- get ring number (position of module in rho)
      DetId id = (*it)->geographicalId();
      int rNumber = 0;
      unsigned int subdet = (unsigned int)id.subdetId();
      if( subdet == SiStripDetId::TID ) {
         TIDDetId tidDet = id;
         rNumber = tidDet.ringNumber()-1;
         if( PRINT )
            std::cout << "-" << tidDet.isStereo()
                      << "-" << tidDet.isRPhi()
                      << "-" << tidDet.isBackRing()
                      << "-" << rNumber
                      << "-" << tidDet.moduleNumber()
                      << "-" << tidDet.diskNumber();
      }
      else if( subdet == SiStripDetId::TEC ) {
         TECDetId tecDet = id;
         rNumber = tecDet.ringNumber()-1;
         if( PRINT )
            std::cout << "-" << tecDet.isStereo()
                      << "-" << tecDet.isRPhi()
                      << "-" << tecDet.isBackPetal()
                      << "-" << rNumber
                      << "-" << tecDet.moduleNumber()
                      << "-" << tecDet.wheelNumber();
      }
      else if( subdet == SiStripDetId::TIB ) {
         TIBDetId tibDet = id;
         rNumber = tibDet.layerNumber()-1;
         if( PRINT )
            std::cout << "-" << tibDet.isStereo()
                      << "-" << tibDet.isRPhi()
                      << "-" << tibDet.isDoubleSide()
                      << "-" << rNumber
                      << "-" << tibDet.moduleNumber()
                      << "-" << tibDet.stringNumber();
      }
      else if( subdet == SiStripDetId::TOB ) {
         TOBDetId tobDet = id;
         rNumber = tobDet.layerNumber()+3;
         if( PRINT )
            std::cout << "-" << tobDet.isStereo()
                      << "-" << tobDet.isRPhi()
                      << "-" << tobDet.isDoubleSide()
                      << "-" << rNumber
                      << "-" << tobDet.moduleNumber()
                      << "-" << tobDet.rodNumber();
      }

      // -- get phi from SiStripHit

      TrackingRecHitRef rechitref = *it;
      const TrackingRecHit *rh      = &(*rechitref);
      const SiStripCluster *Cluster = extractClusterFromTrackingRecHit(rh);

      if (Cluster)
      {
         if (allClusters != 0)
         {
            const edmNew::DetSet<SiStripCluster> & clustersOnThisDet = (*allClusters)[rh->geographicalId().rawId()];
            //if (clustersOnThisDet.size() > 1) std::cout << "DRAWING EXTRA CLUSTERS: N = " << clustersOnThisDet.size() << std::endl;
            for (edmNew::DetSet<SiStripCluster>::const_iterator itc = clustersOnThisDet.begin(), edc = clustersOnThisDet.end(); itc != edc; ++itc)
            {
               double bc = itc->barycenter();
               TVector3 point, pointA, pointB;
               localSiStrip(point, pointA, pointB, bc, id, iItem);
               if (PRINT) std::cout<<"SiStripCluster, bary center "<<bc<<", phi "<<point.Phi()<<std::endl;
               TEveStraightLineSet *scposition = new TEveStraightLineSet(title);
               scposition->SetDepthTest(false);
               scposition->AddLine(pointA.X(), pointA.Y(), pointA.Z(), pointB.X(), pointB.Y(), pointB.Z());
               scposition->SetLineColor(&*itc == Cluster ? kGreen : kRed); 
               setupAddElement(scposition, tList, iItem, master, false);
                
            }
         }
         else
         {
            double bc = Cluster->barycenter();
            TVector3 point, pointA, pointB; 
            localSiStrip(point, pointA, pointB, bc, id, iItem);
            if (PRINT) std::cout<<"SiStripCluster, bary center "<<bc<<", phi "<<point.Phi()<<std::endl;
            TEveStraightLineSet *scposition = new TEveStraightLineSet(title);
            scposition->SetDepthTest(false);
            scposition->AddLine(pointA.X(), pointA.Y(), pointA.Z(), pointB.X(), pointB.Y(), pointB.Z());
            setupAddElement(scposition, tList, iItem, master, true);
         }		
      }
      else if (!rh->isValid() && (id.rawId() != 0)) // lost hit
      {
         if (allClusters != 0) {
            edmNew::DetSetVector<SiStripCluster>::const_iterator itds = allClusters->find(id.rawId());
            if (itds != allClusters->end())
            {
               const edmNew::DetSet<SiStripCluster> & clustersOnThisDet = *itds;
               //if (clustersOnThisDet.size() > 0) std::cout << "DRAWING LOST HITS CLUSTERS: N = " << clustersOnThisDet.size() << std::endl;
               for (edmNew::DetSet<SiStripCluster>::const_iterator itc = clustersOnThisDet.begin(), edc = clustersOnThisDet.end(); itc != edc; ++itc)
               {
                  double bc = itc->barycenter();
                  TVector3 point, pointA, pointB;
                  localSiStrip(point, pointA, pointB, bc, id, iItem);
                  if (PRINT) std::cout<<"SiStripCluster, bary center "<<bc<<", phi "<<point.Phi()<<std::endl;
                  TEveStraightLineSet *scposition = new TEveStraightLineSet(title);
                  scposition->SetDepthTest(false);
                  scposition->AddLine(pointA.X(), pointA.Y(), pointA.Z(), pointB.X(), pointB.Y(), pointB.Z());
                  setupAddElement(scposition, tList, iItem, master, false);
                  scposition->SetLineColor(kRed);
               }
            }
         }
      }
      else
      {
         if (PRINT) std::cout << "*ANOTHER* option possible: valid=" << rh->isValid() << ", rawid=" << id.rawId() << std::endl;
      }
   }
}

//______________________________________________________________________________
	
void
pushTrackerHits(std::vector<TVector3> &monoPoints, std::vector<TVector3> &stereoPoints, 
                const FWEventItem &iItem, const reco::Track &t) {

   /*
    * -- to do:
    * --    better estimate of event vertex
    * --       or should we use track vertex, also: include vx, vy?
    * --    figure out matched hits -> Kevin
    * --    check pixel coords w/r to Kevin's program
    * --    use vz also for clusters
    * --    fix when phi goes from -pi to +pi, like event 32, 58
    * --    check where "funny offsets" come from
    * --    change markers so that overlays can actually be seen, like gsf hits vs ctf hits
    * --    change colors of impact points, etc
    * --    matched hits, like in event 22, show up at odd phis
    * --    check strange events:
    * --      Pixel hits, why do they turn around in phi? like in event 23
    * --      event 20 in e11.root: why are large-rho hits off?
    * --      event 21, why is one of the gsf track hits at phi=0?
    * --      event 25, del is negative?
    * --    check
    * --    add other ECAL hits, like Dave did
    */
   const DetIdToMatrix *detIdToGeo = iItem.getGeom();

   double tanTheta = tan(t.theta());
   double dz = t.dz();

   // -- vertex correction
   double vz = t.vz();
   double zv = dz; //LatB zv = 0.; 

   double etaT = t.eta();
   if (PRINT) std::cout << "Track eta: " << etaT << ", vz: " << vz << ", dz: " << dz;
   if (PRINT) std::cout << std::endl;

   int cnt=0;
   for( trackingRecHit_iterator it = t.recHitsBegin(), itEnd = t.recHitsEnd(); it != itEnd; ++it) {

      TrackingRecHitRef rechitref = *it;
      /*
        if ( !(rechitref.isValid() )) {
        if (PRINT) std::cout << "can't find RecHit" << std::endl;
        continue;
        }
      */
      const TrackingRecHit* rh = &(*rechitref);

      // -- get position of center of wafer, assuming (0,0,0) is the center
      DetId id = (*it)->geographicalId();
      const TGeoHMatrix *m = detIdToGeo->getMatrix(id);
      // -- assert(m != 0);
      if (m == 0) continue;

      // -- calc phi, eta, rho of detector

      double local[3] = { 0.,0.,0. };
      double global[3];
      m->LocalToMaster(local, global);
      TVector3 point(global[0], global[1], global[2]);

      double rhoDet = point.Pt();
      double zDet = point.Z();
      double phiDet = point.Phi();
      // -- get orientation of detector
      local[0] = 1.;
      local[1] = 0.;
      local[2] = 0.;
      m->LocalToMaster(local, global);
      TVector3 pp(global[0], global[1], global[2]);
      double dPhiDet = (pp.Phi()-phiDet) > 0 ? 1. : -1.;



      // -- in which detector are we?

      unsigned int subdet = (unsigned int)id.subdetId();

      if (PRINT) std::cout << cnt++ << " -- ";
      if (PRINT) std::cout << subdets[subdet];

      if ((subdet == PixelSubdetector::PixelBarrel) || (subdet == PixelSubdetector::PixelEndcap)) {
         const SiPixelRecHit* pixel = dynamic_cast<const SiPixelRecHit*>(rh);
         if (!pixel) {
            if (PRINT) std::cout << "can't find SiPixelRecHit" << std::endl;
            continue;
         }
         const SiPixelCluster& c = *(pixel->cluster());
         double row = c.minPixelRow();
         double col = c.minPixelCol();
         double lx = 0.;
         double ly = 0.;
         pixelLocalXY(row, col, id, lx, ly);
         if (PRINT) std::cout << ", row: " << row << ", col: " << col ;
         if (PRINT) std::cout << ", lx: " << lx << ", ly: " << ly ;

         local[0] = lx;
         local[1] = ly;
         local[2] = 0.;
         m->LocalToMaster(local, global);
         TVector3 pb(global[0], global[1], global[2]-zv);

         double rho = pb.Pt();
         double eta = pb.Eta();
         double phi = pb.Phi();
         point.SetPtEtaPhi(rho, eta, phi);
         point[2] += zv;

         monoPoints.push_back(point);

         if (PRINT) std::cout << " x: " << pb.X() << ", y: " << pb.Y() << " z: " << pb.Z()+zv;
         if (PRINT) std::cout << " eta: " << pb.Eta() << ", phi: " << pb.Phi() << " rho: " << pb.Pt();

         if (PRINT) std::cout << " rhoDet: " << rhoDet;
         if (PRINT) std::cout << std::endl;

         continue;
      }

      // -- SiStrips

      double phi = 0.;
      int rNumber = 0;
      bool stereoDet = 0;
      if (subdet == SiStripDetId::TID) {
         TIDDetId tidDet = id;
         rNumber = tidDet.ringNumber()-1;
         stereoDet = tidDet.isStereo();
         if (PRINT) std::cout << "-" << tidDet.isStereo() << "-" << tidDet.isRPhi() << "-" << tidDet.isBackRing() << "-" << rNumber << "-" << tidDet.moduleNumber() << "-" << tidDet.diskNumber();
      }
      else if (subdet == SiStripDetId::TEC) {
         TECDetId tecDet = id;
         rNumber = tecDet.ringNumber()-1;
         stereoDet = tecDet.isStereo();
         if (PRINT) std::cout << "-" << tecDet.isStereo() << "-" << tecDet.isRPhi() << "-" << tecDet.isBackPetal() << "-" << rNumber << "-" << tecDet.moduleNumber() << "-" << tecDet.wheelNumber();
      }
      else if (subdet == SiStripDetId::TIB) {
         TIBDetId tibDet = id;
         rNumber = tibDet.layerNumber()-1;
         stereoDet = tibDet.isStereo();
         if (PRINT) std::cout << "-" << tibDet.isStereo() << "-" << tibDet.isRPhi() << "-" << tibDet.isDoubleSide() << "-" << rNumber << "-" << tibDet.moduleNumber() << "-" << tibDet.stringNumber();
      }
      else if (subdet == SiStripDetId::TOB) {
         TOBDetId tobDet = id;
         rNumber = tobDet.layerNumber()+3;
         stereoDet = tobDet.isStereo();
         if (PRINT) std::cout << "-" << tobDet.isStereo() << "-" << tobDet.isRPhi() << "-" << tobDet.isDoubleSide() << "-" << rNumber << "-" << tobDet.moduleNumber() << "-" << tobDet.rodNumber();
      }

      if (PRINT) std::cout << " rhoDet: " << rhoDet << " zDet: " << zDet << " phiDet: " << phiDet << " dPhiDet: " << dPhiDet;

      // -- get phi from SiStripHit

      const SiStripCluster *Cluster = extractClusterFromTrackingRecHit(rh);
      if (Cluster) {
         /*
           const RecHit2DLocalPos* rechit2D = dynamic_cast<const RecHit2DLocalPos*>(rh);
           DetId detectorId = rechit2D->geographicalId();
           const StripTopology* topology = dynamic_cast<const StripTopology*>(&(geometry->idToDetUnit(detectorId)->topology()));
           ASSERT(topology);
           LocalPoint lp = topology->localPosition(Cluster->barycenter());
         */


         // -- here's my mini SiTracker topology function
         // -- in goes rhoDet, Cluster->barycenter(), subdet (to figure out E vs B)
         // -- out comes dPhi

         double bc = Cluster->barycenter();

         if ((subdet == SiStripDetId::TID) || (subdet == SiStripDetId::TEC)) {
            bc = bc - nEStrips[rNumber]/2.;
            double dPhi = bc*dpEStrips[rNumber] * dPhiDet;
            phi = phiDet + dPhi;

            if (PRINT) std::cout << " bc: "<< bc << ", dPhi: " << dPhi;

         } else {
            bc = bc - nBStrips[rNumber]/2.;
            double dx = bc*dpBStrips[rNumber];

            // mysterious shifts for TOB

            if (rNumber == 4) dx = dx + 2.3444;
            if (rNumber == 5) dx = dx + 2.3444;
            if (rNumber == 8) dx = dx - 1.5595;
            if (rNumber == 9) dx = dx - 1.5595;

            local[0] = dx;
            local[1] = 0.;
            local[2] = 0.;
            m->LocalToMaster(local, global);
            TVector3 pb(global[0], global[1], global[2]-zv);
            phi = pb.Phi();
            if (PRINT) std::cout << " bc: "<< bc  << ", dx: " << dx;

         }
      }
      else
      {
         if (PRINT) std::cout << " matched hit, can't draw" << std::endl;
         /*
           const SiStripMatchedRecHit2D* matched = dynamic_cast<const SiStripMatchedRecHit2D*>(rechit);
           if (matched) {
           localPositions(matched->monoHit(),geometry,points);
           localPositions(matched->stereoHit(),geometry,points);
           }
         */
         continue;

      }


      // -- get eta, rho from intersect of gsfTrack w/ wafer, only dPhi is well-measured

      double z = 0;
      double rho = 0;
      if ((subdet == SiStripDetId::TID) || (subdet == SiStripDetId::TEC)) {
         // -- end cap
         z = zDet;
         rho = (z-zv)*tanTheta;
      } else {
         // -- barrel
         rho = rhoDet;
         z = rho/tanTheta+zv;
      }
      double tanLambda = (z-zv)/rho;
      double eta = log(tanLambda + sqrt(1+tanLambda*tanLambda));

      point.SetPtEtaPhi(rho, eta, phi);
      point[2] += zv;

      // -- save point
      if (!stereoDet)
         monoPoints.push_back(point);
      else
         stereoPoints.push_back(point);

      if (PRINT) std::cout << std::endl;

   }


   return;
}

//______________________________________________________________________________

void
pushNearbyPixelHits(std::vector<TVector3> &pixelPoints, const FWEventItem &iItem, const reco::Track &t) {
   const edmNew::DetSetVector<SiPixelCluster> * allClusters = 0;
   for( trackingRecHit_iterator it = t.recHitsBegin(), itEnd = t.recHitsEnd(); it != itEnd; ++it) {
      if (typeid(**it) == typeid(SiPixelRecHit)) {
         const SiPixelRecHit &hit = static_cast<const SiPixelRecHit &>(**it);
         if (hit.cluster().isNonnull() && hit.cluster().isAvailable()) { allClusters = hit.cluster().product(); break; }
      }
   }
   if (allClusters == 0) return;

   const DetIdToMatrix *detIdToGeo = iItem.getGeom();

   for( trackingRecHit_iterator it = t.recHitsBegin(), itEnd = t.recHitsEnd(); it != itEnd; ++it) {
      const TrackingRecHit* rh = &(**it);

      DetId id = (*it)->geographicalId();
      const TGeoHMatrix *m = detIdToGeo->getMatrix(id);
      // -- assert(m != 0);
      if (m == 0) {
         if (PRINT) std::cout << "can't find Matrix" << std::endl;
         continue;
      }

      // -- in which detector are we?
      unsigned int subdet = (unsigned int)id.subdetId();
      if ((subdet != PixelSubdetector::PixelBarrel) && (subdet != PixelSubdetector::PixelEndcap)) continue;

      const SiPixelCluster *hitCluster = 0;
      const SiPixelRecHit* pixel = dynamic_cast<const SiPixelRecHit*>(rh);
      if (pixel != 0) hitCluster = pixel->cluster().get();
      edmNew::DetSetVector<SiPixelCluster>::const_iterator itds = allClusters->find(id.rawId());
      if (itds != allClusters->end()) {
         const edmNew::DetSet<SiPixelCluster> & clustersOnThisDet = *itds;
         //if (clustersOnThisDet.size() > (hitCluster != 0)) std::cout << "DRAWING EXTRA CLUSTERS: N = " << (clustersOnThisDet.size() - (hitCluster != 0))<< std::endl;
         for (edmNew::DetSet<SiPixelCluster>::const_iterator itc = clustersOnThisDet.begin(), edc = clustersOnThisDet.end(); itc != edc; ++itc) {
            if (&*itc != hitCluster) pushPixelCluster(pixelPoints, m, id, *itc);
         }
      }
   }
}

//______________________________________________________________________________

void
pushPixelHits(std::vector<TVector3> &pixelPoints, const FWEventItem &iItem, const reco::Track &t) {
		
   /*
    * -- return for each Pixel Hit a 3D point
    */
   const DetIdToMatrix *detIdToGeo = iItem.getGeom();
		
   double dz = t.dz();
   double vz = t.vz();
   double etaT = t.eta();
   if (PRINT) std::cout << "Track eta: " << etaT << ", vz: " << vz << ", dz: " << dz;
   if (PRINT) std::cout << std::endl;
		
   int cnt=0;
   for( trackingRecHit_iterator it = t.recHitsBegin(), itEnd = t.recHitsEnd(); it != itEnd; ++it) {
      const TrackingRecHit* rh = &(**it);			
      // -- get position of center of wafer, assuming (0,0,0) is the center
      DetId id = (*it)->geographicalId();
      const TGeoHMatrix *m = detIdToGeo->getMatrix( id );
      if( m == 0 ) {
         if (PRINT) std::cout << "can't find Matrix" << std::endl;
         continue;
      }
			
      // -- in which detector are we?			
      unsigned int subdet = (unsigned int)id.subdetId();
			
      if( (subdet == PixelSubdetector::PixelBarrel) || (subdet == PixelSubdetector::PixelEndcap) ) {
         if (PRINT) std::cout << cnt++ << " -- ";
         if (PRINT) std::cout << subdets[subdet];
								
         const SiPixelRecHit* pixel = dynamic_cast<const SiPixelRecHit*>(rh);
         if( !pixel ) {
            if( PRINT ) std::cout << "can't find SiPixelRecHit" << std::endl;
            continue;
         }
         const SiPixelCluster& c = *( pixel->cluster() );
         pushPixelCluster( pixelPoints, m, id, c );
      } else
         return;         // return if any non-Pixel DetID shows up
   }
}
  
void
pushPixelCluster( std::vector<TVector3> &pixelPoints, const TGeoHMatrix *m, DetId id, const SiPixelCluster &c ) {
   double row = c.minPixelRow();
   double col = c.minPixelCol();
   double lx = 0.;
   double ly = 0.;
   pixelLocalXY( row, col, id, lx, ly );
   if( PRINT )
      std::cout << ", row: " << row << ", col: " << col 
		<< ", lx: " << lx << ", ly: " << ly ;
				
   double local[3] = { lx,ly,0. };
   double global[3];
   m->LocalToMaster( local, global );
   TVector3 pb( global[0], global[1], global[2] );
   pixelPoints.push_back( pb );
				
   if( PRINT )
      std::cout << " x: " << pb.X()
		<< ", y: " << pb.Y()
		<< " z: " << pb.Z()
		<< " eta: " << pb.Eta()
		<< ", phi: " << pb.Phi()
		<< " rho: " << pb.Pt() << std::endl;
}

//______________________________________________________________________________
	
void
pushSiStripHits(std::vector<TVector3> &monoPoints, std::vector<TVector3> &stereoPoints, 
                const FWEventItem &iItem, const reco::Track &t) {
		
   /*
    * -- to do:
    * --    better estimate of event vertex
    * --       or should we use track vertex, also: include vx, vy?
    * --    figure out matched hits -> Kevin
    * --    check pixel coords w/r to Kevin's program
    * --    use vz also for clusters
    * --    fix when phi goes from -pi to +pi, like event 32, 58
    * --    check where "funny offsets" come from
    * --    change markers so that overlays can actually be seen, like gsf hits vs ctf hits
    * --    change colors of impact points, etc
    * --    matched hits, like in event 22, show up at odd phis
    * --    check strange events:
    * --      Pixel hits, why do they turn around in phi? like in event 23
    * --      event 20 in e11.root: why are large-rho hits off?
    * --      event 21, why is one of the gsf track hits at phi=0?
    * --      event 25, del is negative?
    * --    check
    * --    add other ECAL hits, like Dave did
    */

   const DetIdToMatrix *detIdToGeo = iItem.getGeom();
		
   double tanTheta = tan(t.theta());
   double dz = t.dz();
		
   // -- vertex correction
   double vz = t.vz();
   double zv = dz; //LatB zv = 0.; 
		
   double etaT = t.eta();
   if( PRINT ) std::cout << "Track eta: " << etaT << ", vz: " << vz << ", dz: " << dz << std::endl;
		
   int cnt = 0;
   for( trackingRecHit_iterator it = t.recHitsBegin(), itEnd = t.recHitsEnd(); it != itEnd; ++it) {
      const TrackingRecHit* rh = &(**it);
			
      // -- get position of center of wafer, assuming (0,0,0) is the center
      DetId id = (*it)->geographicalId();
      const TGeoHMatrix *m = detIdToGeo->getMatrix(id);
      // -- assert(m != 0);
      if (m == 0) continue;
			
      // -- calc phi, eta, rho of detector
			
      double local[3] = { 0., 0., 0. };
      double global[3];
      m->LocalToMaster( local, global );
      TVector3 point( global[0], global[1], global[2] );
			
      double rhoDet = point.Pt();
      double zDet = point.Z();
      double phiDet = point.Phi();
      // -- get orientation of detector
      local[0] = 1.;
      local[1] = 0.;
      local[2] = 0.;
      m->LocalToMaster( local, global );
      TVector3 pp( global[0], global[1], global[2] );
      double dPhiDet = ( pp.Phi()-phiDet ) > 0 ? 1. : -1.;
			
      // -- in which detector are we?
			
      unsigned int subdet = (unsigned int)id.subdetId();

      if( (subdet == PixelSubdetector::PixelBarrel) || (subdet == PixelSubdetector::PixelEndcap) ) 
         continue;
			
      if (PRINT) std::cout << cnt++ << " -- "
			   << subdets[subdet];
						
      double phi = 0.;
      int rNumber = 0;
      bool stereoDet = 0;
      if( subdet == SiStripDetId::TID ) {
         TIDDetId tidDet = id;
         rNumber = tidDet.ringNumber()-1;
         stereoDet = tidDet.isStereo();
         if( PRINT )
            std::cout << "-" << tidDet.isStereo()
                      << "-" << tidDet.isRPhi()
                      << "-" << tidDet.isBackRing()
                      << "-" << rNumber
                      << "-" << tidDet.moduleNumber()
                      << "-" << tidDet.diskNumber();
      }
      else if (subdet == SiStripDetId::TEC) {
         TECDetId tecDet = id;
         rNumber = tecDet.ringNumber()-1;
         stereoDet = tecDet.isStereo();
         if( PRINT )
            std::cout << "-" << tecDet.isStereo()
                      << "-" << tecDet.isRPhi()
                      << "-" << tecDet.isBackPetal()
                      << "-" << rNumber
                      << "-" << tecDet.moduleNumber()
                      << "-" << tecDet.wheelNumber();
      }
      else if( subdet == SiStripDetId::TIB ) {
         TIBDetId tibDet = id;
         rNumber = tibDet.layerNumber()-1;
         stereoDet = tibDet.isStereo();
         if( PRINT )
            std::cout << "-" << tibDet.isStereo()
                      << "-" << tibDet.isRPhi()
                      << "-" << tibDet.isDoubleSide()
                      << "-" << rNumber
                      << "-" << tibDet.moduleNumber()
                      << "-" << tibDet.stringNumber();
      }
      else if( subdet == SiStripDetId::TOB ) {
         TOBDetId tobDet = id;
         rNumber = tobDet.layerNumber()+3;
         stereoDet = tobDet.isStereo();
         if( PRINT )
            std::cout << "-" << tobDet.isStereo()
                      << "-" << tobDet.isRPhi()
                      << "-" << tobDet.isDoubleSide()
                      << "-" << rNumber
                      << "-" << tobDet.moduleNumber()
                      << "-" << tobDet.rodNumber();
      }
			
      if( PRINT )
         std::cout << " rhoDet: " << rhoDet << " zDet: " << zDet << " phiDet: " << phiDet << " dPhiDet: " << dPhiDet;
			
      // -- get phi from SiStripHit
			
      const SiStripCluster *Cluster = extractClusterFromTrackingRecHit(rh);
      if (Cluster)
      {
         /*
           const RecHit2DLocalPos* rechit2D = dynamic_cast<const RecHit2DLocalPos*>(rh);
           DetId detectorId = rechit2D->geographicalId();
           const StripTopology* topology = dynamic_cast<const StripTopology*>(&(geometry->idToDetUnit(detectorId)->topology()));
           ASSERT(topology);
           LocalPoint lp = topology->localPosition(Cluster->barycenter());
         */
					
         // -- here's my mini SiTracker topology function
         // -- in goes rhoDet, Cluster->barycenter(), subdet (to figure out E vs B)
         // -- out comes dPhi
					
         // E nModules: 24, 24, 40, 56, 40, 56, 80
         // E nStrips: 768, 768, 512, 512, 768, 512, 512
         // B dStrip: 80, 80, 120, 120, 183, 183, 183, 183, 122, 122
					
         double bc = Cluster->barycenter();
					
         if( (subdet == SiStripDetId::TID) || (subdet == SiStripDetId::TEC) ) {
            // -- end cap
            bc = bc - nEStrips[rNumber]/2.;
            double dPhi = bc*dpEStrips[rNumber] * dPhiDet;
            phi = phiDet + dPhi;
						
            if (PRINT) std::cout << " bc: "<< bc << ", dPhi: " << dPhi;
         } else {
            // -- barrel
            bc = bc - nBStrips[rNumber]/2.;
            double dx = bc*dpBStrips[rNumber];
						
            // mysterious shifts for TOB
						
            if (rNumber == 4) dx = dx + 2.3444;
            if (rNumber == 5) dx = dx + 2.3444;
            if (rNumber == 8) dx = dx - 1.5595;
            if (rNumber == 9) dx = dx - 1.5595;
						
            local[0] = dx;
            local[1] = 0.;
            local[2] = 0.;
            m->LocalToMaster(local, global);
            TVector3 pb(global[0], global[1], global[2]-zv);
            phi = pb.Phi();
            if (PRINT) std::cout << " bc: "<< bc  << ", dx: " << dx;
         }
      }
      else
      {
         if (PRINT) std::cout << " matched hit, can't draw" << std::endl;
         /*
           const SiStripMatchedRecHit2D* matched = dynamic_cast<const SiStripMatchedRecHit2D*>(rechit);
           if (matched) {
           localPositions(matched->monoHit(),geometry,points);
           localPositions(matched->stereoHit(),geometry,points);
           }
         */
         continue;
      }
			
      // -- get eta, rho from intersect of gsfTrack w/ wafer, only dPhi is well-measured
			
      double z = 0;
      double rho = 0;
      if( (subdet == SiStripDetId::TID) || (subdet == SiStripDetId::TEC) ) {
         // -- end cap
         z = zDet;
         rho = (z-zv)*tanTheta;
      } else {
         // -- barrel
         rho = rhoDet;
         z = rho/tanTheta+zv;
      }
      double tanLambda = (z-zv)/rho;
      double eta = log(tanLambda + sqrt(1+tanLambda*tanLambda));
			
      point.SetPtEtaPhi(rho, eta, phi);
      point[2] += zv;
			
      // -- save point
      if (!stereoDet)
         monoPoints.push_back(point);
      else
         stereoPoints.push_back(point);
			
      if( PRINT ) std::cout << std::endl;
			
   }
}

//______________________________________________________________________________
	
void
addTrackerHits3D( std::vector<TVector3> &points, class TEveElementList *tList, Color_t color, int size ) 
{
   // !AT this is  detail view specific, should move to track hits
   // detail view

   TEvePointSet* pointSet = new TEvePointSet();
   pointSet->SetMarkerSize(size);
   pointSet->SetMarkerStyle(4);
   pointSet->SetMarkerColor(color);
		
   for( std::vector<TVector3>::const_iterator it = points.begin(), itEnd = points.end(); it != itEnd; ++it) {
      pointSet->SetNextPoint(it->x(), it->y(), it->z());
   }
   tList->AddElement(pointSet);
}

void
addHits(const reco::Track& track,
        const FWEventItem* iItem,
        TEveElement* trkList,
        bool addNearbyHits)
{
   // !AT this is  detail view specific, should move to track hits
   // detail view

   std::vector<TVector3> pixelPoints;
   fireworks::pushPixelHits(pixelPoints, *iItem, track);
   TEveElementList* pixels = new TEveElementList("Pixels");
   trkList->AddElement(pixels);
   if( addNearbyHits ) {
      // get the extra hits
      std::vector<TVector3> pixelExtraPoints;
      fireworks::pushNearbyPixelHits(pixelExtraPoints, *iItem, track);
      // draw first the others
      fireworks::addTrackerHits3D(pixelExtraPoints, pixels, kRed, 1);
      // then the good ones, so they're on top
      fireworks::addTrackerHits3D(pixelPoints, pixels, kGreen, 1);
   } else {
      // just add those points with the default color
      fireworks::addTrackerHits3D(pixelPoints, pixels, iItem->defaultDisplayProperties().color(), 1);
   }

   // strips
   TEveElementList* strips = new TEveElementList("Strips");
   trkList->AddElement(strips);
   fireworks::addSiStripClusters(iItem, track, strips, addNearbyHits, false);
}

//______________________________________________________________________________

void
addModules( const reco::Track& track,
            const FWEventItem* iItem,
            TEveElement* trkList,
            bool addLostHits)
{
   // !AT this is  detail view specific, should move to track hits
   // detail view

   try {
      std::set<unsigned int> ids;
      for( trackingRecHit_iterator recIt = track.recHitsBegin(), recItEnd = track.recHitsEnd();
           recIt != recItEnd; ++recIt ) {
         DetId detid = (*recIt)->geographicalId();
         if (!addLostHits && !(*recIt)->isValid()) continue;
         if(detid.rawId() != 0) {
            TString name("");
            switch (detid.det())
            {
               case DetId::Tracker:
                  switch (detid.subdetId())
                  {
                     case SiStripDetId::TIB:
                        name = "TIB ";
                        break;
                     case SiStripDetId::TOB:
                        name = "TOB ";
                        break;
                     case SiStripDetId::TID:
                        name = "TID ";
                        break;
                     case SiStripDetId::TEC:
                        name = "TEC ";
                        break;
                     case PixelSubdetector::PixelBarrel:
                        name = "Pixel Barrel ";
                        break;
                     case PixelSubdetector::PixelEndcap:
                        name = "Pixel Endcap ";
                     default:
                        break;
                  }
                  break;

               case DetId::Muon:
                  switch (detid.subdetId())
                  {
                     case MuonSubdetId::DT:
                        name = "DT";
                        detid = DetId(DTChamberId(detid)); // get rid of layer bits
                        break;
                     case MuonSubdetId::CSC:
                        name = "CSC";
                        break;
                     case MuonSubdetId::RPC:
                        name = "RPC";
                        break;
                     default:
                        break;
                  }
                  break;
               default:
                  break;
            }
            if( ! ids.insert(detid.rawId()).second ) continue;
            if(iItem->getGeom()) {
               TEveGeoShape* shape = iItem->getGeom()->getShape( detid );
               if(0!=shape) {
                  shape->SetMainTransparency(65);
                  shape->SetPickable(kTRUE);
                  switch ((*recIt)->type()) {
                     case TrackingRecHit::valid:
                        shape->SetMainColor(iItem->defaultDisplayProperties().color());
                        break;
                     case TrackingRecHit::missing:
                        name += "LOST ";
                        shape->SetMainColor(kRed);
                        break;
                     case TrackingRecHit::inactive:
                        name += "INACTIVE ";
                        shape->SetMainColor(28);
                        break;
                     case TrackingRecHit::bad:
                        name += "BAD ";
                        shape->SetMainColor(218);
                        break;
                  }
                  shape->SetTitle(name + ULong_t(detid.rawId()));
                  trkList->AddElement(shape);
               } else {

                  fwLog(fwlog::kInfo) <<  "Failed to get shape extract for a tracking rec hit: "
                                      << "\n" << fireworks::info(detid) << std::endl;
               }
            }
         }
      }
   }
   catch (...) {
      fwLog(fwlog::kInfo) << "Sorry, don't have the recHits for this event." << std::endl;
   }
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
