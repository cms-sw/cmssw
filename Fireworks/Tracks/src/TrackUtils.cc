// -*- C++ -*-
//
// Package:     Core
// Class  :     TrackUtils
// $Id: TrackUtils.cc,v 1.15 2010/01/28 19:02:33 amraktad Exp $
//

// system include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveStraightLineSet.h"
#include "TEveVSDStructs.h"

// user include files

#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/TEveElementIter.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"


#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
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


namespace fireworks {

TEveTrack*
prepareTrack(const reco::Track& track,
             TEveTrackPropagator* propagator,
             Color_t color,
             const std::vector<TEveVector>& extraRefPoints)
{
   // To make use of all available information, we have to order states
   // properly first. Propagator should take care of y=0 transition.

   std::vector<State> refStates;
   TEveVector trackMomentum( track.px(), track.py(), track.pz() );
   refStates.push_back(State(TEveVector(track.vertex().x(),
                                        track.vertex().y(),
                                        track.vertex().z()),
                             trackMomentum));
   if ( track.extra().isAvailable() ) {
      refStates.push_back(State(TEveVector( track.innerPosition().x(),
                                            track.innerPosition().y(),
                                            track.innerPosition().z() ),
                                TEveVector( track.innerMomentum().x(),
                                            track.innerMomentum().y(),
                                            track.innerMomentum().z() )));
      refStates.push_back(State(TEveVector( track.outerPosition().x(),
                                            track.outerPosition().y(),
                                            track.outerPosition().z() ),
                                TEveVector( track.outerMomentum().x(),
                                            track.outerMomentum().y(),
                                            track.outerMomentum().z() )));
   }
   for ( std::vector<TEveVector>::const_iterator point = extraRefPoints.begin();
         point != extraRefPoints.end(); ++point )
      refStates.push_back(State(*point));
   if (track.pt()>1) std::sort( refStates.begin(), refStates.end(), StateOrdering(trackMomentum) );

   // * if the first state has non-zero momentum use it as a starting point
   //   and all other points as PathMarks to follow
   // * if the first state has only position, try the last state. If it has
   //   momentum we propagate backword, if not, we look for the first one
   //   on left that has momentum and ignore all earlier.
   //

   TEveRecTrack t;
   t.fBeta = 1.;
   t.fSign = track.charge();

   if ( refStates.front().valid ) {
      t.fV = refStates.front().position;
      t.fP = refStates.front().momentum;
      TEveTrack* trk = new TEveTrack(&t,propagator);
      trk->SetBreakProjectedTracks(TEveTrack::kBPTAlways);
      trk->SetMainColor(color);
      for( unsigned int i(1); i<refStates.size()-1; ++i) {
         if ( refStates[i].valid )
            trk->AddPathMark( TEvePathMark( TEvePathMark::kReference, refStates[i].position, refStates[i].momentum ) );
         else
            trk->AddPathMark( TEvePathMark( TEvePathMark::kDaughter, refStates[i].position ) );
      }
      if ( refStates.size()>1 ) {
         trk->AddPathMark( TEvePathMark( TEvePathMark::kDecay, refStates.back().position ) );
      }
      return trk;
   }

   if ( refStates.back().valid ) {
      t.fSign = (-1)*track.charge();
      t.fV = refStates.back().position;
      t.fP = refStates.back().momentum * (-1.f);
      TEveTrack* trk = new TEveTrack(&t,propagator);
      trk->SetBreakProjectedTracks(TEveTrack::kBPTAlways);
      trk->SetMainColor(color);
      unsigned int i(refStates.size()-1);
      for(; i>0; --i) {
         if ( refStates[i].valid )
            trk->AddPathMark( TEvePathMark( TEvePathMark::kReference, refStates[i].position, refStates[i].momentum*(-1.f) ) );
         else
            trk->AddPathMark( TEvePathMark( TEvePathMark::kDaughter, refStates[i].position ) );
      }
      if ( refStates.size()>1 ) {
         trk->AddPathMark( TEvePathMark( TEvePathMark::kDecay, refStates.front().position ) );
      }
      return trk;
   }

   unsigned int i(0);
   while ( i<refStates.size() && !refStates[i].valid ) ++i;
   assert ( i < refStates.size() );

   t.fV = refStates[i].position;
   t.fP = refStates[i].momentum;
   TEveTrack* trk = new TEveTrack(&t,propagator);
   trk->SetBreakProjectedTracks(TEveTrack::kBPTAlways);
   trk->SetMainColor(color);

   for( unsigned int j(i+1); j<refStates.size()-1; ++j) {
      if ( refStates[i].valid )
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

TEveTrack*
prepareTrack(const reco::Candidate& track,
             TEveTrackPropagator* propagator,
             Color_t color)
{
   TEveRecTrack t;
   t.fBeta = 1.;
   t.fP = TEveVector( track.px(), track.py(), track.pz() );
   t.fV = TEveVector( track.vertex().x(), track.vertex().y(), track.vertex().z() );
   t.fSign = track.charge();
   TEveTrack* trk = new TEveTrack(&t,propagator);
   trk->SetMainColor(color);
   return trk;
}


//______________________________________________________________________________


void
pixelLocalXY(const double mpx, const double mpy, const DetId& id, double& lpx, double& lpy) {
   int nrows = 0;
   int ncols = 0;
   unsigned int subdet = (unsigned int)id.subdetId();
   if (subdet == PixelSubdetector::PixelBarrel) {
      PXBDetId pxbDet = id;
      int A = pxbDet.layer();
      // int B = pxbDet.module();
      int C = pxbDet.ladder();
      // std::cout << "-" << A << "-" << B << "-" << C;
      nrows = 160;
      ncols = 416;
      switch (A) {
         case 1:
            if (C==5 || C==6 || C==15 || C==16) nrows = 80;
            break;
         case 2:
            if (C==8 || C==9 || C==24 || C==25) nrows = 80;
            break;
         case 3:
            if (C==11 || C==12 || C==33 || C==34) nrows = 80;
            break;
         default:
            //std::cout << "pixelLocalXY wrong DetId" << std::endl;
            return;
      }
   } else if (subdet == PixelSubdetector::PixelEndcap) {
      PXFDetId pxfDet = id;
      // int A = pxfDet.disk();
      int B = pxfDet.module();
      int C = pxfDet.panel();
      // std::cout << "-" << A << "-" << B << "-" << C;
      if (B==1 && C==1) {
         nrows = 80;
         ncols = 104;
      } else if ((B==1 && C==2) || (B==2 && C==1)) {
         nrows = 160; ncols = 156;
      } else if ((B==2 && C==2) || (B==3 && C==1)) {
         nrows = 160; ncols = 208;
      } else if (B==3 && C==2) {
         nrows = 160; ncols = 260;
      } else if (B==4 && C==1) {
         nrows = 80; ncols = 260;
      } else {
         //std::cout << "pixelLocalXY wrong DetId" << std::endl;
         return;
      }

   } else {
      // std::cout << "pixelLocalXY wrong DetId" << std::endl;
      return;
   }
   lpx = pixelLocalX(mpx, nrows);
   lpy = pixelLocalY(mpy, ncols);
   return;
}

//______________________________________________________________________________

double
pixelLocalX(const double mpx, const int m_nrows) {
   const double MICRON = 1./1000./10.;
   const double m_pitchx = 100*MICRON;
   const int ROWS_PER_ROC = 80;       // Num of cols per ROC
   const int BIG_PIX_PER_ROC_X = 1;   // in x direction, rows

   const double m_xoffset = -(m_nrows + BIG_PIX_PER_ROC_X*m_nrows/ROWS_PER_ROC)/2. * m_pitchx;

   int binoffx=int(mpx);               // truncate to int
   double fractionX = mpx - binoffx;   // find the fraction
   double local_pitchx = m_pitchx;        // defaultpitch
   if (binoffx>80) {              // ROC 1 - handles x on edge cluster
      binoffx=binoffx+2;
   } else if (binoffx==80) {      // ROC 1
      binoffx=binoffx+1;
      local_pitchx = 2 * m_pitchx;
   } else if (binoffx==79) {        // ROC 0
      binoffx=binoffx+0;
      local_pitchx = 2 * m_pitchx;
   } else if (binoffx>=0) {         // ROC 0
      binoffx=binoffx+0;

   }
   //else {   // too small
   //std::cout<<" very bad, binx "<< binoffx << std::endl;
   //   std::cout<<mpx<<" "<<binoffx<<" " <<fractionX<<" "<<local_pitchx<<" "<<m_xoffset<<std::endl;
   //}

   // The final position in local coordinates
   double lpX = double(binoffx*m_pitchx) + fractionX*local_pitchx + m_xoffset;
   //if ( lpX<m_xoffset || lpX>(-m_xoffset) ) {
   //std::cout<<" bad lp x "<<lpX<<std::endl;
   //std::cout<<mpx<<" "<<binoffx<<" "<<fractionX<<" "<<local_pitchx<<" "<<m_xoffset<<std::endl;
   //}

   return lpX;
}

//______________________________________________________________________________

double
pixelLocalY(const double mpy, const int m_ncols) {
   const double MICRON = 1./1000./10.;
   const int BIG_PIX_PER_ROC_Y = 2; // in y direction, cols
   const int COLS_PER_ROC = 52;   // Num of Rows per ROC
   const double m_pitchy = 150*MICRON;

   double m_yoffset = -(m_ncols + BIG_PIX_PER_ROC_Y*m_ncols/COLS_PER_ROC)/2. * m_pitchy;

   int binoffy = int(mpy);               // truncate to int
   double fractionY = mpy - binoffy;   // find the fraction
   double local_pitchy = m_pitchy;        // defaultpitch

   if (binoffy>416) {              // ROC 8, not real ROC
      binoffy=binoffy+17;
   } else if (binoffy==416) {      // ROC 8
      binoffy=binoffy+16;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy==415) {      // ROC 7, last big pixel
      binoffy=binoffy+15;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy>364) {       // ROC 7
      binoffy=binoffy+15;
   } else if (binoffy==364) {      // ROC 7
      binoffy=binoffy+14;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy==363) {        // ROC 6
      binoffy=binoffy+13;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy>312) {         // ROC 6
      binoffy=binoffy+13;
   } else if (binoffy==312) {        // ROC 6
      binoffy=binoffy+12;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy==311) {        // ROC 5
      binoffy=binoffy+11;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy>260) {         // ROC 5
      binoffy=binoffy+11;
   } else if (binoffy==260) {        // ROC 5
      binoffy=binoffy+10;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy==259) {        // ROC 4
      binoffy=binoffy+9;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy>208) {         // ROC 4
      binoffy=binoffy+9;
   } else if (binoffy==208) {        // ROC 4
      binoffy=binoffy+8;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy==207) {        // ROC 3
      binoffy=binoffy+7;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy>156) {         // ROC 3
      binoffy=binoffy+7;
   } else if (binoffy==156) {        // ROC 3
      binoffy=binoffy+6;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy==155) {        // ROC 2
      binoffy=binoffy+5;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy>104) {         // ROC 2
      binoffy=binoffy+5;
   } else if (binoffy==104) {        // ROC 2
      binoffy=binoffy+4;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy==103) {        // ROC 1
      binoffy=binoffy+3;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy>52) {         // ROC 1
      binoffy=binoffy+3;
   } else if (binoffy==52) {        // ROC 1
      binoffy=binoffy+2;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy==51) {        // ROC 0
      binoffy=binoffy+1;
      local_pitchy = 2 * m_pitchy;
   } else if (binoffy>0) {          // ROC 0
      binoffy=binoffy+1;
   } else if (binoffy==0) {         // ROC 0
      binoffy=binoffy+0;
      local_pitchy = 2 * m_pitchy;
   }
   //else {   // too small
   //   std::cout<<" very bad, biny "<<binoffy<<std::endl;
   //   std::cout<<mpy<<" "<<binoffy<<" "<<fractionY<<" "<<local_pitchy<<" "<<m_yoffset<<std::endl;
   //}

   // The final position in local coordinates
   double lpY = double(binoffy*m_pitchy) + fractionY*local_pitchy + m_yoffset;
   //if(lpY<m_yoffset || lpY>(-m_yoffset) ) {
   //   std::cout<<" bad lp y "<<lpY<<std::endl;
   //   std::cout<<mpy<<" "<<binoffy<<" "<<fractionY<<" "<<local_pitchy<<" "<<m_yoffset<<std::endl;
   //}
   return lpY;
}

// -- Si module names for printout
static const std::string subdets[7] = {"UNKNOWN", "PXB", "PXF", "TIB", "TID", "TOB", "TEC" };

// -- SiStrip module mini geometry:
// -- end cap nModules: 24, 24, 40, 56, 40, 56, 80
// -- end cap nStrips: 768, 768, 512, 512, 768, 512, 512
// -- barrel dStrip: 80, 80, 120, 120, 183, 183, 183, 183, 122, 122
	
// -- end cap SiStrip module geometry
static const double twopi = 6.28318531;
static const double dpEStrips[7] = { twopi/24/768, twopi/24/768, twopi/40/512, twopi/56/512, twopi/40/768, twopi/56/512, twopi/80/512 };
static const int nEStrips[7] = { 768, 768, 512, 512, 768, 512, 512 };
static const double hEStrips[7] = {8.52, /* 11.09,*/ 8.82, 11.07, 11.52, 8.12+6.32, 9.61+8.49, 10.69+9.08};
// -- barrel SiStrip module geometry
static const double MICRON = 1./1000./10.;
static const double dpBStrips[10] = { 80.*MICRON, 80.*MICRON, 120.*MICRON, 120.*MICRON, 183.*MICRON, 183.*MICRON, 183.*MICRON, 183.*MICRON, 122.*MICRON, 122.*MICRON };
static const int nBStrips[10] = { 768, 768, 512, 512, 768, 768, 512, 512, 512, 512 };
static const double hBStrips[10] = { 11.69, 11.69, 11.69, 11.69, 2*9.16, 2*9.16, 2*9.16, 2*9.16, 2*9.16, 2*9.16 };
static int PRINT=0;

void localSiPixel(TVector3& point, double row, double col, 
                  DetId id, const FWEventItem* iItem) {
		
   const DetIdToMatrix *detIdToGeo = iItem->getGeom();
   const TGeoHMatrix *m = detIdToGeo->getMatrix(id);
   double lx = 0.;
   double ly = 0.;
   pixelLocalXY(row, col, id, lx, ly);
   if (PRINT) std::cout<<"SiPixelCluster, row=" << row << ", col=" << col ;
   if (PRINT) std::cout << ", lx=" << lx << ", ly=" << ly ;
   if (PRINT) std::cout << std::endl;
   double local[3] = { lx,ly,0. };
   double global[3] = { 0.,0.,0. };
   m->LocalToMaster(local, global);
   point.SetXYZ(global[0], global[1], global[2]);
		
}
void localSiStrip(TVector3& point, TVector3& pointA, TVector3& pointB, 
                  double bc, DetId id, const FWEventItem* iItem) {

		
   const DetIdToMatrix *detIdToGeo = iItem->getGeom();
   const TGeoHMatrix *m = detIdToGeo->getMatrix(id);

   // -- calc phi, eta, rho of detector
		
   double local[3] = { 0.,0.,0. };
   double global[3];
   m->LocalToMaster(local, global);
   point.SetXYZ(global[0], global[1], global[2]);
		
   double rhoDet = point.Pt();
   double zDet = point.Z();
   double phiDet = point.Phi();
		
   unsigned int subdet = (unsigned int)id.subdetId();
		
   if (PRINT) std::cout << subdets[subdet];
		
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
		
   if (PRINT) std::cout << " rhoDet: " << rhoDet << " zDet: " << zDet << " phiDet: " << phiDet;

   // -- here we have rNumber, 
   // -- and use the mini geometry to calculate strip position as function of cluster barycenter bc

   if ((subdet == SiStripDetId::TID) || (subdet == SiStripDetId::TEC)) {
		
      // -- get orientation of detector
      local[0] = 1.;
      local[1] = 0.;
      local[2] = 0.;
      m->LocalToMaster(local, global);
      TVector3 pp(global[0], global[1], global[2]);
      double dPhiDet = (pp.Phi()-phiDet) > 0 ? 1. : -1.;
      //LATB this does not quite work for stereo layers	
      bc = bc - nEStrips[rNumber]/2.;
      double dPhi = bc*dpEStrips[rNumber] * dPhiDet;
      if (PRINT) std::cout << " bc: "<< bc << ", dPhi: " << dPhi;
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
	
void addSiStripClusters(const FWEventItem* iItem, const reco::Track &t, class TEveElementList *tList, Color_t color, bool addNearbyClusters) {
   const char* title = "TrackHits";
   const edmNew::DetSetVector<SiStripCluster> * allClusters = 0;
   if (addNearbyClusters) {
      for (trackingRecHit_iterator it = t.recHitsBegin(); it!=t.recHitsEnd(); it++) {
         if (typeid(**it) == typeid(SiStripRecHit2D)) {
            const SiStripRecHit2D &hit = static_cast<const SiStripRecHit2D &>(**it);
            if (hit.cluster().isNonnull() && hit.cluster().isAvailable()) { allClusters = hit.cluster().product(); break; }
         }
      }
   }

   for (trackingRecHit_iterator it = t.recHitsBegin(); it!=t.recHitsEnd(); it++) {

      // -- get ring number (position of module in rho)
      DetId id = (*it)->geographicalId();
      int rNumber = 0;
      unsigned int subdet = (unsigned int)id.subdetId();
      if (subdet == SiStripDetId::TID) {
         TIDDetId tidDet = id;
         rNumber = tidDet.ringNumber()-1;
         if (PRINT) std::cout << "-" << tidDet.isStereo() << "-" << tidDet.isRPhi() << "-" << tidDet.isBackRing() << "-" << rNumber << "-" << tidDet.moduleNumber() << "-" << tidDet.diskNumber();
      }
      else if (subdet == SiStripDetId::TEC) {
         TECDetId tecDet = id;
         rNumber = tecDet.ringNumber()-1;
         if (PRINT) std::cout << "-" << tecDet.isStereo() << "-" << tecDet.isRPhi() << "-" << tecDet.isBackPetal() << "-" << rNumber << "-" << tecDet.moduleNumber() << "-" << tecDet.wheelNumber();
      }
      else if (subdet == SiStripDetId::TIB) {
         TIBDetId tibDet = id;
         rNumber = tibDet.layerNumber()-1;
         if (PRINT) std::cout << "-" << tibDet.isStereo() << "-" << tibDet.isRPhi() << "-" << tibDet.isDoubleSide() << "-" << rNumber << "-" << tibDet.moduleNumber() << "-" << tibDet.stringNumber();
      }
      else if (subdet == SiStripDetId::TOB) {
         TOBDetId tobDet = id;
         rNumber = tobDet.layerNumber()+3;
         if (PRINT) std::cout << "-" << tobDet.isStereo() << "-" << tobDet.isRPhi() << "-" << tobDet.isDoubleSide() << "-" << rNumber << "-" << tobDet.moduleNumber() << "-" << tobDet.rodNumber();
      }

      // -- get phi from SiStripHit
			
      TrackingRecHitRef rechitref = *it;
      const TrackingRecHit* rh = &(*rechitref);
      const SiStripRecHit2D* single = dynamic_cast<const SiStripRecHit2D*>(rh);
      if (single)     {
         if (PRINT) std::cout << " single hit ";
				
         const SiStripCluster* Cluster = 0;
         if (single->cluster().isNonnull())
            Cluster = single->cluster().get();
         else if (single->cluster_regional().isNonnull())
            Cluster = single->cluster_regional().get();
         else 
            if (PRINT) std::cout << " no cluster found!";
            
         if (Cluster) {
            if (allClusters != 0) {
               const edmNew::DetSet<SiStripCluster> & clustersOnThisDet = (*allClusters)[rh->geographicalId().rawId()];
               //if (clustersOnThisDet.size() > 1) std::cout << "DRAWING EXTRA CLUSTERS: N = " << clustersOnThisDet.size() << std::endl;
               for (edmNew::DetSet<SiStripCluster>::const_iterator itc = clustersOnThisDet.begin(), edc = clustersOnThisDet.end(); itc != edc; ++itc) {
                  double bc = itc->barycenter();
                  TVector3 point, pointA, pointB;
                  localSiStrip(point, pointA, pointB, bc, id, iItem);
                  if (PRINT) std::cout<<"SiStripCluster, bary center "<<bc<<", phi "<<point.Phi()<<std::endl;
                  TEveStraightLineSet *scposition = new TEveStraightLineSet(title);
                  scposition->SetDepthTest(false);
                  scposition->AddLine(pointA.X(), pointA.Y(), pointA.Z(), pointB.X(), pointB.Y(), pointB.Z());
                  scposition->SetLineColor(&*itc == Cluster ? kGreen : kRed);
                  tList->AddElement(scposition);
               }
            } else {
               double bc = Cluster->barycenter();
               TVector3 point, pointA, pointB; 
               localSiStrip(point, pointA, pointB, bc, id, iItem);
               if (PRINT) std::cout<<"SiStripCluster, bary center "<<bc<<", phi "<<point.Phi()<<std::endl;
               TEveStraightLineSet *scposition = new TEveStraightLineSet(title);
               scposition->SetDepthTest(false);
               scposition->AddLine(pointA.X(), pointA.Y(), pointA.Z(), pointB.X(), pointB.Y(), pointB.Z());
               scposition->SetLineColor(color);
               tList->AddElement(scposition);
            }

				
         }					
      } else if (!rh->isValid() && (id.rawId() != 0)) {    // lost hit
         if (allClusters != 0) {
            edmNew::DetSetVector<SiStripCluster>::const_iterator itds = allClusters->find(id.rawId());
            if (itds != allClusters->end()) {
               const edmNew::DetSet<SiStripCluster> & clustersOnThisDet = *itds;
               //if (clustersOnThisDet.size() > 0) std::cout << "DRAWING LOST HITS CLUSTERS: N = " << clustersOnThisDet.size() << std::endl;
               for (edmNew::DetSet<SiStripCluster>::const_iterator itc = clustersOnThisDet.begin(), edc = clustersOnThisDet.end(); itc != edc; ++itc) {
                  double bc = itc->barycenter();
                  TVector3 point, pointA, pointB;
                  localSiStrip(point, pointA, pointB, bc, id, iItem);
                  if (PRINT) std::cout<<"SiStripCluster, bary center "<<bc<<", phi "<<point.Phi()<<std::endl;
                  TEveStraightLineSet *scposition = new TEveStraightLineSet(title);
                  scposition->SetDepthTest(false);
                  scposition->AddLine(pointA.X(), pointA.Y(), pointA.Z(), pointB.X(), pointB.Y(), pointB.Z());
                  scposition->SetLineColor(kRed);
                  tList->AddElement(scposition);
               }
            }
         }
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
   for (trackingRecHit_iterator it = t.recHitsBegin(); it!=t.recHitsEnd(); it++) {

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

      const SiStripRecHit2D* single = dynamic_cast<const SiStripRecHit2D*>(rh);
      if (single)     {
         if (PRINT) std::cout << " single hit ";

         const SiStripCluster* Cluster = 0;
         if (single->cluster().isNonnull())
            Cluster = single->cluster().get();
         else if (single->cluster_regional().isNonnull())
            Cluster = single->cluster_regional().get();
         else 
            if (PRINT) std::cout << " no cluster found!";

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
   for (trackingRecHit_iterator it = t.recHitsBegin(); it!=t.recHitsEnd(); it++) {
      if (typeid(**it) == typeid(SiPixelRecHit)) {
         const SiPixelRecHit &hit = static_cast<const SiPixelRecHit &>(**it);
         if (hit.cluster().isNonnull() && hit.cluster().isAvailable()) { allClusters = hit.cluster().product(); break; }
      }
   }
   if (allClusters == 0) return;

   const DetIdToMatrix *detIdToGeo = iItem.getGeom();

   for (trackingRecHit_iterator it = t.recHitsBegin(); it!=t.recHitsEnd(); it++) {
      const TrackingRecHit* rh = &**it;

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
   for (trackingRecHit_iterator it = t.recHitsBegin(); it!=t.recHitsEnd(); it++) {
			
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
      if (m == 0) {
         if (PRINT) std::cout << "can't find Matrix" << std::endl;
         continue;
      }
			
      // -- in which detector are we?
			
      unsigned int subdet = (unsigned int)id.subdetId();
			
      if ((subdet == PixelSubdetector::PixelBarrel) || (subdet == PixelSubdetector::PixelEndcap)) {

         static const std::string subdets[7] = {"UNKNOWN", "PXB", "PXF", "TIB", "TID", "TOB", "TEC" };
         if (PRINT) std::cout << cnt++ << " -- ";
         if (PRINT) std::cout << subdets[subdet];
								
         const SiPixelRecHit* pixel = dynamic_cast<const SiPixelRecHit*>(rh);
         if (!pixel) {
            if (PRINT) std::cout << "can't find SiPixelRecHit" << std::endl;
            continue;
         }
         const SiPixelCluster& c = *(pixel->cluster());
         pushPixelCluster(pixelPoints, m, id, c);
      } else
         return;         // return if any non-Pixel DetID shows up
   }
}
void
pushPixelCluster(std::vector<TVector3> &pixelPoints, const TGeoHMatrix *m, DetId id, const SiPixelCluster &c) {
   double row = c.minPixelRow();
   double col = c.minPixelCol();
   double lx = 0.;
   double ly = 0.;
   pixelLocalXY(row, col, id, lx, ly);
   if (PRINT) std::cout << ", row: " << row << ", col: " << col ;
   if (PRINT) std::cout << ", lx: " << lx << ", ly: " << ly ;
				
   double local[3] = { lx,ly,0. };
   double global[3];
   m->LocalToMaster(local, global);
   TVector3 pb(global[0], global[1], global[2]);
   pixelPoints.push_back(pb);
				
				
   if (PRINT) std::cout << " x: " << pb.X() << ", y: " << pb.Y() << " z: " << pb.Z();
   if (PRINT) std::cout << " eta: " << pb.Eta() << ", phi: " << pb.Phi() << " rho: " << pb.Pt();
				
   if (PRINT) std::cout << std::endl;
				
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
   if (PRINT) std::cout << "Track eta: " << etaT << ", vz: " << vz << ", dz: " << dz;
   if (PRINT) std::cout << std::endl;
		
   int cnt=0;
   for (trackingRecHit_iterator it = t.recHitsBegin(); it!=t.recHitsEnd(); it++) {
			
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

      if ((subdet == PixelSubdetector::PixelBarrel) || (subdet == PixelSubdetector::PixelEndcap)) 
         continue;
			
			
			
      static const std::string subdets[7] = {"UNKNOWN", "PXB", "PXF", "TIB", "TID", "TOB", "TEC" };
			
      if (PRINT) std::cout << cnt++ << " -- ";
      if (PRINT) std::cout << subdets[subdet];
						
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
			
      const SiStripRecHit2D* single = dynamic_cast<const SiStripRecHit2D*>(rh);
      if (single)     {
         if (PRINT) std::cout << " single hit ";
				
         const SiStripCluster* Cluster = 0;
         if (single->cluster().isNonnull())
            Cluster = single->cluster().get();
         else if (single->cluster_regional().isNonnull())
            Cluster = single->cluster_regional().get();
         else 
            if (PRINT) std::cout << " no cluster found!";
				
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
					
            // E nModules: 24, 24, 40, 56, 40, 56, 80
            // E nStrips: 768, 768, 512, 512, 768, 512, 512
            // B dStrip: 80, 80, 120, 120, 183, 183, 183, 183, 122, 122
					
					
					
            double bc = Cluster->barycenter();
					
            if ((subdet == SiStripDetId::TID) || (subdet == SiStripDetId::TEC)) {
               // -- end cap
               const double twopi = 6.28318531;
               const double dpEStrips[7] = { twopi/24/768, twopi/24/768, twopi/40/512, twopi/56/512, twopi/40/768, twopi/56/512, twopi/80/512 };
               const int nEStrips[7] = { 768, 768, 512, 512, 768, 512, 512 };
               bc = bc - nEStrips[rNumber]/2.;
               double dPhi = bc*dpEStrips[rNumber] * dPhiDet;
               phi = phiDet + dPhi;
						
               if (PRINT) std::cout << " bc: "<< bc << ", dPhi: " << dPhi;
						
            } else {
               // -- barrel
               const double MICRON = 1./1000./10.;
               const double dpBStrips[10] = { 80.*MICRON, 80.*MICRON, 120.*MICRON, 120.*MICRON, 183.*MICRON, 183.*MICRON, 183.*MICRON, 183.*MICRON, 122.*MICRON, 122.*MICRON };
               const int nBStrips[10] = { 768, 768, 512, 512, 768, 768, 512, 512, 512, 512 };
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
addTrackerHitsEtaPhi(std::vector<TVector3> &points, class TEveElementList *tList, Color_t color, int size) {

   // -- draw points for TrackHits
   int cnt = 0;
   double zv = 0.;

   for(std::vector<TVector3>::const_iterator it = points.begin(); it != points.end(); ++it) {
      std::cout << cnt << " -- ";

      const double RHOB = 115.;
      const double ETAB = 1.572;
      const double ZPLUS = 265.;
      const double ZMINUS = -265.;
      const int nfac = 1;        // this should be under user control, going from 1...10 or so
      const double kfac = nfac*0.00036;

      double eta = (*it).Eta();
      double phi = (*it).Phi();
      double rho = (*it).Pt();
      double z = (*it).Z()-zv;

      double rhoMax = 0.;
      if (eta > ETAB)
         rhoMax = (ZPLUS-zv)/z*rho;
      else if (eta > -ETAB)
         rhoMax = RHOB;
      else    {
         rhoMax = (ZMINUS-zv)/z*rho;
      }

      double del = kfac*(rhoMax-rho);

      double xp1 = eta-del;
      double xp2 = eta+del;
      double yp = phi;
      double zp = -del*10.;

      TEveStraightLineSet *thposition = new TEveStraightLineSet("th position");
      thposition->SetDepthTest(kFALSE);
      thposition->AddLine(xp1, yp, zp, xp2, yp, zp);
      thposition->AddMarker(0, 0.);
      thposition->AddMarker(0, 1.);
      thposition->SetLineColor(kGray);


#ifdef LATBdebug
      if ( (cnt%2) == 0)
         thposition->SetMarkerSize(1);
      else
         thposition->SetMarkerSize(2);
      int icol = int(cnt/2)%10;
      thposition->SetMarkerColor(icol);
      tList->AddElement(thposition);
#else
      thposition->SetMarkerSize(size);
      thposition->SetMarkerColor(color);
      tList->AddElement(thposition);
#endif


      /*
        std::stringstream out;
        out << i;
        const char* sCnt = out.str().c_str();
        TEveText* t1 = new TEveText(sCnt);
        t1->SetFontSize(20);
        thposition->AddElement(t1);
      */
      std::cout << "eta: " << eta << " phi: " << phi << " z: " << z << " rho: " << rho;
      std::cout << std::endl;

      cnt++;
   }
}

//______________________________________________________________________________
	
void
addTrackerHits3D(std::vector<TVector3> &points, class TEveElementList *tList, Color_t color, int size) {
   TEvePointSet* pointSet = new TEvePointSet();
   pointSet->SetMarkerSize(size);
   pointSet->SetMarkerStyle(4);
   pointSet->SetMarkerColor(color);
		
   for(std::vector<TVector3>::const_iterator it = points.begin(); it != points.end(); ++it) {
      pointSet->SetNextPoint(it->x(),it->y(),it->z());
   }
   tList->AddElement(pointSet);
}
	
void addTrackerHits2Dbarrel(std::vector<TVector3> &points, class TEveElementList *tList, Color_t color, int size) {
		
		
}
	
}
