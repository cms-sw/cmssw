/*
 * \file DTTrigGeomUtils.cc
 * 
 * $Date: 2009/08/03 16:08:38 $
 * $Revision: 1.2 $
 * \author C. Battilana - CIEMAT
 *
*/

#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"

// Trigger
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"

// Geometry & Segment
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include <iostream>

using namespace edm;
using namespace std;


DTTrigGeomUtils::DTTrigGeomUtils(ESHandle<DTGeometry> muonGeom, bool dirInDeg) : muonGeom_(muonGeom) {

  radToDeg_ = dirInDeg ? 180./Geom::pi() : 1;

  for (int ist=1; ist<=4; ++ist) {
    const DTChamberId chId(-2,ist,4);
    const DTChamber *chamb = muonGeom_->chamber(chId);
    const DTSuperLayer *sl1 = chamb->superLayer(DTSuperLayerId(chId,1));
    const DTSuperLayer *sl3 = chamb->superLayer(DTSuperLayerId(chId,3));
    zcn_[ist-1] = .5*(chamb->surface().toLocal(sl1->position()).z() + chamb->surface().toLocal(sl3->position()).z());
  }
  
  const DTChamber* chamb   = muonGeom_->chamber(DTChamberId(-2,4,13));
  const DTChamber* scchamb = muonGeom_->chamber(DTChamberId(-2,4,4));
  xCenter_[0] = scchamb->toLocal(chamb->position()).x()*.5;
  chamb   = muonGeom_->chamber(DTChamberId(-2,4,14));
  scchamb = muonGeom_->chamber(DTChamberId(-2,4,10));
  xCenter_[1] = scchamb->toLocal(chamb->position()).x()*.5;
      
  
}


DTTrigGeomUtils::~DTTrigGeomUtils() {

}


void DTTrigGeomUtils::computeSCCoordinates(const DTRecSegment4D* track, int& scsec, float& x, float& xdir, float& y, float& ydir){

  int sector = track->chamberId().sector();
  int station = track->chamberId().station();
  xdir = atan(track->localDirection().x()/ track->localDirection().z())*radToDeg_;
  ydir = atan(track->localDirection().y()/ track->localDirection().z())*radToDeg_;


  scsec = sector>12 ? sector==13 ? 4 : 10 : sector;
  float xcenter = (scsec==4||scsec==10) ? (sector-12.9)/abs(sector-12.9)*xCenter_[(sector==10||sector==14)] : 0.;
  x = track->localPosition().x()+xcenter*(station==4);
  y = track->localPosition().y();

}


void DTTrigGeomUtils::phiRange(const DTChamberId& id, float& min, float& max, int& nbins, float step){

  int station = id.station();
  int sector  = id.sector(); 
  
  const DTLayer  *layer = muonGeom_->layer(DTLayerId(id,1,1));
  DTTopology topo = layer->specificTopology();
  double range = topo.channels()*topo.cellWidth();
  min = -range*.5;
  max =  range*.5;

  if (station==4 && (sector==4 || sector == 10)){
    min = -range-10;
    max =  range+10;
  }
  nbins = static_cast<int>((max-min)/step);

  return;
 
}


void DTTrigGeomUtils::thetaRange(const DTChamberId& id, float& min, float& max, int& nbins, float step){

  const DTLayer  *layer = muonGeom_->layer(DTLayerId(id,2,1));
  DTTopology topo = layer->specificTopology();
  double range = topo.channels()*topo.cellWidth();
  min = -range*.5;
  max =  range*.5;

  nbins = static_cast<int>((max-min)/step);

  return;
 
}


float DTTrigGeomUtils::trigPos(const L1MuDTChambPhDigi* trig){

  
  int wh   = trig->whNum();
  int sec  = trig->scNum()+1;
  int st   = trig->stNum();
  int phi  = trig->phi();

  float phin = (sec-1)*Geom::pi()/6;
  float phicenter = 0;
  float r = 0;
  float xcenter = 0;

  if (sec==4 && st==4) {
    GlobalPoint gpos = phi>0 ? muonGeom_->chamber(DTChamberId(wh,st,13))->position() : muonGeom_->chamber(DTChamberId(wh,st,4))->position();
    xcenter = phi>0 ? xCenter_[0] : -xCenter_[0];
    phicenter =  gpos.phi();
    r = gpos.perp();
  } else if (sec==10 && st==4) {
    GlobalPoint gpos = phi>0 ? muonGeom_->chamber(DTChamberId(wh,st,14))->position() : muonGeom_->chamber(DTChamberId(wh,st,10))->position();
    xcenter = phi>0 ? xCenter_[1] : -xCenter_[1];  
    phicenter =  gpos.phi();
    r = gpos.perp();
  } else {
    GlobalPoint gpos = muonGeom_->chamber(DTChamberId(wh,st,sec))->position();
    phicenter =  gpos.phi();
    r = gpos.perp();
  }  

  float deltaphi = phicenter-phin;
  float x = (tan(phi/4096.)-tan(deltaphi))*(r*cos(deltaphi) - zcn_[st-1]); //zcn is in local coordinates -> z invreases approching to vertex
  if (hasPosRF(wh,sec)){ x = -x; } // change sign in case of positive wheels
  x+=xcenter;

  return x;

}


float DTTrigGeomUtils::trigDir(const L1MuDTChambPhDigi* trig){


  int wh   = trig->whNum();
  int sec  = trig->scNum()+1;
  int phi  = trig->phi();
  int phib = trig->phiB();

  float dir = (phib/512.+phi/4096.)*radToDeg_;

  // change sign in case of negative wheels
  if (!hasPosRF(wh,sec)) { dir = -dir; }

  return dir;

}

