
// This Class Header 
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

// Collaborating Class Header
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Constructor 
MuonPatternRecoDumper::MuonPatternRecoDumper() {
}

// Destructor
MuonPatternRecoDumper::~MuonPatternRecoDumper() {
}

// Operations

void MuonPatternRecoDumper::dumpLayer(const DetLayer* layer) const {
  BoundSurface* sur=0;
  BoundCylinder* bc=0;
  BoundDisk* bd=0;

  // FIXME
  //  cout << " Next layer: " << layer->part() << " " << layer->module() << ":" ;

  // Debug
  sur = (BoundSurface*)&(layer->surface());
  if ( (bc = dynamic_cast<BoundCylinder*>(sur)) ) {
    cout << "  Cylinder of radius: " << bc->radius() << endl;
  }
  else if ( (bd = dynamic_cast<BoundDisk*>(sur)) ) {
    cout << "  Disk at: " <<  bd->position().z() << endl;
  }
}

void MuonPatternRecoDumper::dumpFTS(FreeTrajectoryState& fts) const {
  cout  << 
    " pos: " << fts.position() << 
    " radius: " << fts.position().perp() << endl << 
    " charge*pt: " << fts.momentum().perp()*fts.parameters().charge() <<
    " eta: " << fts.momentum().eta() <<
    " phi: " << fts.momentum().phi() << endl;
}

void MuonPatternRecoDumper::dumpTSOS(TrajectoryStateOnSurface& tsos) const{
  dumpFTS(tsos.freeTrajectoryState());
}

void MuonPatternRecoDumper::dumpLayer(const DetLayer* layer, std::string &where) const {
  BoundSurface* sur=0;
  BoundCylinder* bc=0;
  BoundDisk* bd=0;

  // FIXME
  //  LogDebug(where) << " Next layer: " << layer->part() << " " << layer->module() << ":" ;

  // Debug
  sur = (BoundSurface*)&(layer->surface());
  if ( (bc = dynamic_cast<BoundCylinder*>(sur)) ) {
    LogDebug(where) << "  Cylinder of radius: " << bc->radius() << endl;
  }
  else if ( (bd = dynamic_cast<BoundDisk*>(sur)) ) {
    LogDebug(where) << "  Disk at: " <<  bd->position().z() << endl;
  }
}

void MuonPatternRecoDumper::dumpFTS(FreeTrajectoryState& fts,std::string &where) const {
  LogDebug(where)  << 
    " pos: " << fts.position() << 
    " radius: " << fts.position().perp() << endl << 
    " charge*pt: " << fts.momentum().perp()*fts.parameters().charge() <<
    " eta: " << fts.momentum().eta() <<
    " phi: " << fts.momentum().phi() << endl;
}

void MuonPatternRecoDumper::dumpTSOS(TrajectoryStateOnSurface& tsos,std::string &where) const{
  dumpFTS(tsos.freeTrajectoryState(),where);
}

