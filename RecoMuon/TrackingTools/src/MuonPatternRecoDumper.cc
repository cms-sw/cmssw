
// This Class Header 
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

// Collaborating Class Header
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

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
  cout<<"dir: "<<tsos.globalDirection();
  dumpFTS(*tsos.freeTrajectoryState());
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
    " radius: " << fts.position().perp() << "\n" << 
    " charge*pt: " << fts.momentum().perp()*fts.parameters().charge() <<
    " eta: " << fts.momentum().eta() <<
    " phi: " << fts.momentum().phi() << endl;
}

void MuonPatternRecoDumper::dumpTSOS(TrajectoryStateOnSurface& tsos,std::string &where) const{
  LogDebug(where) <<"dir: "<<tsos.globalDirection();
  dumpFTS(*tsos.freeTrajectoryState(),where);
}


void MuonPatternRecoDumper::dumpMuonId(const DetId &id, std::string &where) const{
  
  if(id.subdetId() == MuonSubdetId::DT ){
    DTChamberId chamberId(id.rawId());
    LogDebug(where)<<"(DT): "<<chamberId<<endl;  
  }
  else if(id.subdetId() == MuonSubdetId::CSC){
    CSCDetId chamberId(id.rawId());
    LogDebug(where)<<"(CSC): "<<chamberId<<endl;  
  }
  else if(id.subdetId() == MuonSubdetId::RPC){
    RPCDetId chamberId(id.rawId());
    LogDebug(where)<<"(RPC): "<<chamberId<<endl;  
  }
  else edm::LogWarning(where)<<"The DetLayer is not a valid Muon DetLayer. ";
}

void MuonPatternRecoDumper::dumpMuonId(const DetId &id) const{
  if(id.subdetId() == MuonSubdetId::DT ){
    DTChamberId chamberId(id.rawId());
    cout<<"(DT): "<<chamberId<<endl;  
  }
  else if(id.subdetId() == MuonSubdetId::CSC){
    CSCDetId chamberId(id.rawId());
    cout<<"(CSC): "<<chamberId<<endl;  
  }
  else if(id.subdetId() == MuonSubdetId::RPC){
    RPCDetId chamberId(id.rawId());
    cout<<"(RPC): "<<chamberId<<endl;  
  }
  else cout<<"The DetLayer is not a valid Muon DetLayer. ";
}
