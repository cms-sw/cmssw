
// This Class Header 
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

// Collaborating Class Header
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include <sstream>

using namespace std;

// Constructor 
MuonPatternRecoDumper::MuonPatternRecoDumper() {
}

// Destructor
MuonPatternRecoDumper::~MuonPatternRecoDumper() {
}

// Operations

string MuonPatternRecoDumper::dumpLayer(const DetLayer* layer) const {
  stringstream output;
  
  const BoundSurface* sur=0;
  const BoundCylinder* bc=0;
  const BoundDisk* bd=0;

  sur = &(layer->surface());
  if ( (bc = dynamic_cast<const BoundCylinder*>(sur)) ) {
    output << "  Cylinder of radius: " << bc->radius() << endl;
  }
  else if ( (bd = dynamic_cast<const BoundDisk*>(sur)) ) {
    output << "  Disk at: " <<  bd->position().z() << endl;
  }
  return output.str();
}

string MuonPatternRecoDumper::dumpFTS(const FreeTrajectoryState& fts) const {
  stringstream output;
  
  output  << 
    " pos: " << fts.position() << 
    " radius: " << fts.position().perp() << endl << 
    " charge*pt: " << fts.momentum().perp()*fts.parameters().charge() <<
    " eta: " << fts.momentum().eta() <<
    " phi: " << fts.momentum().phi() << endl;

  return output.str();
}

string MuonPatternRecoDumper::dumpTSOS(const TrajectoryStateOnSurface& tsos) const{
  stringstream output;
  
  output<<tsos<<endl;
  output<<"dir: "<<tsos.globalDirection()<<endl;
  output<<dumpFTS(*tsos.freeTrajectoryState());

  return output.str();
}

string MuonPatternRecoDumper::dumpMuonId(const DetId &id) const{
  stringstream output;
  
  if(id.subdetId() == MuonSubdetId::DT ){
    DTWireId wireId(id.rawId());

    output<<"(DT): "<<wireId<<endl;  
  }
  else if(id.subdetId() == MuonSubdetId::CSC){
    CSCDetId chamberId(id.rawId());
    output<<"(CSC): "<<chamberId<<endl;  
  }
  else if(id.subdetId() == MuonSubdetId::GEM){
    GEMDetId chamberId(id.rawId());
    output<<"(GEM): "<<chamberId<<endl;  
  }
  else if(id.subdetId() == MuonSubdetId::RPC){
    RPCDetId chamberId(id.rawId());
    output<<"(RPC): "<<chamberId<<endl;  
  }
  else output<<"The DetLayer is not a valid Muon DetLayer. ";

  return output.str();
}
