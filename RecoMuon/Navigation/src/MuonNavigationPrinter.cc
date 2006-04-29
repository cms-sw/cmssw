/** \class MuonNavigationPrinter
 *
 * Description:
 *  class to print the MuonNavigationSchool
 *
 * $Date: 2006/04/24 20:03:36 $
 * $Revision: 1.2 $
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 * Chang Liu:
 * add compatibleLayers
 */

#include "RecoMuon/Navigation/interface/MuonNavigationPrinter.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h" 
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <iomanip>
using namespace std;

MuonNavigationPrinter::MuonNavigationPrinter(const MuonDetLayerGeometry * muonLayout) {

  edm::LogInfo ("MuonNavigationPrinter")<< "MuonNavigationPrinter::MuonNavigationPrinter" ;
  vector<DetLayer*>::const_iterator iter;
  edm::LogInfo ("MuonNavigationPrinter")<<"================================";
  edm::LogInfo ("MuonNavigationPrinter")<< "BARREL:";
  vector<DetLayer*> barrel = muonLayout->allBarrelLayers();
  edm::LogInfo ("MuonNavigationPrinter")<<"There are "<<barrel.size()<<" Barrel DetLayers";
  for ( iter = barrel.begin(); iter != barrel.end(); iter++ ) printLayer(*iter);
  edm::LogInfo ("MuonNavigationPrinter");
  edm::LogInfo ("MuonNavigationPrinter")  << "BACKWARD:";
  vector<DetLayer*> backward = muonLayout->allBackwardLayers();
  edm::LogInfo ("MuonNavigationPrinter")<<"There are "<<backward.size()<<" Backward DetLayers";
  for ( iter = backward.begin(); iter != backward.end(); iter++ ) printLayer(*iter);
  edm::LogInfo ("MuonNavigationPrinter") << "==============================";
  edm::LogInfo ("MuonNavigationPrinter") << "FORWARD:";
  vector<DetLayer*> forward = muonLayout->allForwardLayers();
  edm::LogInfo ("MuonNavigationPrinter")<<"There are "<<forward.size()<<" Forward DetLayers";
  for ( iter = forward.begin(); iter != forward.end(); iter++ ) printLayer(*iter);

}


/// print layer
void MuonNavigationPrinter::printLayer(DetLayer* layer) const {

  vector<const DetLayer*> nextLayers = layer->nextLayers(alongMomentum);
  vector<const DetLayer*> compatibleLayers = layer->compatibleLayers(alongMomentum);

  if (BarrelDetLayer* bdl = dynamic_cast<BarrelDetLayer*>(layer)) {
    edm::LogInfo ("MuonNavigationPrinter") << endl
         << layerPart(layer) << " " << layerModule(layer) << "layer at R: "
         << setiosflags(ios::showpoint | ios::fixed)
         << setw(8) << setprecision(2)
         << bdl->specificSurface().radius() << "  length: "
         << setw(6) << setprecision(2)
         << layer->surface().bounds().length();
          
  }
  else if (ForwardDetLayer* fdl = dynamic_cast<ForwardDetLayer*>(layer)) {
    edm::LogInfo ("MuonNavigationPrinter") << endl
         << layerPart(layer) << " " << layerModule(layer) << "layer at z: "
         << setiosflags(ios::showpoint | ios::fixed)
         << setw(8) << setprecision(2)
         << layer->surface().position().z() << "  inner r: "
         << setw(6) << setprecision(2)
         << fdl->specificSurface().innerRadius() << "  outer r: "
         << setw(6) << setprecision(2)
         << fdl->specificSurface().outerRadius();
  }
  edm::LogInfo ("MuonNavigationPrinter");
  edm::LogInfo ("MuonNavigationPrinter") << " has " << nextLayers.size() << " next layers along momentum: ";
  printNextLayers(nextLayers);

  nextLayers.clear();
  nextLayers = layer->nextLayers(oppositeToMomentum);

   edm::LogInfo ("MuonNavigationPrinter") << " has " << nextLayers.size() << " next layers opposite to momentum: ";
  printNextLayers(nextLayers);

  edm::LogInfo ("MuonNavigationPrinter") << " has " << compatibleLayers.size() << " compatible layers along momentum: ";
  printNextLayers(compatibleLayers);
  compatibleLayers.clear();
  compatibleLayers = layer->compatibleLayers(oppositeToMomentum);

   edm::LogInfo ("MuonNavigationPrinter") << " has " << compatibleLayers.size() << " compatible layers opposite to momentum: ";
  printNextLayers(compatibleLayers);

}

/// print next layers
void MuonNavigationPrinter::printNextLayers(vector<const DetLayer*> nextLayers) const {

  for ( vector<const DetLayer*>::const_iterator inext = nextLayers.begin();
      inext != nextLayers.end(); inext++ ) {

     edm::LogInfo ("MuonNavigationPrinter") << " --> "; 
    if ( (*inext)->part() == barrel ) {
      const BarrelDetLayer* l = dynamic_cast<const BarrelDetLayer*>(&(**inext));
       edm::LogInfo ("MuonNavigationPrinter") << layerPart(*inext) << "   "
           << layerModule(*inext);
      if ( (*inext)->module() == dt )  edm::LogInfo ("MuonNavigationPrinter") << " ";
       edm::LogInfo ("MuonNavigationPrinter") << "layer at R: "
           << setiosflags(ios::showpoint | ios::fixed)
           << setw(8) << setprecision(2)
           << l->specificSurface().radius() << "   ";
    }
    else {
      const ForwardDetLayer* l = dynamic_cast<const ForwardDetLayer*>(&(**inext));
       edm::LogInfo ("MuonNavigationPrinter") << layerPart(*inext) << " ";
           if ( layerPart(*inext) == "forward" )  edm::LogInfo ("MuonNavigationPrinter") << " ";
       edm::LogInfo ("MuonNavigationPrinter") << layerModule(*inext)
           << "layer at z: "
           << setiosflags(ios::showpoint | ios::fixed)
           << setw(8) << setprecision(2)
           << l->surface().position().z() << "   ";
    }
    edm::LogInfo ("MuonNavigationPrinter") << setiosflags(ios::showpoint | ios::fixed)
         << setprecision(1)
         << setw(6) << (*inext)->surface().bounds().length() << ", "
         << setw(6) << (*inext)->surface().bounds().width() << ", "
         << setw(4) <<(*inext)->surface().bounds().thickness() << " : " 
         << (*inext)->surface().position();
  }

}


/// determine whether the layer is forward or backward 
string MuonNavigationPrinter::layerPart(const DetLayer* layer) const {

  string result = "forward";
  
  if ( layer->part() == barrel ) return "barrel";
  if ( layer->part() == forward && layer->surface().position().z() < 0 ) {
    result = "backward"; 
  }
  
  return result;    

}

/// determine the module (pixel, sililcon, msgc, dt, csc, rpc)
string MuonNavigationPrinter::layerModule(const DetLayer* layer) const {

  string result = "unknown";

  if ( layer->module() == pixel ) return "pixel";
  if ( layer->module() == silicon ) return "silicon";
  if ( layer->module() == msgc ) return "msgc";
  if ( layer->module() == dt ) return "dt";
  if ( layer->module() == csc ) return "csc";
  if ( layer->module() == rpc ) return "rpc";

  return result;

}

