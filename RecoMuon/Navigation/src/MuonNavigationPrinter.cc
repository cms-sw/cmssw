/** \class MuonNavigationPrinter
 *
 * Description:
 *  class to print the MuonNavigationSchool
 *
 * $Date: 2013/05/28 16:39:22 $
 * $Revision: 1.11 $
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 * Chang Liu:
 * add compatibleLayers
 * add constructor for MuonTkNavigation
 */

#include "RecoMuon/Navigation/interface/MuonNavigationPrinter.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h" 
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <iomanip>
using namespace std;

MuonNavigationPrinter::MuonNavigationPrinter(const MuonDetLayerGeometry * muonLayout, bool enableRPC) {

  edm::LogInfo ("MuonNavigationPrinter")<< "MuonNavigationPrinter::MuonNavigationPrinter" ;
  vector<DetLayer*>::const_iterator iter;
  edm::LogInfo ("MuonNavigationPrinter")<<"================================";
  edm::LogInfo ("MuonNavigationPrinter")<< "BARREL:";
  vector<DetLayer*> barrel;
  if ( enableRPC ) barrel = muonLayout->allBarrelLayers();
  else barrel = muonLayout->allDTLayers();

  edm::LogInfo ("MuonNavigationPrinter")<<"There are "<<barrel.size()<<" Barrel DetLayers";
  for ( iter = barrel.begin(); iter != barrel.end(); iter++ ) printLayer(*iter);
  edm::LogInfo ("MuonNavigationPrinter")<<"================================";
  edm::LogInfo ("MuonNavigationPrinter")  << "BACKWARD:";

  vector<DetLayer*> backward;
  if ( enableRPC ) backward = muonLayout->allBackwardLayers();
  else backward = muonLayout->backwardCSCLayers();

  edm::LogInfo ("MuonNavigationPrinter")<<"There are "<<backward.size()<<" Backward DetLayers";
  for ( iter = backward.begin(); iter != backward.end(); iter++ ) printLayer(*iter);
  edm::LogInfo ("MuonNavigationPrinter") << "==============================";
  edm::LogInfo ("MuonNavigationPrinter") << "FORWARD:";
  vector<DetLayer*> forward;
  if ( enableRPC ) forward = muonLayout->allForwardLayers();
  else forward = muonLayout->forwardCSCLayers();

  edm::LogInfo ("MuonNavigationPrinter")<<"There are "<<forward.size()<<" Forward DetLayers";
  for ( iter = forward.begin(); iter != forward.end(); iter++ ) printLayer(*iter);

}

MuonNavigationPrinter::MuonNavigationPrinter(const MuonDetLayerGeometry * muonLayout, const GeometricSearchTracker * tracker) {

  edm::LogInfo ("MuonNavigationPrinter")<< "MuonNavigationPrinter::MuonNavigationPrinter" ;
  vector<DetLayer*>::const_iterator iter;
//  vector<BarrelDetLayer*>::const_iterator tkiter;
//  vector<ForwardDetLayer*>::const_iterator tkfiter;
  edm::LogInfo ("MuonNavigationPrinter")<<"================================";
  edm::LogInfo ("MuonNavigationPrinter")<< "BARREL:";
  vector<BarrelDetLayer*> tkbarrel = tracker->barrelLayers();
  edm::LogInfo ("MuonNavigationPrinter")<<"There are "<<tkbarrel.size()<<" Tk Barrel DetLayers";
//  for ( tkiter = tkbarrel.begin(); tkiter != tkbarrel.end(); tkiter++ ) printLayer(*tkiter);
  vector<DetLayer*> barrel = muonLayout->allBarrelLayers();
  edm::LogInfo ("MuonNavigationPrinter")<<"There are "<<barrel.size()<<" Mu Barrel DetLayers";
  for ( iter = barrel.begin(); iter != barrel.end(); iter++ ) printLayer(*iter);
  edm::LogInfo ("MuonNavigationPrinter")<<"================================";
  edm::LogInfo ("MuonNavigationPrinter")  << "BACKWARD:";
  vector<ForwardDetLayer*> tkbackward = tracker->negForwardLayers();
  edm::LogInfo ("MuonNavigationPrinter")<<"There are "<<tkbackward.size()<<" Tk Backward DetLayers";
///  for ( tkfiter = tkbackward.begin(); tkfiter != tkbackward.end(); tkfiter++ ) printLayer(*tkfiter);
  vector<DetLayer*> backward = muonLayout->allBackwardLayers();
  edm::LogInfo ("MuonNavigationPrinter")<<"There are "<<backward.size()<<" Mu Backward DetLayers";
  for ( iter = backward.begin(); iter != backward.end(); iter++ ) printLayer(*iter);
  edm::LogInfo ("MuonNavigationPrinter") << "==============================";
  edm::LogInfo ("MuonNavigationPrinter") << "FORWARD:";
  vector<ForwardDetLayer*> tkforward =  tracker->posForwardLayers();
  edm::LogInfo ("MuonNavigationPrinter")<<"There are "<<tkforward.size()<<" Tk Forward DetLayers";
//  for ( tkfiter = tkforward.begin(); tkfiter != tkforward.end(); tkfiter++ ) printLayer(*tkfiter);

  vector<DetLayer*> forward = muonLayout->allForwardLayers();
  edm::LogInfo ("MuonNavigationPrinter")<<"There are "<<forward.size()<<" Mu Forward DetLayers";
  for ( iter = forward.begin(); iter != forward.end(); iter++ ) printLayer(*iter);

}

/// print layer
void MuonNavigationPrinter::printLayer(DetLayer* layer) const {
  vector<const DetLayer*> nextLayers = layer->nextLayers(insideOut);
  vector<const DetLayer*> compatibleLayers = layer->compatibleLayers(insideOut);
  if (BarrelDetLayer* bdl = dynamic_cast<BarrelDetLayer*>(layer)) {
    edm::LogInfo ("MuonNavigationPrinter") 
         << layer->location() << " " << layer->subDetector() << " layer at R: "
         << setiosflags(ios::showpoint | ios::fixed)
         << setw(8) << setprecision(2)
         << bdl->specificSurface().radius() << "  length: "
         << setw(6) << setprecision(2)
         << layer->surface().bounds().length();
          
  }
  else if (ForwardDetLayer* fdl = dynamic_cast<ForwardDetLayer*>(layer)) {
    edm::LogInfo ("MuonNavigationPrinter") << endl
         << layer->location() << " " << layer->subDetector() << "layer at z: "
         << setiosflags(ios::showpoint | ios::fixed)
         << setw(8) << setprecision(2)
         << layer->surface().position().z() << "  inner r: "
         << setw(6) << setprecision(2)
         << fdl->specificSurface().innerRadius() << "  outer r: "
         << setw(6) << setprecision(2)
         << fdl->specificSurface().outerRadius();
  }
  edm::LogInfo ("MuonNavigationPrinter") << " has " << nextLayers.size() << " next layers in the direction inside-out: ";
  printLayers(nextLayers);

  nextLayers.clear();
  nextLayers = layer->nextLayers(outsideIn);

   edm::LogInfo ("MuonNavigationPrinter") << " has " << nextLayers.size() << " next layers in the direction outside-in: ";
  printLayers(nextLayers);

  edm::LogInfo ("MuonNavigationPrinter") << " has " << compatibleLayers.size() << " compatible layers in the direction inside-out:: ";
  printLayers(compatibleLayers);
  compatibleLayers.clear();
  compatibleLayers = layer->compatibleLayers(outsideIn);
  
  edm::LogInfo ("MuonNavigationPrinter") << " has " << compatibleLayers.size() << " compatible layers in the direction outside-in: ";
  printLayers(compatibleLayers);

}

/// print next layers
void MuonNavigationPrinter::printLayers(const vector<const DetLayer*>& nextLayers) const {

  for ( vector<const DetLayer*>::const_iterator inext = nextLayers.begin();
      inext != nextLayers.end(); inext++ ) {

     edm::LogInfo ("MuonNavigationPrinter") << " --> "; 
     if ( (*inext)->location() == GeomDetEnumerators::barrel ) {
      const BarrelDetLayer* l = dynamic_cast<const BarrelDetLayer*>(&(**inext));
      edm::LogInfo ("MuonNavigationPrinter") << (*inext)->location() << " "
           << (*inext)->subDetector()
           << " layer at R: "
           << setiosflags(ios::showpoint | ios::fixed)
           << setw(8) << setprecision(2)
           << l->specificSurface().radius() << "   ";
    }
    else {
      const ForwardDetLayer* l = dynamic_cast<const ForwardDetLayer*>(&(**inext));
       edm::LogInfo ("MuonNavigationPrinter") << (*inext)->location() << " "
           << (*inext)->subDetector()
           << " layer at z: "
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


/// These should not be useful anymore as SubDetector and Location enums now have << operators

// /// determine whether the layer is forward or backward 
// string MuonNavigationPrinter::layerPart(const DetLayer* layer) const {

//   string result = "unknown";
  
//   if ( layer->part() == barrel ) return "barrel";
//   if ( layer->part() == forward && layer->surface().position().z() < 0 ) {
//     result = "backward"; 
//   }
//   if ( layer->part() == forward && layer->surface().position().z() >= 0 ) {
//     result = "forward";
//   }
  
//   return result;    

// }

// /// determine the module (pixel, sililcon, msgc, dt, csc, rpc)
// string MuonNavigationPrinter::layerModule(const DetLayer* layer) const {

//   string result = "unknown";

//   GeomDetEnumerators::SubDetector det = layer->subDetector();

//   if ( det == Pixel ) return "Pixel";
//   if ( det == TIB || det == TOB
//        || det == TID || det == TEC ) return "silicon";
//   if ( det == DT ) return "DT";
//   if ( det == CSC ) return "CSC";
//   if ( det == RPC ) return "RPC";

//   return result;

// }

