/** \class MuonNavigationPrinter
 *
 * Description:
 *  class to print the MuonNavigationSchool
 *
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 * Chang Liu:
 * add compatibleLayers
 * add constructor for MuonTkNavigation
 *
 * Cesare Calabria:
 * GEMs implementation.
 */

#include "RecoMuon/Navigation/interface/MuonNavigationPrinter.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h" 
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <iomanip>
using namespace std;

// #define VI_DEBUG

#ifdef VI_DEBUG
#define PRINT(x) std::cout << x << ' '
#else
#define PRINT(x) edm::LogInfo(x)
#endif

MuonNavigationPrinter::MuonNavigationPrinter(const MuonDetLayerGeometry * muonLayout,  MuonNavigationSchool const & sh,   bool enableCSC, bool enableRPC, bool enableGEM) :
  school(&sh) {

  PRINT("MuonNavigationPrinter")<< "MuonNavigationPrinter::MuonNavigationPrinter" << std::endl;
  PRINT("MuonNavigationPrinter")<<"================================" << std::endl;
  PRINT("MuonNavigationPrinter")<< "BARREL:" << std::endl;
  vector<const DetLayer*> barrel;
  if ( enableRPC ) barrel = muonLayout->allBarrelLayers();
  else barrel = muonLayout->allDTLayers();

  PRINT("MuonNavigationPrinter")<<"There are "<<barrel.size()<<" Barrel DetLayers";
  for (auto i: barrel ) printLayer(i);
  PRINT("MuonNavigationPrinter")<<"================================" << std::endl;
  PRINT("MuonNavigationPrinter")  << "BACKWARD:" << std::endl;

  vector<const DetLayer*> backward;
  if ( enableCSC & enableGEM & enableRPC ) backward = muonLayout->allBackwardLayers();
  else if ( enableCSC & enableGEM & !enableRPC ) backward = muonLayout->allCscGemBackwardLayers(); // CSC + GEM
  else if ( !enableCSC & enableGEM & !enableRPC ) backward = muonLayout->backwardGEMLayers(); //GEM only
  else if ( enableCSC & !enableGEM & !enableRPC ) backward = muonLayout->backwardCSCLayers(); //CSC only
  else backward = muonLayout->allBackwardLayers();

  PRINT("MuonNavigationPrinter")<<"There are "<<backward.size()<<" Backward DetLayers";
  for (auto i : backward ) printLayer(i);
  PRINT("MuonNavigationPrinter") << "==============================" << std::endl;
  PRINT("MuonNavigationPrinter") << "FORWARD:" << std::endl;
  vector<const DetLayer*> forward;
  if ( enableCSC & enableGEM & enableRPC ) forward = muonLayout->allForwardLayers();
  else if ( enableCSC & enableGEM & !enableRPC ) forward = muonLayout->allCscGemForwardLayers(); // CSC + GEM
  else if ( !enableCSC & enableGEM & !enableRPC ) forward = muonLayout->forwardGEMLayers(); //GEM only
  else if ( enableCSC & !enableGEM & !enableRPC ) forward = muonLayout->forwardCSCLayers(); //CSC only
  else forward = muonLayout->allForwardLayers();

  PRINT("MuonNavigationPrinter")<<"There are "<<forward.size()<<" Forward DetLayers" << std::endl;
  for (auto i : forward ) printLayer(i);

}

MuonNavigationPrinter::MuonNavigationPrinter(const MuonDetLayerGeometry * muonLayout,  MuonNavigationSchool const & sh, const GeometricSearchTracker * tracker)  :
  school(&sh){

  PRINT("MuonNavigationPrinter")<< "MuonNavigationPrinter::MuonNavigationPrinter" << std::endl ;
//  vector<BarrelDetLayer*>::const_iterator tkiter;
//  vector<ForwardDetLayer*>::const_iterator tkfiter;
  PRINT("MuonNavigationPrinter")<<"================================" << std::endl;
  PRINT("MuonNavigationPrinter")<< "BARREL:" << std::endl;
  vector<const BarrelDetLayer*> tkbarrel = tracker->barrelLayers();
  PRINT("MuonNavigationPrinter")<<"There are "<<tkbarrel.size()<<" Tk Barrel DetLayers" << std::endl;
//  for ( tkiter = tkbarrel.begin(); tkiter != tkbarrel.end(); tkiter++ ) printLayer(*tkiter);
  vector<const DetLayer*> barrel = muonLayout->allBarrelLayers();
  PRINT("MuonNavigationPrinter")<<"There are "<<barrel.size()<<" Mu Barrel DetLayers";
  for ( auto i : barrel ) printLayer(i);
  PRINT("MuonNavigationPrinter")<<"================================" << std::endl;
  PRINT("MuonNavigationPrinter")  << "BACKWARD:" << std::endl;
  vector<const ForwardDetLayer*> tkbackward = tracker->negForwardLayers();
  PRINT("MuonNavigationPrinter")<<"There are "<<tkbackward.size()<<" Tk Backward DetLayers" << std::endl;
///  for ( tkfiter = tkbackward.begin(); tkfiter != tkbackward.end(); tkfiter++ ) printLayer(*tkfiter);
  vector<const DetLayer*> backward = muonLayout->allBackwardLayers();
  PRINT("MuonNavigationPrinter")<<"There are "<<backward.size()<<" Mu Backward DetLayers << std::endl";
  for (auto i : backward ) printLayer(i);
  PRINT("MuonNavigationPrinter") << "==============================" << std::endl;
  PRINT("MuonNavigationPrinter") << "FORWARD:" << std::endl;
  vector<const ForwardDetLayer*> tkforward =  tracker->posForwardLayers();
  PRINT("MuonNavigationPrinter")<<"There are "<<tkforward.size()<<" Tk Forward DetLayers" << std::endl;
//  for ( tkfiter = tkforward.begin(); tkfiter != tkforward.end(); tkfiter++ ) printLayer(*tkfiter);

  vector<const DetLayer*> forward = muonLayout->allForwardLayers();
  PRINT("MuonNavigationPrinter")<<"There are "<<forward.size()<<" Mu Forward DetLayers";
  for ( auto i : forward ) printLayer(i);

}

/// print layer
void MuonNavigationPrinter::printLayer(const DetLayer* layer) const {
  vector<const DetLayer*> nextLayers = school->nextLayers(*layer,insideOut);
  vector<const DetLayer*> compatibleLayers = school->compatibleLayers(*layer,insideOut);
  if (const BarrelDetLayer* bdl = dynamic_cast<const BarrelDetLayer*>(layer)) {
    PRINT("MuonNavigationPrinter") 
         << layer->location() << " " << layer->subDetector() << " layer at R: "
         << setiosflags(ios::showpoint | ios::fixed)
         << setw(8) << setprecision(2)
         << bdl->specificSurface().radius() << "  length: "
         << setw(6) << setprecision(2)
         << layer->surface().bounds().length() << std::endl;
          
  }
  else if (const ForwardDetLayer* fdl = dynamic_cast<const ForwardDetLayer*>(layer)) {
    PRINT("MuonNavigationPrinter") << endl
         << layer->location() << " " << layer->subDetector() << "layer at z: "
         << setiosflags(ios::showpoint | ios::fixed)
         << setw(8) << setprecision(2)
         << layer->surface().position().z() << "  inner r: "
         << setw(6) << setprecision(2)
         << fdl->specificSurface().innerRadius() << "  outer r: "
         << setw(6) << setprecision(2)
         << fdl->specificSurface().outerRadius() << std::endl;
  }
  PRINT("MuonNavigationPrinter") << " has " << nextLayers.size() << " next layers in the direction inside-out: " << std::endl;
  printLayers(nextLayers);

  nextLayers.clear();
  nextLayers = school->nextLayers(*layer,outsideIn);

   PRINT("MuonNavigationPrinter") << " has " << nextLayers.size() << " next layers in the direction outside-in: " << std::endl;
  printLayers(nextLayers);

  PRINT("MuonNavigationPrinter") << " has " << compatibleLayers.size() << " compatible layers in the direction inside-out:: " << std::endl;
  printLayers(compatibleLayers);
  compatibleLayers.clear();
  compatibleLayers = school->compatibleLayers(*layer,outsideIn);
  
  PRINT("MuonNavigationPrinter") << " has " << compatibleLayers.size() << " compatible layers in the direction outside-in: " << std::endl;
  printLayers(compatibleLayers);

}

/// print next layers
void MuonNavigationPrinter::printLayers(const vector<const DetLayer*>& nextLayers) const {

  for ( vector<const DetLayer*>::const_iterator inext = nextLayers.begin();
      inext != nextLayers.end(); inext++ ) {

     PRINT("MuonNavigationPrinter") << " --> " << std::endl; 
     if ( (*inext)->location() == GeomDetEnumerators::barrel ) {
      const BarrelDetLayer* l = dynamic_cast<const BarrelDetLayer*>(&(**inext));
      PRINT("MuonNavigationPrinter") << (*inext)->location() << " "
           << (*inext)->subDetector()
           << " layer at R: "
           << setiosflags(ios::showpoint | ios::fixed)
           << setw(8) << setprecision(2)
           << l->specificSurface().radius() << "   " << std::endl;
    }
    else {
      const ForwardDetLayer* l = dynamic_cast<const ForwardDetLayer*>(&(**inext));
       PRINT("MuonNavigationPrinter") << (*inext)->location() << " "
           << (*inext)->subDetector()
           << " layer at z: "
           << setiosflags(ios::showpoint | ios::fixed)
           << setw(8) << setprecision(2)
           << l->surface().position().z() << "   " << std::endl;
    }
    PRINT("MuonNavigationPrinter") << setiosflags(ios::showpoint | ios::fixed)
         << setprecision(1)
         << setw(6) << (*inext)->surface().bounds().length() << ", "
         << setw(6) << (*inext)->surface().bounds().width() << ", "
         << setw(4) <<(*inext)->surface().bounds().thickness() << " : " 
         << (*inext)->surface().position() << std::endl;
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

