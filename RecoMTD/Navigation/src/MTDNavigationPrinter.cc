/** \class MTDNavigationPrinter
 *
 * Description:
 *  class to print the MTDNavigationSchool
 *
 *
 * \author : L. Gray - FNAL
 * 
 */

#include "RecoMTD/Navigation/interface/MTDNavigationPrinter.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h" 
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoMTD/Navigation/interface/MTDNavigationSchool.h"
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

MTDNavigationPrinter::MTDNavigationPrinter(const MTDDetLayerGeometry * mtdLayout,  MTDNavigationSchool const & sh,   bool enableBTL, bool enableETL) :
  school(&sh) {

  PRINT("MTDNavigationPrinter")<< "MTDNavigationPrinter::MTDNavigationPrinter" << std::endl;
  PRINT("MTDNavigationPrinter")<<"================================" << std::endl;
  PRINT("MTDNavigationPrinter")<< "BARREL:" << std::endl;
  vector<const DetLayer*> barrel;
  if ( enableBTL ) barrel = mtdLayout->allBarrelLayers();
  else barrel = mtdLayout->allBarrelLayers();

  PRINT("MTDNavigationPrinter")<<"There are "<<barrel.size()<<" Barrel DetLayers";
  for (auto i: barrel ) printLayer(i);
  PRINT("MTDNavigationPrinter")<<"================================" << std::endl;
  PRINT("MTDNavigationPrinter")  << "BACKWARD:" << std::endl;

  vector<const DetLayer*> backward;

  if ( enableETL ) backward = mtdLayout->allBackwardLayers();  
  else backward = mtdLayout->allBackwardLayers();

  PRINT("MTDNavigationPrinter")<<"There are "<<backward.size()<<" Backward DetLayers";
  for (auto i : backward ) printLayer(i);
  PRINT("MTDNavigationPrinter") << "==============================" << std::endl;
  PRINT("MTDNavigationPrinter") << "FORWARD:" << std::endl;
  vector<const DetLayer*> forward;

  if ( enableETL ) forward = mtdLayout->allForwardLayers();  
  else forward = mtdLayout->allForwardLayers();

  PRINT("MTDNavigationPrinter")<<"There are "<<forward.size()<<" Forward DetLayers" << std::endl;
  for (auto i : forward ) printLayer(i);

}

MTDNavigationPrinter::MTDNavigationPrinter(const MTDDetLayerGeometry * mtdLayout,  MTDNavigationSchool const & sh, const GeometricSearchTracker * tracker)  :
  school(&sh){

  PRINT("MTDNavigationPrinter")<< "MTDNavigationPrinter::MTDNavigationPrinter" << std::endl ;
//  vector<BarrelDetLayer*>::const_iterator tkiter;
//  vector<ForwardDetLayer*>::const_iterator tkfiter;
  PRINT("MTDNavigationPrinter")<<"================================" << std::endl;
  PRINT("MTDNavigationPrinter")<< "BARREL:" << std::endl;
  const vector<const BarrelDetLayer*>& tkbarrel = tracker->barrelLayers();
  PRINT("MTDNavigationPrinter")<<"There are "<<tkbarrel.size()<<" Tk Barrel DetLayers" << std::endl;
//  for ( tkiter = tkbarrel.begin(); tkiter != tkbarrel.end(); tkiter++ ) printLayer(*tkiter);
  vector<const DetLayer*> barrel = mtdLayout->allBarrelLayers();
  PRINT("MTDNavigationPrinter")<<"There are "<<barrel.size()<<" Mu Barrel DetLayers";
  for ( auto i : barrel ) printLayer(i);
  PRINT("MTDNavigationPrinter")<<"================================" << std::endl;
  PRINT("MTDNavigationPrinter")  << "BACKWARD:" << std::endl;
  const vector<const ForwardDetLayer*>& tkbackward = tracker->negForwardLayers();
  PRINT("MTDNavigationPrinter")<<"There are "<<tkbackward.size()<<" Tk Backward DetLayers" << std::endl;
///  for ( tkfiter = tkbackward.begin(); tkfiter != tkbackward.end(); tkfiter++ ) printLayer(*tkfiter);
  vector<const DetLayer*> backward = mtdLayout->allBackwardLayers();
  PRINT("MTDNavigationPrinter")<<"There are "<<backward.size()<<" Mu Backward DetLayers << std::endl";
  for (auto i : backward ) printLayer(i);
  PRINT("MTDNavigationPrinter") << "==============================" << std::endl;
  PRINT("MTDNavigationPrinter") << "FORWARD:" << std::endl;
  const vector<const ForwardDetLayer*>& tkforward =  tracker->posForwardLayers();
  PRINT("MTDNavigationPrinter")<<"There are "<<tkforward.size()<<" Tk Forward DetLayers" << std::endl;
//  for ( tkfiter = tkforward.begin(); tkfiter != tkforward.end(); tkfiter++ ) printLayer(*tkfiter);

  vector<const DetLayer*> forward = mtdLayout->allForwardLayers();
  PRINT("MTDNavigationPrinter")<<"There are "<<forward.size()<<" Mu Forward DetLayers";
  for ( auto i : forward ) printLayer(i);

}

/// print layer
void MTDNavigationPrinter::printLayer(const DetLayer* layer) const {
  vector<const DetLayer*> nextLayers = school->nextLayers(*layer,insideOut);
  vector<const DetLayer*> compatibleLayers = school->compatibleLayers(*layer,insideOut);
  if (const BarrelDetLayer* bdl = dynamic_cast<const BarrelDetLayer*>(layer)) {
    PRINT("MTDNavigationPrinter") 
         << layer->location() << " " << layer->subDetector() << " layer at R: "
         << setiosflags(ios::showpoint | ios::fixed)
         << setw(8) << setprecision(2)
         << bdl->specificSurface().radius() << "  length: "
         << setw(6) << setprecision(2)
         << layer->surface().bounds().length() << std::endl;
          
  }
  else if (const ForwardDetLayer* fdl = dynamic_cast<const ForwardDetLayer*>(layer)) {
    PRINT("MTDNavigationPrinter") << endl
         << layer->location() << " " << layer->subDetector() << "layer at z: "
         << setiosflags(ios::showpoint | ios::fixed)
         << setw(8) << setprecision(2)
         << layer->surface().position().z() << "  inner r: "
         << setw(6) << setprecision(2)
         << fdl->specificSurface().innerRadius() << "  outer r: "
         << setw(6) << setprecision(2)
         << fdl->specificSurface().outerRadius() << std::endl;
  }
  PRINT("MTDNavigationPrinter") << " has " << nextLayers.size() << " next layers in the direction inside-out: " << std::endl;
  printLayers(nextLayers);

  nextLayers.clear();
  nextLayers = school->nextLayers(*layer,outsideIn);

   PRINT("MTDNavigationPrinter") << " has " << nextLayers.size() << " next layers in the direction outside-in: " << std::endl;
  printLayers(nextLayers);

  PRINT("MTDNavigationPrinter") << " has " << compatibleLayers.size() << " compatible layers in the direction inside-out:: " << std::endl;
  printLayers(compatibleLayers);
  compatibleLayers.clear();
  compatibleLayers = school->compatibleLayers(*layer,outsideIn);
  
  PRINT("MTDNavigationPrinter") << " has " << compatibleLayers.size() << " compatible layers in the direction outside-in: " << std::endl;
  printLayers(compatibleLayers);

}

/// print next layers
void MTDNavigationPrinter::printLayers(const vector<const DetLayer*>& nextLayers) const {

  for ( vector<const DetLayer*>::const_iterator inext = nextLayers.begin();
      inext != nextLayers.end(); inext++ ) {

     PRINT("MTDNavigationPrinter") << " --> " << std::endl; 
     if ( (*inext)->location() == GeomDetEnumerators::barrel ) {
      const BarrelDetLayer* l = dynamic_cast<const BarrelDetLayer*>(&(**inext));
      PRINT("MTDNavigationPrinter") << (*inext)->location() << " "
           << (*inext)->subDetector()
           << " layer at R: "
           << setiosflags(ios::showpoint | ios::fixed)
           << setw(8) << setprecision(2)
           << l->specificSurface().radius() << "   " << std::endl;
    }
    else {
      const ForwardDetLayer* l = dynamic_cast<const ForwardDetLayer*>(&(**inext));
       PRINT("MTDNavigationPrinter") << (*inext)->location() << " "
           << (*inext)->subDetector()
           << " layer at z: "
           << setiosflags(ios::showpoint | ios::fixed)
           << setw(8) << setprecision(2)
           << l->surface().position().z() << "   " << std::endl;
    }
    PRINT("MTDNavigationPrinter") << setiosflags(ios::showpoint | ios::fixed)
         << setprecision(1)
         << setw(6) << (*inext)->surface().bounds().length() << ", "
         << setw(6) << (*inext)->surface().bounds().width() << ", "
         << setw(4) <<(*inext)->surface().bounds().thickness() << " : " 
         << (*inext)->surface().position() << std::endl;
  }

}
