#ifndef RPCTechnicalTrigger_RBCEmulator_h
#define RPCTechnicalTrigger_RBCEmulator_h

/**  \class RBCEmulator
 *
 *  \author M. Maggi, C. Viviani, D. Pagano - University of Pavia & INFN Pavia
 *
 *
 */


#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/RPCDigi/interface/RPCDigi.h"

#include "TFile.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RBCLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCPolicy.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

/*
  class PSimHit;
  class RPCRoll;
  class RPCCluster;
  class RPCGeometry;
*/

class RBCLogic;
class RBCPolicy;
class RBCOutputSignalContainer;
class RBCEmulator {
//: public edm::EDAnalyzer{
  
  
 public:
  
  RBCEmulator(const edm::Event & event, const edm::EventSetup& eventSetup);
  virtual ~RBCEmulator();
  void emulate(RBCPolicy* policy);
  RBCOutputSignalContainer triggers();
  
 private:
  
  RBCLogic* l;
  char  poly;

  bool neighbours; 
  int BX;
  int  majority;
  std::string digiLabel;
  
    
  
};
#endif
