#ifndef RPCEventSummary_H
#define RPCEventSummary_H


/** \class RPCEventSummary
 * *
 *  DQM Event Summary module for RPCs
 *
 *  $Date: 2008/03/16 18:52:20 $
 *  $Revision: 1.0 $
 *  \author Anna Cimmino
 *   
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
//#include <FWCore/Framework/interface/Event.h>
//#include <FWCore/Framework/interface/Run.h>
#include <FWCore/Framework/interface/MakerMacros.h>

//#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
//#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DQM/RPCMonitorClient/interface/RPCClient.h"

#include <memory>
#include <string>
#include <map>

class DQMStore;
class RPCDetId;


class RPCEventSummary:public edm::EDAnalyzer {
public:

  /// Constructor
  RPCEventSummary(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCEventSummary();

  /// BeginJob
  void beginJob(const edm::EventSetup& iSetup);

  //Begin Run
   void beginRun(const edm::Run& r, const edm::EventSetup& c);
  
  
  /// Begin Lumi block 
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Analyze  
  void analyze(const edm::Event& iEvent, const edm::EventSetup& c);

  /// End Lumi Block
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
 



 protected:
  
  /// Get the ME name
  std::string getMEName(RPCDetId & detId);
  
  
 private:
  
  std::string eventInfoPath_, prefixDir_;
  
  bool enableDetectorSegmentation_;

  DQMStore* dbe_;

  int nLumiSegs_;
  
  edm::ESHandle<RPCGeometry> muonGeom;
  //edm::ESHandle<DTTtrig> tTrigMap;

  std::map<RPCDetId,std::string>  meCollection;
};

#endif
