#ifndef RPCEventSummary_H
#define RPCEventSummary_H


/** \class RPCEventSummary
 * *
 *  DQM Event Summary module for RPCs
 *
 *  $Date: 2008/09/28 15:37:48 $
 *  $Revision: 1.6 $
 *  \author Anna Cimmino
 *   
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
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
  void fillReportSummary(const std::map<int,std::map<int,std::pair<float,float> > >& sumMap, int region);
  
 private:
  
  std::string eventInfoPath_, prefixDir_;

  bool tier0_;  
  bool enableReportSummary_;
  int prescaleFactor_;
  bool verbose_;
  DQMStore* dbe_;
  int minHitsInRoll_;
  int nLumiSegs_;
  
  float allRolls_;
  float allGood_;

  edm::ESHandle<RPCGeometry> muonGeom;
  //edm::ESHandle<DTTtrig> tTrigMap;


  std::vector<std::string> segmentNames; 

  std::map<RPCDetId,std::string>  meCollection;
};

#endif
