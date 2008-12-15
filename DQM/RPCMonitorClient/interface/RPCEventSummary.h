#ifndef RPCEventSummary_H
#define RPCEventSummary_H


/** \class RPCEventSummary
 * *
 *  DQM Event Summary module for RPCs
 *
 *  $Date: 2008/11/12 12:26:53 $
 *  $Revision: 1.7 $
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

  std::vector<MonitorElement *>  myOccupancyMe_;
  std::vector<RPCDetId>   myDetIds_;
  std::vector<std::string>    myRollNames_;
  
  std::vector<std::string> segmentNames; 


};

#endif
