#ifndef RPCOccupancyChipTest_H
#define RPCOccupancyChipTest_H


/** \class RPCOccupancyTest
 * *
 *  DQM Event Summary module for RPCs
 *
 *  $Date: 2008/12/04 20:17:20 $
 *  $Revision: 1.1 $
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


class RPCOccupancyChipTest:public edm::EDAnalyzer {
public:

  /// Constructor
  RPCOccupancyChipTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCOccupancyChipTest();

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
 

 private:
  
  std::string prefixDir_;

  int prescaleFactor_;
  bool verbose_;
  DQMStore* dbe_;
  int nLumiSegs_;


  std::vector<MonitorElement *>  myOccupancyMe_;
  std::vector<RPCDetId>   myDetIds_;

  std::vector<std::string> segmentNames; 

  std::map<RPCDetId,std::string>  meCollection;
};

#endif
