#ifndef RPCDeadChannelTest_H
#define RPCDeadChannelTest_H


/** \class RPCDeadChannelTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/03/16 18:52:20 $
 *  $Revision: 1.3 $
 *  \author 
 *   
 */

//#include "DataFormats/Common/interface/Handle.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Run.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/MonitorElement.h"
//#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
//#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DQM/RPCMonitorClient/interface/RPCClient.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class QTestHandle;
class DQMOldReceiver;
class DQMStore;
class RPCDetId;


class RPCDeadChannelTest: public RPCClient {
public:

  /// Constructor
  RPCDeadChannelTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCDeadChannelTest();

  /// BeginJob
  void beginJob(DQMStore * dbe);

  //Begin Run
   void beginRun(const edm::Run& r, const edm::EventSetup& c);
  
  
  /// Begin Lumi block 
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Analyze  
  void analyze(const edm::Event& iEvent, const edm::EventSetup& c);

  /// End Lumi Block
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
 
  //End Run
  void endRun(const edm::Run& r, const edm::EventSetup& c); 		
  
  /// Endjob
  void endJob();


 protected:

   /// Get the ME name
   MonitorElement*getMEs(RPCDetId & detId);
 

private:

  int nevents;
  unsigned int nLumiSegs;
   int prescaleFactor;
  int run; 
  int lumiBlock;
  char dateTime[32];

     bool referenceOldChannels;
     bool getQualityTestsFromFile;

  std::ofstream myfile;
  std::ifstream referenceFile_;


  DQMStore* dbe_;

  edm::ParameterSet parameters;
  edm::ESHandle<RPCGeometry> muonGeom;
  //edm::ESHandle<DTTtrig> tTrigMap;

  std::map<RPCDetId,MonitorElement*>  meCollection;
};

#endif
