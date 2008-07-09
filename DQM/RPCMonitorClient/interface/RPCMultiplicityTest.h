#ifndef RPCMultiplicityTest_H
#define RPCMultiplicityTest_H


/** \class RPCDeadChannelTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/03/08 19:15:10 $
 *  $Revision: 1.2 $
 *  \author 
 *   
 */

//#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/Run.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
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


class RPCMultiplicityTest: public RPCClient {
public:

  /// Constructor
  RPCMultiplicityTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCMultiplicityTest();

  /// BeginJob
  void beginJob( DQMStore * dbe);

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

  //configurable in cfg file
  bool referenceOldChannels;
  bool getQualityTestsFromFile;
  std::ifstream referenceFile_;
  
  std::ofstream myfile; 

  DQMStore* dbe_;
  //  QTestHandle *qtHandler;
  // DQMOldReceiver * mui_;

  edm::ParameterSet parameters;
  edm::ESHandle<RPCGeometry> muonGeom;
  //edm::ESHandle<DTTtrig> tTrigMap;

  // std::map< std::string , MonitorElement* > OccupancyHistos;
  std::map<RPCDetId,MonitorElement*>  meCollection;
};

#endif
