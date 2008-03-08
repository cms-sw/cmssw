#ifndef RPCDeadChannelTest_H
#define RPCDeadChannelTest_H


/** \class RPCDeadChannelTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/03/07 18:16:55 $
 *  $Revision: 1.1 $
 *  \author 
 *   
 */

//#include "DataFormats/Common/interface/Handle.h"

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


class RPCDeadChannelTest: public edm::EDAnalyzer {
public:

  /// Constructor
  RPCDeadChannelTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCDeadChannelTest();

protected:

  /// BeginJob
  void beginJob( const edm::EventSetup& c);
		
  /// Analyze
  void analyze(const edm::Event& iEvent, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// book the new ME
 //  void bookHistos(/*const DTLayerId & ch, int firstWire, int lastWire*/);

  /// Get the ME name
  MonitorElement*getMEs(RPCDetId & detId);

    void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
   void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

 

private:

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run; 
  int lumiBlock;
  char dateTime[32];

  bool getQualityTestsFromFile;
  bool referenceOldDeadChannels;

  std::ofstream myfile;
  std::ifstream referenceFile_;

  DQMStore* dbe_;
  QTestHandle *qtHandler;
  DQMOldReceiver * mui_;

  edm::ParameterSet parameters;
  edm::ESHandle<RPCGeometry> muonGeom;
  //edm::ESHandle<DTTtrig> tTrigMap;

  // std::map< std::string , MonitorElement* > OccupancyHistos;
  std::map<RPCDetId,MonitorElement*>  meCollection;
};

#endif
