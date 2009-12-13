#ifndef RPCChamberQuality_H
#define RPCChamberQuality_H

/****************************************
*****************************************
**                                     **
**  Class RPCClusterSizeTest           **
**  DQM Event Summary module for RPCs  **
**                                     **
**  $Date: 2009/10/29 22:50:23 $       **
**  $Revision: 1.4 $                   **
**  David Lomidze                      **
**  INFN di Napoli                     **
**                                     **
*****************************************   
****************************************/                 
        

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


class RPCChamberQuality:public edm::EDAnalyzer {
public:

  /// Constructor
  RPCChamberQuality(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCChamberQuality();

  /// BeginJob
  void beginJob(const edm::EventSetup& iSetup);

  //Begin Run
   void beginRun(const edm::Run& r, const edm::EventSetup& c);
   //End Run
   void endRun(const edm::Run& r, const edm::EventSetup& c);
  
  /// Begin Lumi block 
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Analyze  
  void analyze(const edm::Event& iEvent, const edm::EventSetup& c);

  /// End Lumi Block
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
 
  ///end Job
  //  void endJob(void);
  
 private:
  
  std::string prefixDir_;
  
  int prescaleFactor_;
  //  bool verbose_;
  bool init_;
  DQMStore* dbe_;
  int nLumiSegs_;
  int minEvents;
  int numLumBlock_;
  int nLumBlock_;

  edm::ESHandle<RPCGeometry> muonGeom;
  
  std::vector<std::string> segmentNames; 
  
  std::map<RPCDetId,std::string>  meCollection;

  //bool saveRootFile;
  //  std::string RootFileName;
  
  

};

#endif
