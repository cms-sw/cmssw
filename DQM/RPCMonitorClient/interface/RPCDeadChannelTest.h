#ifndef RPCDeadChannelTest_H
#define RPCDeadChannelTest_H


/** \class RPCDeadChannelTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/04/25 14:24:56 $
 *  $Revision: 1.4 $
 *  \author 
 *   
 */

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
#include "DQMServices/Core/interface/DQMStore.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include <memory>
#include <string>
#include <map>

class RPCDetId;


class RPCDeadChannelTest:public edm::EDAnalyzer{

public:

  /// Constructor
  RPCDeadChannelTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCDeadChannelTest();

  /// BeginJob
  void beginJob(const edm::EventSetup& );

  //Begin Run
   void beginRun(const edm::Run& , const edm::EventSetup& );
  
  
  /// Begin Lumi block 
  void beginLuminosityBlock(edm::LuminosityBlock const& , edm::EventSetup const& ) ;

  /// Analyze  
  void analyze(const edm::Event& , const edm::EventSetup& );

  /// End Lumi Block
  void endLuminosityBlock(edm::LuminosityBlock const& , edm::EventSetup const& );
 
  //End Run
  void endRun(const edm::Run& , const edm::EventSetup& ); 		
  
  /// Endjob
  void endJob();

 protected:
  void fillDeadChannelHisto(const std::map<int,std::map<int,std::pair<float,float> > > & sumMap, int region);
  
private:

  int prescaleFactor_;
  std::string globalFolder_,prefixDir_;


  DQMStore* dbe_;

  edm::ParameterSet parameters;
  edm::ESHandle<RPCGeometry> muonGeom;
  std::map<RPCDetId,MonitorElement*>  meCollection;

};

#endif
