#ifndef RPCDeadChannelTest_H
#define RPCDeadChannelTest_H


<<<<<<< RPCDeadChannelTest.h
#include "DQM/RPCMonitorClient/interface/RPCClient.h"
=======
/** \class RPCDeadChannelTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2009/02/24 13:15:32 $
 *  $Revision: 1.7 $
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
>>>>>>> 1.7

#include "DQMServices/Core/interface/DQMStore.h"

<<<<<<< RPCDeadChannelTest.h
=======
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
>>>>>>> 1.7

<<<<<<< RPCDeadChannelTest.h
class RPCDeadChannelTest:public RPCClient{
=======
#include <map>
#include <memory>
#include <string>
#include <vector>

class RPCDeadChannelTest:public edm::EDAnalyzer{
>>>>>>> 1.7

public:

  /// Constructor
  RPCDeadChannelTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCDeadChannelTest();

  /// BeginJob
  void beginJob(DQMStore *);

  //Begin Run
   void beginRun(const edm::Run& , const edm::EventSetup& ,std::vector<MonitorElement *> , std::vector<RPCDetId>);
  
  
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
<<<<<<< RPCDeadChannelTest.h
  void CalculateDeadChannelPercentage(RPCDetId & , MonitorElement *  , edm::EventSetup const& );
   
 private:
=======
  void fillDeadChannelHisto(const std::map<int,std::map<int,std::pair<float,float> > > & sumMap, int region);
  void CalculateDeadChannelPercentage(RPCDetId & detId, MonitorElement * myMe,  edm::EventSetup const& iSetup);

  
 private:
>>>>>>> 1.7
  int prescaleFactor_;
  std::string globalFolder_;
  std::vector<MonitorElement *>  myOccupancyMe_;
  std::vector<RPCDetId>   myDetIds_;
 
  DQMStore* dbe_;
<<<<<<< RPCDeadChannelTest.h
 
  int numberOfDisks_;

  MonitorElement * DEADWheel[5];
  MonitorElement * DEADDisk[10]; 


=======
  std:: map<int, std::map< int ,  std::pair<float,float> > >  barrelMap_, endcapMap_;
>>>>>>> 1.7
  
};

#endif
