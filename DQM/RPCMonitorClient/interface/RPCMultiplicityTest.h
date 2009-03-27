#ifndef RPCMultiplicityTest_H
#define RPCMultiplicityTest_H

<<<<<<< RPCMultiplicityTest.h
#include "DQM/RPCMonitorClient/interface/RPCClient.h"
#include "DQMServices/Core/interface/DQMStore.h"
=======
/** \class  RPCMultiplicityTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2009/02/24 13:15:32 $
 *  $Revision: 1.2 $
 *  \author   
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

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
>>>>>>> 1.2

#include <map>
#include <memory>
#include <string>
#include <vector>
<<<<<<< RPCMultiplicityTest.h
=======

class  RPCMultiplicityTest:public edm::EDAnalyzer{
>>>>>>> 1.2

<<<<<<< RPCMultiplicityTest.h
class  RPCMultiplicityTest:public RPCClient{

=======
>>>>>>> 1.2
public:

  /// Constructor
  RPCMultiplicityTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCMultiplicityTest();

  /// BeginJob
<<<<<<< RPCMultiplicityTest.h
  void beginJob(DQMStore * );
=======
  void beginJob(const edm::EventSetup& );
>>>>>>> 1.2

  //Begin Run
<<<<<<< RPCMultiplicityTest.h
   void beginRun(const edm::Run& , const edm::EventSetup& , std::vector<MonitorElement *> , std::vector<RPCDetId>);
=======
   void beginRun(const edm::Run& , const edm::EventSetup& );
>>>>>>> 1.2
  
  
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
<<<<<<< RPCMultiplicityTest.h
  void fillGlobalME(RPCDetId & detId, MonitorElement * myMe);
=======
  void fillGlobalME(RPCDetId & detId, MonitorElement * myMe,  edm::EventSetup const& iSetup);
>>>>>>> 1.2

<<<<<<< RPCMultiplicityTest.h
 private:
  int prescaleFactor_;
  std::string globalFolder_;
  int numberOfDisks_;

  std::vector<MonitorElement *>  myNumDigiMe_;
  std::vector<RPCDetId>   myDetIds_;

  MonitorElement * MULTWheel[5];
  MonitorElement * MULTDWheel[5];
  MonitorElement * MULTDisk[10]; 
  MonitorElement * MULTDDisk[10];

  DQMStore* dbe_;
  // std:: map<int, std::map< int ,  std::pair<float,float> > >  barrelMap_, endcapMap_;
=======
>>>>>>> 1.2
  
<<<<<<< RPCMultiplicityTest.h
=======
 private:
  int prescaleFactor_;
  std::string globalFolder_;
 std::string prefixDir_;
  std::vector<MonitorElement *>  myNumDigiMe_;
  std::vector<RPCDetId>   myDetIds_;
  std::vector<std::string>    myRollNames_;
  DQMStore* dbe_;
  std:: map<int, std::map< int ,  std::pair<float,float> > >  barrelMap_, endcapMap_;
  
>>>>>>> 1.2
};
#endif
