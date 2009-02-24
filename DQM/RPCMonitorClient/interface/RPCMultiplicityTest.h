#ifndef RPCMultiplicityTest_H
#define RPCMultiplicityTest_H

/** \class  RPCMultiplicityTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/12/15 16:28:30 $
 *  $Revision: 1.6 $
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

#include <map>
#include <memory>
#include <string>
#include <vector>

class  RPCMultiplicityTest:public edm::EDAnalyzer{

public:

  /// Constructor
  RPCMultiplicityTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~RPCMultiplicityTest();

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
  void fillGlobalME(RPCDetId & detId, MonitorElement * myMe,  edm::EventSetup const& iSetup);

  
 private:
  int prescaleFactor_;
  std::string globalFolder_;
 std::string prefixDir_;
  std::vector<MonitorElement *>  myNumDigiMe_;
  std::vector<RPCDetId>   myDetIds_;
  std::vector<std::string>    myRollNames_;
  DQMStore* dbe_;
  std:: map<int, std::map< int ,  std::pair<float,float> > >  barrelMap_, endcapMap_;
  
};
#endif
