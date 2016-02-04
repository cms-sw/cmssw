
/*!
  \file RPCClient.h
   \author A. Cimmino
  \version $Revision: 1.6 $
  \date $Date: 2011/03/02 16:59:58 $
*/


#ifndef RPCClient_H
#define RPCClient_H



#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <DataFormats/MuonDetId/interface/RPCDetId.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Run.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>
//#include "FWCore/ServiceRegistry/interface/Service.h"

#include <map>
#include <vector>
#include <string>


class RPCClient {

 public:
  
  //RPCClient(const edm::ParameterSet& ps) {}
  virtual ~RPCClient(void) {}

  virtual void beginLuminosityBlock(edm::LuminosityBlock const& , edm::EventSetup const& )=0 ;

  virtual void clientOperation(edm::EventSetup const& c)=0;

  virtual void getMonitorElements(std::vector<MonitorElement *> &, std::vector<RPCDetId> &)= 0;

  virtual void endLuminosityBlock(edm::LuminosityBlock const& , edm::EventSetup const& )=0;
  
  virtual void analyze(const edm::Event & , const edm::EventSetup& )      = 0;		       
		       
  virtual void beginJob(DQMStore * , std::string )     = 0;
  
  virtual void endJob(void)       = 0;
  
  virtual void beginRun(const edm::Run& , const edm::EventSetup& )     = 0;
  
  virtual void endRun(const edm::Run& , const edm::EventSetup& )       = 0;  


  //  private:
  //parameters used to configure quality tests



};

#endif 
