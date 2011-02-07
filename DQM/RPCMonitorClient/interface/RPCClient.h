
/*!
  \file RPCClient.h
   \author A. Cimmino
  \version $Revision: 1.4 $
  \date $Date: 2009/10/29 22:50:23 $
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

  virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context)=0 ;

  virtual void clientOperation(edm::EventSetup const& c)=0;

  virtual void bookHisto(std::vector<MonitorElement *> , std::vector<RPCDetId>)= 0;

  virtual void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c)=0;
  
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& c)      = 0;		       
		       
  virtual void beginJob(DQMStore* dbe)     = 0;
  
  virtual void endJob(void)       = 0;
  
  virtual void beginRun(const edm::Run& r, const edm::EventSetup& c)     = 0;
  
  virtual void endRun(const edm::Run& r, const edm::EventSetup& c)       = 0;  


  //  private:
  //parameters used to configure quality tests



};

#endif 
