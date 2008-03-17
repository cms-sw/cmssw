
/*!
  \file RPCClient.h
   \author A. Cimmino
  \version $Revision: 1.00 $
  \date $Date: 2008/03/11 12:27:48 $
*/


#ifndef RPCClient_H
#define RPCClient_H

#include <string>

class DQMStore;

class RPCClient {

 public:
  
  //   RPCClient(const edm::ParameterSet& ps) {}
  virtual ~RPCClient(void) {}

  virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context)=0 ;
  
  
  virtual void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c)=0;
  
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& c)      = 0;		       
		       
  virtual void beginJob(DQMStore* dbe)     = 0;
  
  virtual void endJob(void)       = 0;
  
  virtual void beginRun(const edm::Run& r, const edm::EventSetup& c)     = 0;
  
  virtual void endRun(const edm::Run& r, const edm::EventSetup& c)       = 0;  


  // private:
  //parameters used to configure quality tests



};

#endif 
