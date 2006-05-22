#ifndef DQM_SiStripCommissioningClients_SiStripCommissioningClient_H
#define DQM_SiStripCommissioningClients_SiStripCommissioningClient_H

#include "DQMServices/Components/interface/DQMBaseClient.h"
#include "DQMServices/Components/interface/UpdateObserver.h"
#include "DQMServices/Components/interface/Updater.h"
#include <string>
#include <vector>

class SiStripCommissioningWebClient;
class CommissioningHistograms;

class SiStripCommissioningClient : public DQMBaseClient, public dqm::UpdateObserver {
  
 public:
  
  // This line is necessary
  XDAQ_INSTANTIATOR();
  
  /** Constructor. */
  SiStripCommissioningClient( xdaq::ApplicationStub* );
  /** Destructor. */
  ~SiStripCommissioningClient();
  
  /** Called whenever the client enters the "Configured" state. */ 
  void configure();
  /** Called whenever the client enters the "Enabled" state. */ 
  void newRun();
  /** Called whenever the client enters the "Halted" state. */ 
  void endRun();
  /** Called by the "Updater" whenever there is an update. */
  void onUpdate() const;

  /** Outputs the page with the widgets (declared in DQMBaseClient) */
  void general( xgi::Input*, xgi::Output* ) throw ( xgi::exception::Exception );
  
  /** Answers all HTTP requests of the form ".../Request?RequestID=..." */
  void handleWebRequest( xgi::Input*, xgi::Output* );
  
 private: // ----- methods -----
  
  /** */
  CommissioningHistograms* createHistograms( std::vector<std::string>& added_contents ) const;

  /** */
  void createCollateMonitorElements( std::vector<std::string>& added_contents ) const;
  
 private: // ----- member data -----
  
  /** Web-based commissioning client. */
  SiStripCommissioningWebClient* web_;

  /** */
  mutable CommissioningHistograms* histo_;

};

// This line is necessary
XDAQ_INSTANTIATOR_IMPL(SiStripCommissioningClient)
     
#endif // DQM_SiStripCommissioningClients_SiStripCommissioningClient_H
     
