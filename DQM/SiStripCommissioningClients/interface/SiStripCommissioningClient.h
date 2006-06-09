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

  /** "Configured" state. */ 
  void configure();
  /** "Enabled" state. */ 
  void newRun();
  /** "Halted" state. */ 
  void endRun();
  /** Called by the "Updater". */
  void onUpdate() const;
  
  /** Answers all HTTP requests of the form ".../Request?RequestID=..." */
  void handleWebRequest( xgi::Input*, xgi::Output* );
  
  /** Outputs the page with the widgets (declared in DQMBaseClient) */
  void general( xgi::Input*, xgi::Output* ) throw ( xgi::exception::Exception );
  
  /** Friend method to allow access to CommissioningHistograms object. */
  inline friend CommissioningHistograms* histo( const SiStripCommissioningClient& );
  
 private:

  /** Extracts "commissioning task" string and creates a new
      CommissioningHistogram object based on the task. */
  void createCommissioningHistos( const std::vector<std::string>& added_contents ) const;
  
  /** Web-based commissioning client. */
  SiStripCommissioningWebClient* web_;
  /** */
  mutable CommissioningHistograms* histo_;
  
};

// ---------- inline methods ----------

CommissioningHistograms* histo( const SiStripCommissioningClient& client ) {
  return client.histo_;
}

#endif // DQM_SiStripCommissioningClients_SiStripCommissioningClient_H

