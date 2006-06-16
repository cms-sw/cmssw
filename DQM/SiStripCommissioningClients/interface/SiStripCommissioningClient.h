#ifndef DQM_SiStripCommissioningClients_SiStripCommissioningClient_H
#define DQM_SiStripCommissioningClients_SiStripCommissioningClient_H

#include "DQMServices/Components/interface/DQMBaseClient.h"
#include "DQMServices/Components/interface/UpdateObserver.h"
#include "DQMServices/Components/interface/Updater.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include <string>
#include <vector>

class SiStripCommissioningWebClient;
class CommissioningHistograms;

class SiStripCommissioningClient : public DQMBaseClient, public dqm::UpdateObserver {
  
 public:
  
  // This line is necessary!
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
  
  /** Friend method to allow web interface access to histos. */
  friend CommissioningHistograms* histos( const SiStripCommissioningClient& );

  /** */
  inline void lock() const;
  /** */
  inline void unlock() const;
  
 private:

  /** Extracts "commissioning task" string and creates a new
      CommissioningHistogram object based on the task. */
  void createCommissioningHistos( const std::vector<std::string>& added_contents ) const;
  
  /** Web-based commissioning client. */
  SiStripCommissioningWebClient* web_;
  /** */
  mutable CommissioningHistograms* histos_;
  
};

void SiStripCommissioningClient::lock() const { if ( mui_ ) { mui_->getBEInterface()->lock(); } }
void SiStripCommissioningClient::unlock() const { if ( mui_ ) { mui_->getBEInterface()->unlock(); } }

#endif // DQM_SiStripCommissioningClients_SiStripCommissioningClient_H

