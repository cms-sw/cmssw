#ifndef DQM_SiStripCommissioningClients_SiStripCommissioningClient_H
#define DQM_SiStripCommissioningClients_SiStripCommissioningClient_H

#include "DQMServices/Components/interface/DQMBaseClient.h"
#include "DQMServices/Components/interface/UpdateObserver.h"
#include "DQMServices/Components/interface/Updater.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <string>
#include <vector>

class SiStripCommissioningWebClient;
class CommissioningHistograms;

class SiStripCommissioningClient : public DQMBaseClient, 
				   public dqm::UpdateObserver {
  
 public:
  
  // This line is necessary!
  XDAQ_INSTANTIATOR();
  
  SiStripCommissioningClient( xdaq::ApplicationStub* );
  ~SiStripCommissioningClient();
  
  void configure();
  void newRun();
  void endRun();
  void onUpdate() const;
  
  /** */
  sistrip::Task extractTask( const std::vector<std::string>& added_contents ) const;

  /** */
  void createTaskHistograms( const sistrip::Task& task ) const;

  /** Friend method to allow web interface access to commissioning histos. */
  friend CommissioningHistograms* histos( const SiStripCommissioningClient& );
  
  /** Answers all HTTP requests of the form ".../Request?RequestID=..." */
  void handleWebRequest( xgi::Input*, xgi::Output* );
  
  /** Outputs the page with the widgets (declared in DQMBaseClient) */
  void general( xgi::Input*, xgi::Output* ) throw ( xgi::exception::Exception );
  
 private:
  
  /** Web-based commissioning client. */
  SiStripCommissioningWebClient* web_;

  /** Object holding commissioning histograms (mutable as used in
      const onUpdate() method). */
  mutable CommissioningHistograms* histos_;
  
};

#endif // DQM_SiStripCommissioningClients_SiStripCommissioningClient_H

