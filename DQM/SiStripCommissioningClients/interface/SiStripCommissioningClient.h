#ifndef DQM_SiStripCommissioningClients_SiStripCommissioningClient_H
#define DQM_SiStripCommissioningClients_SiStripCommissioningClient_H

#include "DQMServices/Components/interface/DQMBaseClient.h"
#include "DQMServices/Components/interface/UpdateObserver.h"
#include "DQMServices/Components/interface/Updater.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"

#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "xdata/include/xdata/String.h"
#include "boost/shared_ptr.hpp"
#include <string>
#include <vector>
#include <BSem.h>

class SiStripCommissioningWebClient;
class CommissioningHistograms;
class HistogramDisplayHandler;

class SiStripCommissioningClient : public DQMBaseClient, public dqm::UpdateObserver {
  
 public:


  // -------------------- Instantiation --------------------

  
  //@@ This line is necessary!
  XDAQ_INSTANTIATOR();
  
  SiStripCommissioningClient( xdaq::ApplicationStub* );
  virtual ~SiStripCommissioningClient();


  // -------------------- States and monitoring --------------------


  /** Initialization performed during "Configure". */
  void configure();

  /** Initialization performed during "Enable". */
  void newRun();

  /** "Tidy up" performed during "Halt". */
  void endRun();

  /** Method called during monitoring loops. */
  void onUpdate() const;

  
  // -------------------- Client "actions" --------------------
  
  /** */
  void subscribeAll( std::string match_pattern = "" );

  /** */
  void unsubscribeAll( std::string match_pattern = "" );

  /** */
  void removeAll( std::string match_pattern = "" );

  /** */
  void updateHistos();

  /** */
  void saveHistos( std::string filename );

  /** */
  void histoAnalysis( bool debug );
  
  /** */
  void createSummaryHisto( sistrip::Monitorable, 
			   sistrip::Presentation, 
			   std::string top_level_dir,
			   sistrip::Granularity );
  
  /** */
  virtual void uploadToConfigDb(); 
  

  // -------------------- Web-interface --------------------


  /** Answers all HTTP requests of the form ".../Request?RequestID=..." */
  void handleWebRequest( xgi::Input*, xgi::Output* ) throw ( xgi::exception::Exception );
  
  /** Outputs the page with the widgets (declared in DQMBaseClient) */
  void general( xgi::Input*, xgi::Output* ) throw ( xgi::exception::Exception );

  /** */
  void CBHistogramViewer( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception );


 protected:
 

  // ---------- "Actions", wrapped by SealCallback ----------


  /** */
  void subscribe( std::string match_pattern );

  /** */
  void unsubscribe( std::string match_pattern );

  /** */
  void remove( std::string match_pattern );

  /** */
  void update();

  /** */
  void save( std::string filename );

  
  // ---------- Management of client histograms ----------  
  
  /** */
  virtual void createHistograms( const sistrip::RunType& task ) const;
  
  /** */
  void handleException( const std::string& method_name,
			const std::string& message = "" );
  
  /** */
  bool parameterSetToString( const std::string& config_file,
			     std::string& parameter_set );


  // -------------------- Protected member data --------------------
  

  /** Web interface class. */
  SiStripCommissioningWebClient* web_;
  
  /** Action "executor" */
  mutable CommissioningHistograms* histos_;
  
  mutable sistrip::RunType runType_;
  
  mutable bool first_;

  BSem* fCallBack;

  HistogramDisplayHandler* hdis_;

  xdata::String cfgFile_;
  edm::AssertHandler* handler_;
  boost::shared_ptr<edm::Presence> presence_;
  edm::ServiceToken token_;
  
};

#endif // DQM_SiStripCommissioningClients_SiStripCommissioningClient_H

