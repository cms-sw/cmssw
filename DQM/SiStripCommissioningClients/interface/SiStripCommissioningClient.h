#ifndef DQM_SiStripCommissioningClients_SiStripCommissioningClient_H
#define DQM_SiStripCommissioningClients_SiStripCommissioningClient_H

#include "DQMServices/Components/interface/DQMBaseClient.h"
#include "DQMServices/Components/interface/UpdateObserver.h"
#include "DQMServices/Components/interface/Updater.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <string>
#include <vector>
#include <BSem.h>

class SiStripCommissioningWebClient;
class CommissioningHistograms;
class HistogramDisplayHandler;

class SiStripCommissioningClient : public DQMBaseClient, public dqm::UpdateObserver {
  
 public:
  
  //@@ This line is necessary!
  XDAQ_INSTANTIATOR();
  
  SiStripCommissioningClient( xdaq::ApplicationStub* );
  virtual ~SiStripCommissioningClient();

  // ---------- States and monitoring ----------

  void configure();
  void newRun();
  void endRun();
  void onUpdate() const;
  
  // ---------- Web-related ----------

  /** Answers all HTTP requests of the form ".../Request?RequestID=..." */
  void handleWebRequest( xgi::Input*, xgi::Output* );
  
  /** Outputs the page with the widgets (declared in DQMBaseClient) */
  void general( xgi::Input*, xgi::Output* ) throw ( xgi::exception::Exception );

  /** */
  void CBHistogramViewer( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception );
  
  // ---------- "Actions" ----------
  
  /** */
  void subscribeAll( std::string match_pattern = "" );
  /** */
  void updateHistos();
  /** */
  void unsubscribeAll( std::string match_pattern = "" );
  /** */
  void removeAll( std::string match_pattern = "" );
  /** */
  void saveHistos( std::string filename );
  /** */
  void histoAnalysis( bool debug );
  /** */
  void createSummaryHisto( sistrip::SummaryHisto, 
			   sistrip::SummaryType, 
			   std::string top_level_dir,
			   sistrip::Granularity );
  /** */
  virtual void uploadToConfigDb(); 

 protected:
 
  /** */
  void subscribe( std::string match_pattern );
  /** */
  void update();
  /** */
  void unsubscribe( std::string match_pattern );
  /** */
  void remove( std::string match_pattern );
  /** */
  void save( std::string filename );

  /** */
  sistrip::Task extractTask( const std::vector<std::string>& added_contents ) const;
  
  /** */
  virtual void createHistograms( const sistrip::Task& task ) const;
  
  /** Web-based commissioning client. */
  SiStripCommissioningWebClient* web_;
  
  /** Object holding commissioning histograms (mutable as used in
      const onUpdate() method). */
  mutable CommissioningHistograms* histos_;

  mutable sistrip::Task task_;
  
  mutable bool first_;

  BSem* fCallBack;

  HistogramDisplayHandler* hdis_;

};

#endif // DQM_SiStripCommissioningClients_SiStripCommissioningClient_H

