
#ifndef DQM_SiStripCommissioningDbClients_SiStripCommissioningOfflineDbClient_H
#define DQM_SiStripCommissioningDbClients_SiStripCommissioningOfflineDbClient_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningOfflineClient.h"
#include "DQM/SiStripCommissioningClients/interface/SiStripTFile.h"

/**
   @class SiStripCommissioningOfflineDbClient 
   @author R.Bainbridge, M.Wingham
   
   @brief Class which reads a root file containing "commissioning
   histograms", analyzes the histograms to extract "monitorables",
   creates summary histograms, and uploads to DB.
*/
class SiStripCommissioningOfflineDbClient : public SiStripCommissioningOfflineClient {

 public:
  
  SiStripCommissioningOfflineDbClient( const edm::ParameterSet& );

  ~SiStripCommissioningOfflineDbClient() override;
  
 protected:
  
  void createHistos( const edm::ParameterSet&, const edm::EventSetup& ) override;
  
  void uploadToConfigDb() override;
  
 private:
  
  bool uploadToDb_;
  
  bool uploadAnal_;
  
  bool uploadConf_;
  

};

#endif // DQM_SiStripCommissioningDbClients_SiStripCommissioningOfflineDbClient_H

