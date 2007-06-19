// Last commit: $Id: SiStripCommissioningOfflineDbClient.h,v 1.1 2007/06/04 13:32:11 bainbrid Exp $

#ifndef DQM_SiStripCommissioningDbClients_SiStripCommissioningOfflineDbClient_H
#define DQM_SiStripCommissioningDbClients_SiStripCommissioningOfflineDbClient_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningOfflineClient.h"
#include "DQM/SiStripCommissioningClients/interface/SiStripTFile.h"
#include "DQM/SiStripCommissioningClients/interface/ConfigParser.h"
#include <string>
#include <vector>
#include <map>

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
  virtual ~SiStripCommissioningOfflineDbClient();
  
 protected:
  
  void createCommissioningHistograms();
  void testUploadToDb();
  void uploadToDb();

 private:

  bool uploadToDb_;

  bool test_;

  bool uploadPllSettings_;

  bool uploadFedSettings_;
  
};

#endif // DQM_SiStripCommissioningDbClients_SiStripCommissioningOfflineDbClient_H

