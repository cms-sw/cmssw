// Last commit: $Id: SiStripCommissioningOfflineDbClient.h,v 1.1 2007/04/04 07:21:08 bainbrid Exp $

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
  
};

#endif // DQM_SiStripCommissioningDbClients_SiStripCommissioningOfflineDbClient_H

