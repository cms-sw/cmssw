#ifndef DQMServices_Diagnostic_DQMHistoryServiceBase_H
#define DQMServices_Diagnostic_DQMHistoryServiceBase_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/DQMObjects/interface/HDQMSummary.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <string>
#include <memory>
/**
  @author D. Giordano, A.-C. Le Bihan
  @EDAnalyzer to read DQM root file & insert summary informations to DB 
*/

class HDQMfitUtilities;

class DQMHistoryServiceBase {
 public:

  DQMHistoryServiceBase(const edm::ParameterSet&,const edm::ActivityRegistry&);
  virtual ~DQMHistoryServiceBase();
 
  virtual void getMetaDataString(std::stringstream& ss){ss << "Run " << getRunNumber();};

  virtual bool checkForCompatibility(std::string ss);
 
  virtual void getObj(HDQMSummary* & obj){createSummary(); obj=obj_;}

  virtual void initialize();
  
  protected:

  virtual void createSummary();
  virtual void openRequestedFile(); 
  virtual void scanTreeAndFillSummary(const std::vector<MonitorElement*>& MEs, HDQMSummary* summary,std::string& histoName, std::vector<std::string>& Quantities);
  virtual uint32_t getRunNumber() const;
  virtual uint32_t returnDetComponent(std::string& str){return 999999;}

  
  DQMStore* dqmStore_;
  edm::ParameterSet iConfig_;
  HDQMSummary* obj_;
  HDQMfitUtilities *fitME;
};

#endif //DQMServices_Diagnostic_DQMHistoryServiceBase_H
