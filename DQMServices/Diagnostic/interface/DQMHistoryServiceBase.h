#ifndef DQMServices_Diagnostic_DQMHistoryServiceBase_H
#define DQMServices_Diagnostic_DQMHistoryServiceBase_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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

  virtual void setSeparator (std::string const&);
  
  protected:

  virtual void createSummary();
  virtual void openRequestedFile(); 
  virtual void scanTreeAndFillSummary(const std::vector<MonitorElement*>& MEs, HDQMSummary* summary,std::string& histoName, std::vector<std::string>& Quantities);
  virtual uint32_t getRunNumber() const;
  virtual uint32_t returnDetComponent(const MonitorElement* MEs){return 999999;}
  
  virtual bool setDBLabelsForLandau(std::string& keyName, std::vector<std::string>& userDBContent);
  virtual bool setDBLabelsForGauss (std::string& keyName, std::vector<std::string>& userDBContent);
  virtual bool setDBLabelsForStat  (std::string& keyName, std::vector<std::string>& userDBContent);
  virtual bool setDBLabelsForUser  (std::string& keyName, std::vector<std::string>& userDBContent, std::string& quantity ){return setDBLabelsForUser(keyName, userDBContent);}
  virtual bool setDBLabelsForUser  (std::string& keyName, std::vector<std::string>& userDBContent){return false;}

  virtual bool setDBValuesForLandau(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values);
  virtual bool setDBValuesForGauss(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values );
  virtual bool setDBValuesForStat(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values  );
  virtual bool setDBValuesForUser(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values, std::string& quantity ){return setDBValuesForUser(iterMes,values);}
  virtual bool setDBValuesForUser(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values){return false;}
  
  DQMStore* dqmStore_;
  edm::ParameterSet iConfig_;
  HDQMSummary* obj_;
  HDQMfitUtilities *fitME;
  std::string fSep;
};

#endif //DQMServices_Diagnostic_DQMHistoryServiceBase_H
