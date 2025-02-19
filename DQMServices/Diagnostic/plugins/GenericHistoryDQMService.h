#ifndef DQM_GenericDQMService_H
#define DQM_GenericDQMService_H

#include "DQMServices/Diagnostic/interface/DQMHistoryServiceBase.h" 

/**
  @author D. Giordano
  @EDAnalyzer to read DQM root file & insert summary informations to DB 
*/

class GenericHistoryDQMService : public DQMHistoryServiceBase {
 public:

  explicit GenericHistoryDQMService(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~GenericHistoryDQMService();
  
 private:
  //Methods to be specified by each subdet
  uint32_t returnDetComponent(const MonitorElement* ME);
  bool setDBLabelsForUser  (std::string& keyName, std::vector<std::string>& userDBContent, std::string& quantity );
  bool setDBValuesForUser(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values, std::string& quantity );
   
  edm::ParameterSet iConfig_;
};

#endif 
