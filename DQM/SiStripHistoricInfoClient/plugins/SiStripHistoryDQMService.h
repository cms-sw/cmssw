#ifndef DQM_SiStripHistoricInfoClient_SiStripHistoryDQMService_H
#define DQM_SiStripHistoricInfoClient_SiStripHistoryDQMService_H

#include "DQMServices/Diagnostic/interface/DQMHistoryServiceBase.h" 

/**
  @author D. Giordano, A.-C. Le Bihan
  @EDAnalyzer to read DQM root file & insert summary informations to DB 
*/

class SiStripHistoryDQMService : public DQMHistoryServiceBase {
 public:

  explicit SiStripHistoryDQMService(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripHistoryDQMService() override;
  
 private:
  //Methods to be specified by each subdet
  uint32_t returnDetComponent(const MonitorElement* ME) override;
  //bool setDBLabelsForUser  (std::string& keyName, std::vector<std::string>& userDBContent);
  //bool setDBValuesForUser(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values  );
  bool setDBLabelsForUser  (std::string& keyName, std::vector<std::string>& userDBContent, std::string& quantity) override;
  bool setDBValuesForUser(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values, std::string& quantity) override;
   
   edm::ParameterSet iConfig_;
};

#endif //DQM_SiStripHistoricInfoClient_SiStripHistoryDQMService_H
