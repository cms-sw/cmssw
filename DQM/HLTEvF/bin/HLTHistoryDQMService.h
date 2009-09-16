#ifndef DQM_HLTEVF_HLTTHistoryDQMService_H
#define DQM_HLTEVF_HLTTHistoryDQMService_H

#include "DQMServices/Diagnostic/interface/DQMHistoryServiceBase.h" 

/**
  @author D. Giordano, A.-C. Le Bihan
  @EDAnalyzer to read DQM root file & insert summary informations to DB 
*/

class HLTHistoryDQMService : public DQMHistoryServiceBase {
 public:

  explicit HLTHistoryDQMService(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~HLTHistoryDQMService();
  
 private:
  //Methods to be specified by each subdet
  uint32_t returnDetComponent(const MonitorElement* ME);
  bool setDBLabelsForUser  (std::string& keyName, std::vector<std::string>& userDBContent);
  bool setDBValuesForUser(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values  );
   
   edm::ParameterSet iConfig_;
   double threshold_;
};

#endif //DQM_SiStripHistoricInfoClient_HLTHistoryDQMService_H
