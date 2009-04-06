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
  ~SiStripHistoryDQMService();
  
 //private:
   uint32_t returnDetComponent(std::string& histoName);
   edm::ParameterSet iConfig_;
};

#endif //DQM_SiStripHistoricInfoClient_SiStripHistoryDQMService_H
