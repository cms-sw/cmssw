#ifndef DQM_SiStripHistoricInfoClient_SiStripNewHistoricDQMService_H
#define DQM_SiStripHistoricInfoClient_SiStripNewHistoricDQMService_H

#include "DQMServices/Diagnostic/interface/DQMHistoryServiceBase.h" 

/**
  @author D. Giordano, A.-C. Le Bihan
  @EDAnalyzer to read DQM root file & insert summary informations to DB 
*/

class SiStripNewHistoricDQMService : public DQMHistoryServiceBase {
 public:

  explicit SiStripNewHistoricDQMService(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripNewHistoricDQMService();
  
 //private:
   uint32_t returnDetComponent(std::string& histoName);
   edm::ParameterSet iConfig_;
};

#endif //DQM_SiStripHistoricInfoClient_SiStripNewHistoricDQMService_H
