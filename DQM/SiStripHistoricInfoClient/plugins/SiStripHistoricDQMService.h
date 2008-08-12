#ifndef DQM_SiStripHistoricInfoClient_SiStripHistoricDQMService_H
#define DQM_SiStripHistoricInfoClient_SiStripHistoricDQMService_H

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include <string>
/**
  @class ReadFromFile
  @author D. Giordano, A.-C. Le Bihan
  @EDAnalyzer to read DQM root file & insert summary informations to DB 
*/


class SiStripHistoricDQMService : public SiStripCondObjBuilderBase<SiStripSummary> {
 public:

  explicit SiStripHistoricDQMService(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripHistoricDQMService();
  
  void getMetaDataString(std::stringstream& ss){ss << "Run " << getRunNumber();};
  
  void getObj(SiStripSummary* & obj){createSummary(); obj=obj_;}

  void initialize();
  
 private:
  
  void createSummary();
  void openRequestedFile(); 
  void scanTreeAndFillSummary(std::string top_dir,SiStripSummary* summary,std::string& histoName, std::vector<std::string>& Quantities);
  uint32_t getRunNumber() const;
  uint32_t returnDetComponent(std::string& histoName);
  
  DQMStore* dqmStore_;
  
  std::string FILE_NAME;  
  std::string ME_DIR;  
  
  edm::ParameterSet iConfig_;
};

#endif //DQM_SiStripHistoricInfoClient_SiStripHistoricDQMService_H
