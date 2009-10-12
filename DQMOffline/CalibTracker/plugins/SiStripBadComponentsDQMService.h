#ifndef DQMOffline_SiStripBadComponentsDQMService_SiStripBadComponentsDQMService_H
#define DQMOffline_SiStripBadComponentsDQMService_SiStripBadComponentsDQMService_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include <string>
#include <memory>
#include <sstream>
// #include "DQM/SiStripMonitorClient/interface/SiStripQualityChecker.h"

using namespace std;

/**
  @class SiStripBadComponentsDQMService
  @author M. De Mattia, S. Dutta, D. Giordano
  @EDAnalyzer to read modules flagged by the DQM as bad and write in the database.
*/

class SiStripBadComponentsDQMService : public SiStripCondObjBuilderBase<SiStripBadStrip> {
 public:

  explicit SiStripBadComponentsDQMService(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripBadComponentsDQMService();
  
  /// Used to fill the logDB
  void getMetaDataString(std::stringstream& ss);

  /// Check is the transfer is needed
  virtual bool checkForCompatibility(std::string ss);

  void getObj(SiStripBadStrip* & obj){readBadComponents(); obj=obj_;}

 private:

  void readBadComponents();
  void openRequestedFile();
  // void scanTreeAndFillSummary(const std::vector<MonitorElement*>& MEs, SiStripBadStrip* summary,std::string& histoName, std::vector<std::string>& Quantities);
  uint32_t getRunNumber() const;
  bool goToDir(DQMStore * dqm_store, string name);
  void getModuleFolderList(DQMStore * dqm_store, vector<string>& mfolders);

  DQMStore* dqmStore_;

  edm::ParameterSet iConfig_;
  edm::FileInPath fp_;
  // SiStripQualityChecker*   qualityChecker_;
  bool notAlreadyRead_;
};

#endif //DQMOffline_SiStripBadComponentsDQMService_SiStripBadComponentsDQMService_H
