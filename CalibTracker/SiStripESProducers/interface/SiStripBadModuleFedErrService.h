#ifndef SiStripESSources_SiStripBadModuleFedErrService_h
#define SiStripESSources_SiStripBadModuleFedErrService_h
// -*- C++ -*-
//
//
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondTools/SiStrip/interface/SiStripDepCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DQMServices/Core/interface/DQMStore.h"

class MonitorElement;
class SiStripQuality;

class SiStripBadModuleFedErrService : public SiStripDepCondObjBuilderBase<SiStripBadStrip, SiStripFedCabling> {

public:

  explicit SiStripBadModuleFedErrService(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripBadModuleFedErrService();

  /// Used to fill the logDB
  void getMetaDataString(std::stringstream& ss);

  /// Check is the transfer is needed
  virtual bool checkForCompatibility(std::string ss);

  void getObj(SiStripBadStrip* & obj, const SiStripFedCabling * cabling){obj = readBadComponentsFromFed(cabling);}
  uint32_t getRunNumber() const;

private:


  SiStripBadStrip* readBadComponentsFromFed(const SiStripFedCabling* cabling);
  bool openRequestedFile();
  void getFedBadChannelList(MonitorElement* me, std::vector<std::pair<uint16_t, uint16_t>>& list);
  float getProcessedEvents();

  DQMStore* dqmStore_;

  edm::ParameterSet iConfig_;
  bool notAlreadyRead_;
};
#endif
