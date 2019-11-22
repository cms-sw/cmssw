#ifndef SiPixelMonitorRawData_SiPixelRawDataErrorSource_h
#define SiPixelMonitorRawData_SiPixelRawDataErrorSource_h
// -*- C++ -*-
//
// Package:     SiPixelMonitorRawData
// Class  :     SiPixelRawDataErrorSource
//
/**

 Description:
 Produces histograms for error information generated at the raw2digi stage for
 the pixel tracker.

 Usage:
 Takes a DetSetVector<SiPixelRawDataError> as input, and uses it to populate  a
 folder hierarchy (organized by detId) with histograms containing information
 about the errors.  Uses SiPixelRawDataErrorModule class to book and fill
 individual folders. Note that this source is different than other DQM sources
 in the creation of an unphysical detId folder (detId=0xffffffff) to hold
 information about errors for which there is no detId available (except the
 dummy detId given to it at raw2digi).

*/
//
// Original Author:  Andrew York
//

#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiPixelMonitorRawData/interface/SiPixelRawDataErrorModule.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <cstdint>

class SiPixelRawDataErrorSource : public DQMOneLumiEDAnalyzer<> {
public:
  explicit SiPixelRawDataErrorSource(const edm::ParameterSet &conf);
  ~SiPixelRawDataErrorSource() override;

  typedef edm::DetSet<SiPixelRawDataError>::const_iterator ErrorIterator;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void dqmBeginRun(const edm::Run &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  virtual void buildStructure(edm::EventSetup const &);
  virtual void bookMEs(DQMStore::IBooker &);

private:
  edm::ParameterSet conf_;
  edm::EDGetTokenT<edm::DetSetVector<SiPixelRawDataError>> src_;
  edm::EDGetTokenT<FEDRawDataCollection> inputSourceToken_;
  std::string topFolderName_;
  bool saveFile;
  bool isPIB;
  bool slowDown;
  bool reducedSet;
  bool modOn;
  bool ladOn;
  bool bladeOn;
  bool isUpgrade;
  int eventNo;
  std::map<uint32_t, SiPixelRawDataErrorModule *> thePixelStructure;
  std::map<uint32_t, SiPixelRawDataErrorModule *> theFEDStructure;
  bool firstRun;
  MonitorElement *byLumiErrors;
  MonitorElement *errorRate;
  MonitorElement *fedcounter;

  MonitorElement *meErrorType_[40];
  MonitorElement *meNErrors_[40];
  MonitorElement *meFullType_[40];
  MonitorElement *meTBMMessage_[40];
  MonitorElement *meTBMType_[40];
  MonitorElement *meEvtNbr_[40];
  MonitorElement *meEvtSize_[40];

  MonitorElement *meFedChNErr_[40];
  MonitorElement *meFedChLErr_[40];
  MonitorElement *meFedETypeNErr_[40];

  std::map<std::string, MonitorElement **> meMapFEDs_;
};

#endif
