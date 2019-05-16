#ifndef SiStripMonitorPedestals_SiStripMonitorPedestals_h
#define SiStripMonitorPedestals_SiStripMonitorPedestals_h
// -*- C++ -*-
//
// Package:     SiStripMonitorPedestals
// Class  :     SiStripMonitorPedestals
//
/**\class SiStripMonitorPedestals SiStripMonitorPedestals.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  gennai, dutta
//         Created:  Sat Feb  4 20:49:51 CET 2006
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// data formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

// cabling
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "boost/cstdint.hpp"
#include "DQMServices/Core/interface/DQMStore.h"
#include <iomanip>
#include <string>

class ApvAnalysisFactory;
class SiStripDetCabling;

class SiStripMonitorPedestals : public DQMEDAnalyzer {
public:
  explicit SiStripMonitorPedestals(const edm::ParameterSet &);
  ~SiStripMonitorPedestals() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endRun(edm::Run const &run, edm::EventSetup const &eSetup) override;
  void endJob() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi>> digiToken_;

  void resetMEs(uint32_t idet);
  void createMEs(DQMStore::IBooker &, edm::EventSetup const &eSetup);
  void fillCondDBMEs(edm::EventSetup const &eSetup);

  struct ModMEs {
    MonitorElement *PedsPerStrip;
    MonitorElement *PedsDistribution;
    MonitorElement *PedsEvolution;

    MonitorElement *CMSubNoisePerStrip;
    MonitorElement *RawNoisePerStrip;
    MonitorElement *CMSubNoiseProfile;
    MonitorElement *RawNoiseProfile;
    MonitorElement *NoisyStrips;
    MonitorElement *NoisyStripDistribution;

    MonitorElement *CMDistribution;
    MonitorElement *CMSlopeDistribution;

    // MonitorElements for CondDB data display
    MonitorElement *PedsPerStripDB;
    MonitorElement *CMSubNoisePerStripDB;
    MonitorElement *BadStripsDB;
  };

  DQMStore *dqmStore_;
  edm::ParameterSet conf_;
  std::map<uint32_t, ModMEs> PedMEs;
  edm::ESHandle<SiStripDetCabling> detcabling;
  edm::ParameterSet pedsPSet_;
  bool analyzed;
  bool firstEvent;

  // The following to be put inside the parametersets
  int16_t nEvUpdate_;
  int16_t signalCutPeds_;
  int16_t nEvTot_;
  int16_t nEvInit_;
  int nIteration_;
  ApvAnalysisFactory *apvFactory_;
  int theEventInitNumber_;
  int theEventIterNumber_;
  int NumCMstripsInGroup_;
  std::string runTypeFlag_;
  std::string outPutFileName;
  unsigned long long m_cacheID_;

  static const std::string RunMode1;
  static const std::string RunMode2;
  static const std::string RunMode3;
};

#endif
