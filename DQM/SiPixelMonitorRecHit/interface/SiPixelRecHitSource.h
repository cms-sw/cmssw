#ifndef SiPixelMonitorRecHits_SiPixelRecHitSource_h
#define SiPixelMonitorRecHits_SiPixelRecHitSource_h
// -*- C++ -*-
//
// Package:     SiPixelMonitorRecHits
// Class  :     SiPixelRecHitSource
//
/**

 Description: header file for Pixel Monitor Rec Hits

 Usage:
    see description

*/
//
// Original Author:  Vincenzo Chiochia
//         Created:
//
// Updated by: Keith Rose
// for use in SiPixelMonitorRecHits
// Updated by: Lukas Wehrli
// for pixel offline DQM

#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiPixelMonitorRecHit/interface/SiPixelRecHitModule.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <cstdint>

class SiPixelRecHitSource : public DQMEDAnalyzer {
public:
  explicit SiPixelRecHitSource(const edm::ParameterSet &conf);
  ~SiPixelRecHitSource() override;

  //       typedef edm::DetSet<PixelRecHit>::const_iterator    RecHitIterator;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, const edm::EventSetup &) override;
  void dqmBeginRun(const edm::Run &, edm::EventSetup const &) override;

  virtual void buildStructure(edm::EventSetup const &);
  virtual void bookMEs(DQMStore::IBooker &, const edm::EventSetup &iSetup);

  std::string topFolderName_;

private:
  edm::ParameterSet conf_;
  edm::EDGetTokenT<SiPixelRecHitCollection> src_;

  bool saveFile;
  bool isPIB;
  bool slowDown;
  int eventNo;
  std::map<uint32_t, SiPixelRecHitModule *> thePixelStructure;
  std::map<uint32_t, int> rechit_count;
  bool modOn;
  bool twoDimOn;
  bool reducedSet;
  // barrel:
  bool ladOn, layOn, phiOn;
  // forward:
  bool ringOn, bladeOn, diskOn;

  bool firstRun;
  bool isUpgrade;
};

#endif
