#ifndef SiPixelDigiToRaw_H
#define SiPixelDigiToRaw_H

/** \class SiPixelDigiToRaw_H
 */

#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameReverter.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

class SiPixelFedCablingTree;
class SiPixelFrameReverter;
class TH1D;
class R2DTimerObserver;

class SiPixelDigiToRaw final : public edm::EDProducer {
public:

  /// ctor
  explicit SiPixelDigiToRaw( const edm::ParameterSet& );

  /// dtor
  virtual ~SiPixelDigiToRaw();


  /// dummy end of job 
  virtual void endJob() {}

  /// get data, convert to raw event, attach again to Event
  virtual void produce( edm::Event&, const edm::EventSetup& ) override;

private:

  std::unique_ptr<SiPixelFedCablingTree> cablingTree_;
  SiPixelFrameReverter* frameReverter_;
  edm::ParameterSet config_;
  TH1D *hCPU, *hDigi;
  R2DTimerObserver * theTimer;
  unsigned long eventCounter;
  edm::InputTag label;  //label of input digi data
  int allDigiCounter;
  int allWordCounter;
  std::vector<unsigned int> fedIds;
  edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher;
  bool debug;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> tPixelDigi; 
};
#endif
