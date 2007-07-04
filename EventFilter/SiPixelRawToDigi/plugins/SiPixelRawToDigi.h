#ifndef SiPixelRawToDigi_H
#define SiPixelRawToDigi_H

/** \class SiPixelRawToDigi_H
 *  Plug-in module that performs Raw data to digi conversion 
 *  for pixel subdetector
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiPixelFedCablingMap;
class TH1D;
class TFile;
class R2DTimerObserver;


class SiPixelRawToDigi : public edm::EDProducer {
public:

  /// ctor
  explicit SiPixelRawToDigi( const edm::ParameterSet& );

  /// dtor
  virtual ~SiPixelRawToDigi();

  /// initialisation. Retrieves cabling map from EventSetup. 
  virtual void beginJob( const edm::EventSetup& );

  /// dummy end of job 
  virtual void endJob() {}

  /// get data, convert to digis attach againe to Event
  virtual void produce( edm::Event&, const edm::EventSetup& );

private:

  unsigned long eventCounter_;
//  edm::InputTag theLabel;
  edm::ParameterSet config_;
  SiPixelFedCablingMap * fedCablingMap_;
  TH1D *hCPU, *hDigi;
  TFile * rootFile;
  R2DTimerObserver * theTimer;
  bool includeErrors;
};
#endif
