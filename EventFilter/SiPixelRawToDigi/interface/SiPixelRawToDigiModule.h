#ifndef SiPixelRawToDigiModule_H
#define SiPixelRawToDigiModule_H

/** \class SiPixelRawToDigiModule_H
 *  Plug-in module that performs Raw data to digi conversion 
 *  for pixel subdetector
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>
#include <string>

class PixelDataFormatter;
class PixelFEDConnectivity;

class SiPixelRawToDigiModule : public edm::EDProducer {
public:

  /// ctor
  explicit SiPixelRawToDigiModule( const edm::ParameterSet& );

  /// dtor
  virtual ~SiPixelRawToDigiModule();

  /// initialisation. Retrieves cabling map from EventSetup. 
  virtual void beginJob( const edm::EventSetup& );

  /// dummy end of job 
  virtual void endJob();

  /// get data, convert to digis attach againe to Event
  virtual void produce( edm::Event&, const edm::EventSetup& );

private:

  /// formatter does actual conversion
  PixelDataFormatter * formatter;
  PixelFEDConnectivity * connectivity;

};
#endif
