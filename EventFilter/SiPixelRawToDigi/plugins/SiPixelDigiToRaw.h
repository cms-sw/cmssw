#ifndef SiPixelDigiToRaw_H
#define SiPixelDigiToRaw_H

/** \class SiPixelDigiToRaw_H
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
class SiPixelFedCablingMap;

class SiPixelDigiToRaw : public edm::EDProducer {
public:

  /// ctor
  explicit SiPixelDigiToRaw( const edm::ParameterSet& );

  /// dtor
  virtual ~SiPixelDigiToRaw();

  /// initialisation. Retrieves cabling map from EventSetup. 
  virtual void beginJob( const edm::EventSetup& );

  /// dummy end of job 
  virtual void endJob() {}

  /// get data, convert to raw event, attach again to Event
  virtual void produce( edm::Event&, const edm::EventSetup& );

private:

  unsigned long eventCounter_;
  SiPixelFedCablingMap * fedCablingMap_;
//  edm::InputTag src_;
  edm::ParameterSet config_;
};
#endif
