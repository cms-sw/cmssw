#ifndef SiPixelDigiToRaw_H
#define SiPixelDigiToRaw_H

/** \class SiPixelDigiToRaw_H
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
class SiPixelFedCablingTree;

class SiPixelDigiToRaw : public edm::EDProducer {
public:

  /// ctor
  explicit SiPixelDigiToRaw( const edm::ParameterSet& );

  /// dtor
  virtual ~SiPixelDigiToRaw();


  /// dummy end of job 
  virtual void endJob() {}

  /// get data, convert to raw event, attach again to Event
  virtual void produce( edm::Event&, const edm::EventSetup& );

private:

  SiPixelFedCablingTree * cablingTree_;
  edm::ParameterSet config_;
};
#endif
