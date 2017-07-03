#ifndef CalibTracker_SiPixelESProducers_SiPixelFakeGenErrorDBObjectESSource_h
#define CalibTracker_SiPixelESProducers_SiPixelFakeGenErrorDBObjectESSource_h

#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"

class SiPixelFakeGenErrorDBObjectESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {

 public:
  SiPixelFakeGenErrorDBObjectESSource(const edm::ParameterSet &);
  ~SiPixelFakeGenErrorDBObjectESSource() override;
  
  typedef std::vector<std::string> vstring;
  
  virtual std::unique_ptr<SiPixelGenErrorDBObject>  produce(const SiPixelGenErrorDBObjectRcd &);
  
 protected:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& ) override;
  
 private:
  
 	vstring GenErrorCalibrations_;
	float version_;

};
#endif
