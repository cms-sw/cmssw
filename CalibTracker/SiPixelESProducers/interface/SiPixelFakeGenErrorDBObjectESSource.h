#ifndef CalibTracker_SiPixelESProducers_SiPixelFakeGenErrorDBObjectESSource_h
#define CalibTracker_SiPixelESProducers_SiPixelFakeGenErrorDBObjectESSource_h

#include <memory>
#include "boost/shared_ptr.hpp"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"

class SiPixelFakeGenErrorDBObjectESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {

 public:
  SiPixelFakeGenErrorDBObjectESSource(const edm::ParameterSet &);
  ~SiPixelFakeGenErrorDBObjectESSource();
  
  typedef std::vector<std::string> vstring;
  
  virtual std::auto_ptr<SiPixelGenErrorDBObject>  produce(const SiPixelGenErrorDBObjectRcd &);
  
 protected:
  
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
 private:
  
 	vstring GenErrorCalibrations_;
	float version_;

};
#endif
