#ifndef CalibTracker_SiPixelESProducers_SiPixelFakeTemplateDBObjectESSource_h
#define CalibTracker_SiPixelESProducers_SiPixelFakeTemplateDBObjectESSource_h

#include <memory>
#include "boost/shared_ptr.hpp"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObjectRcd.h"

class SiPixelFakeTemplateDBObjectESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {

 public:
  SiPixelFakeTemplateDBObjectESSource(const edm::ParameterSet &);
  ~SiPixelFakeTemplateDBObjectESSource();
  
  typedef std::vector<std::string> vstring;
  
  virtual std::auto_ptr<SiPixelTemplateDBObject>  produce(const SiPixelTemplateDBObjectRcd &);
  
 protected:
  
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
 private:
  
 	vstring templateCalibrations_;
	float version_;

};
#endif
