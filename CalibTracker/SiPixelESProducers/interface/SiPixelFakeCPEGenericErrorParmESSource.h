#ifndef CalibTracker_SiPixelESProducers_SiPixelFakeCPEGenericErrorParmESSource_h
#define CalibTracker_SiPixelESProducers_SiPixelFakeCPEGenericErrorParmESSource_h

#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"
#include "CondFormats/DataRecord/interface/SiPixelCPEGenericErrorParmRcd.h"

class SiPixelFakeCPEGenericErrorParmESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {

 public:
  SiPixelFakeCPEGenericErrorParmESSource(const edm::ParameterSet &);
  ~SiPixelFakeCPEGenericErrorParmESSource() override;
  
   virtual std::unique_ptr<SiPixelCPEGenericErrorParm>  produce(const SiPixelCPEGenericErrorParmRcd &);
  
 protected:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& ) override;
  
 private:
  
  edm::FileInPath fp_;
	double version_;

};
#endif
