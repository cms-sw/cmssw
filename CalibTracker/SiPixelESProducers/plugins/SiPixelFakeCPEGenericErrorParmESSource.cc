#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeCPEGenericErrorParmESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiPixelFakeCPEGenericErrorParmESSource::SiPixelFakeCPEGenericErrorParmESSource(const edm::ParameterSet& conf_) : fp_(conf_.getParameter<edm::FileInPath>("file")), version_(conf_.getParameter<double>("version"))
{
	edm::LogInfo("SiPixelFakeCPEGenericErrorParmESSource::SiPixelFakeCPEGenericErrorParmESSource");
	//the following line is needed to tell the framework what
	// data is being produced
	setWhatProduced(this);
	findingRecord<SiPixelCPEGenericErrorParmRcd>();
}

SiPixelFakeCPEGenericErrorParmESSource::~SiPixelFakeCPEGenericErrorParmESSource()
{
}

std::auto_ptr<SiPixelCPEGenericErrorParm> SiPixelFakeCPEGenericErrorParmESSource::produce(const SiPixelCPEGenericErrorParmRcd & )
{
	using namespace edm::es;
	SiPixelCPEGenericErrorParm * obj = new SiPixelCPEGenericErrorParm();
	obj->fillCPEGenericErrorParm(version_, fp_.fullPath());
	//std::cout << *obj << std::endl;

	return std::auto_ptr<SiPixelCPEGenericErrorParm>(obj);
}

void SiPixelFakeCPEGenericErrorParmESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;  
}
