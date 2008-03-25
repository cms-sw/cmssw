#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <vector>
#include <memory>

#include <boost/shared_ptr.hpp>

#include <TClass.h>
#include <TBufferXML.h>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"

using namespace PhysicsTools::Calibration;

class BTauFakeMVAJetTagConditions : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
    public:
	typedef boost::shared_ptr<MVAComputerContainer> ReturnType;

	BTauFakeMVAJetTagConditions(const edm::ParameterSet &params);
	virtual ~BTauFakeMVAJetTagConditions();

	ReturnType produce(const BTauGenericMVAJetTagComputerRcd &record);

    private:
	void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
	                    const edm::IOVSyncValue &syncValue,
	                    edm::ValidityInterval &oValidity);

	edm::FileInPath	xmlCalibration;
};

BTauFakeMVAJetTagConditions::BTauFakeMVAJetTagConditions(
					const edm::ParameterSet &params) :
	xmlCalibration(params.getParameter<edm::FileInPath>("xmlCalibration"))
{
	setWhatProduced(this);
	findingRecord<BTauGenericMVAJetTagComputerRcd>();
}

BTauFakeMVAJetTagConditions::~BTauFakeMVAJetTagConditions()
{
}

BTauFakeMVAJetTagConditions::ReturnType
BTauFakeMVAJetTagConditions::produce(
				const BTauGenericMVAJetTagComputerRcd &record)
{
	std::ifstream xmlFile(xmlCalibration.fullPath().c_str());
	if (!xmlFile.good())
		throw cms::Exception("BTauFakeMVAJetTagConditions")
			<< "File \"" << xmlCalibration.fullPath()
			<< "\" could not be opened for reading."
			<< std::endl;

	std::ostringstream ss;
	ss << xmlFile.rdbuf();
	xmlFile.close();

	TClass *classType = 0;
	void *ptr = TBufferXML(TBuffer::kRead).ConvertFromXMLAny(
				ss.str().c_str(), &classType, kTRUE, kFALSE);
	if (!ptr)
		throw cms::Exception("BTauFakeMVAJetTagConditions")
			<< "Unknown error parsing XML serialization"
			<< std::endl;

	if (std::strcmp(classType->GetName(),
		"PhysicsTools::Calibration::MVAComputerContainer")) {
		classType->Destructor(ptr);
		throw cms::Exception("BTauFakeMVAJetTagConditions")
			<< "Serialized object has wrong C++ type."
			<< std::endl;
	}

	return ReturnType(static_cast<MVAComputerContainer*>(ptr));
}

void BTauFakeMVAJetTagConditions::setIntervalFor(
			const edm::eventsetup::EventSetupRecordKey &key,
			const edm::IOVSyncValue &syncValue,
			edm::ValidityInterval &oValidity)
{
	oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),
	                                  edm::IOVSyncValue::endOfTime());
}

DEFINE_FWK_EVENTSETUP_SOURCE(BTauFakeMVAJetTagConditions);
