#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <vector>
#include <memory>

#include <boost/shared_ptr.hpp>

#include <TClass.h>
#include <TBuffer.h>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DataRecord/interface/BTagTrackProbability2DRcd.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability3DRcd.h"
#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"


#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"

//using namespace PhysicsTools::Calibration;

class TrackProbabilityFakeCondES : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
    public:
	typedef boost::shared_ptr<TrackProbabilityCalibration> ReturnType;

	TrackProbabilityFakeCondES(const edm::ParameterSet &params);
	virtual ~TrackProbabilityFakeCondES();

	ReturnType produce2D(const BTagTrackProbability2DRcd &record);
	ReturnType produce3D(const BTagTrackProbability3DRcd &record);
	ReturnType calibration(const edm::FileInPath &);

    private:
	void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
	                    const edm::IOVSyncValue &syncValue,
	                    edm::ValidityInterval &oValidity);

	edm::FileInPath	xmlCalibration;
};

TrackProbabilityFakeCondES::TrackProbabilityFakeCondES(
					const edm::ParameterSet &params) 
{
	setWhatProduced(this,&TrackProbabilityFakeCondES::produce2D);
	setWhatProduced(this,&TrackProbabilityFakeCondES::produce3D);

findingRecord<BTagTrackProbability2DRcd>();
findingRecord<BTagTrackProbability3DRcd>();
}

TrackProbabilityFakeCondES::~TrackProbabilityFakeCondES()
{
}

TrackProbabilityFakeCondES::ReturnType
TrackProbabilityFakeCondES::produce2D(
				const BTagTrackProbability2DRcd &record)
{
 return calibration(edm::FileInPath("RecoBTag/TrackProbability/data/2D.dat"));

}
TrackProbabilityFakeCondES::ReturnType
TrackProbabilityFakeCondES::produce3D(
                                const BTagTrackProbability3DRcd &record)
{
 return calibration(edm::FileInPath("RecoBTag/TrackProbability/data/3D.dat"));
}

TrackProbabilityFakeCondES::ReturnType TrackProbabilityFakeCondES::calibration(const edm::FileInPath & file)
{
       std::ifstream xmlFile(file.fullPath().c_str());
        if (!xmlFile.good())
                throw cms::Exception("TrackProbabilityFakeCondES")
                        << "File \"" << xmlCalibration.fullPath()
                        << "\" could not be opened for reading."
                        << std::endl;

        std::ostringstream ss;
        ss << xmlFile.rdbuf();
        xmlFile.close();
        std::string s = ss.str();
        TBuffer b(TBuffer::kRead,s.size(), const_cast<void*>( static_cast<const void*>(s.c_str())),  kFALSE);
        b.InitMap();
        TrackProbabilityCalibration * c = new TrackProbabilityCalibration();
        TClass  * a =  TClass::GetClass("TrackProbabilityCalibration");
        b.StreamObject(c,a);
        return ReturnType(c);
}


void TrackProbabilityFakeCondES::setIntervalFor(
			const edm::eventsetup::EventSetupRecordKey &key,
			const edm::IOVSyncValue &syncValue,
			edm::ValidityInterval &oValidity)
{
	oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),
	                                  edm::IOVSyncValue::endOfTime());
}

DEFINE_FWK_EVENTSETUP_SOURCE(TrackProbabilityFakeCondES);
