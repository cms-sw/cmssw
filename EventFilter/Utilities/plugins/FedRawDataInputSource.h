#ifndef EventFilter_Utilities_FedRawDataInputSource_h
#define EventFilter_Utilities_FedRawDataInputSource_h

#include <memory>
#include <stdio.h>

#include "boost/filesystem.hpp"

#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Sources/interface/RawInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Sources/interface/DaqProvenanceHelper.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"

#include "../interface/JsonMonitorable.h"
#include "../interface/DataPointMonitor.h"
#include "../interface/JSONSerializer.h"

#include "EvFDaqDirector.h"

using namespace jsoncollector;

class FEDRawDataCollection;
class InputSourceDescription;
class ParameterSet;

class FedRawDataInputSource: public edm::RawInputSource {

public:
	explicit FedRawDataInputSource(edm::ParameterSet const&,
			edm::InputSourceDescription const&);
	virtual ~FedRawDataInputSource();

protected:
	virtual bool checkNextEvent();
	virtual void read(edm::EventPrincipal& eventPrincipal);

private:
	virtual void preForkReleaseResources();
	virtual void postForkReacquireResources(
			boost::shared_ptr<edm::multicore::MessageReceiverForSource>);
	virtual void rewind_();

	void createWorkingDirectory();
	//void findRunDir(const std::string& rootFUDirectory);
	void findRunDir();
	edm::Timestamp fillFEDRawDataCollection(
			std::auto_ptr<FEDRawDataCollection>&);
	bool openNextFile();
	void openFile(boost::filesystem::path const&);
	bool searchForNextFile(boost::filesystem::path const&);
	//bool grabNextFile(boost::filesystem::path const&,boost::filesystem::path const&);
	bool grabNextFile(boost::filesystem::path&, boost::filesystem::path const&);
	bool eofReached() const;
	bool runEnded() const;

	uint32_t getEventSizeFromBuffer();
	uint32_t getPaddingSizeFromBuffer();
	uint32_t fillFedSizesFromBuffer(uint32_t *fedSizes);
	bool getEventHeaderFromBuffer(FRDEventHeader_V2 *eventHeader);
	bool checkIfBuffered();

	// get LS from filename instead of event header
	bool getLSFromFilename_;

	bool testModeNoBuilderUnit_;

	edm::DaqProvenanceHelper daqProvenanceHelper_;

	// the BU run directory
	boost::filesystem::path buRunDirectory_;
	// the OUTPUT run directory
	boost::filesystem::path localRunBaseDirectory_;
	boost::filesystem::path localRunDirectory_;
	edm::RunNumber_t runNumber_;
	uint32_t formatVersion_;

	boost::filesystem::path workingDirectory_;
	boost::filesystem::path openFile_;
	size_t fileIndex_;
	FILE* fileStream_;
	bool workDirCreated_;
	edm::EventID eventID_;

	unsigned int lastOpenedLumi_;
	boost::filesystem::path currentDataDir_;
	bool eorFileSeen_;
	uint32_t buffer_left;
	uint32_t buffer_cursor;
	unsigned int eventChunkSize_;
	unsigned char *data_buffer; // temporarily hold multiple event data
};

#endif // EventFilter_Utilities_FedRawDataInputSource_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
