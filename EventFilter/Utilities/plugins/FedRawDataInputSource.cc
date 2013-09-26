#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <sys/file.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <stdio.h>

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"
#include "EventFilter/Utilities/plugins/FedRawDataInputSource.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/FEDInterface/interface/fed_header.h"
#include "EventFilter/FEDInterface/interface/fed_trailer.h"

#include "EventFilter/Utilities/interface/FileIO.h"
#include "FastMonitoringService.h"

FedRawDataInputSource::FedRawDataInputSource(edm::ParameterSet const& pset,
		edm::InputSourceDescription const& desc) :
			edm::RawInputSource(pset, desc),
			getLSFromFilename_(
					pset.getUntrackedParameter<bool> ("getLSFromFilename", true)),
			testModeNoBuilderUnit_(
					pset.getUntrackedParameter<bool> ("testModeNoBuilderUnit",
							false)),
			daqProvenanceHelper_(edm::TypeID(typeid(FEDRawDataCollection))),
			formatVersion_(0),
			fileIndex_(0),
			fileStream_(0),
			workDirCreated_(false),
			eventID_(),
			lastOpenedLumi_(0),
			eorFileSeen_(false),
			buffer_left(0),
			eventChunkSize_(
					pset.getUntrackedParameter<unsigned int> ("eventChunkSize",
							16)),
			data_buffer(new unsigned char[1024 * 1024 * eventChunkSize_]) {

	std::cout << "FedRawDataInputSource -> test mode: "
			<< testModeNoBuilderUnit_ << ", chunk size: " << eventChunkSize_
			<< std::endl;
	buRunDirectory_ = boost::filesystem::path(
			pset.getUntrackedParameter<std::string> ("rootBUDirectory"));
	findRunDir();
	daqProvenanceHelper_.daqInit(productRegistryUpdate(), processHistoryRegistryForUpdate());
	setNewRun();
	setRunAuxiliary(
			new edm::RunAuxiliary(runNumber_, edm::Timestamp::beginOfTime(),
					edm::Timestamp::invalidTimestamp()));
}

FedRawDataInputSource::~FedRawDataInputSource() {
	if (fileStream_)
		fclose(fileStream_);
	fileStream_ = 0;
}

void FedRawDataInputSource::findRunDir() {
	boost::filesystem::path runDirectory(edm::Service<evf::EvFDaqDirector>()->findHighestRunDir());
	localRunDirectory_ = localRunBaseDirectory_ = runDirectory;
	runNumber_ = edm::Service<evf::EvFDaqDirector>()->findHighestRun();

	// get the corresponding BU dir
	buRunDirectory_ /= localRunDirectory_.filename();

	edm::LogInfo("FedRawDataInputSource") << "Getting data from "
			<< buRunDirectory_.string();
}

bool FedRawDataInputSource::checkNextEvent() {
	FRDEventHeader_V2 eventHeader;
	if (!getEventHeaderFromBuffer(&eventHeader)) {
		// run has ended
		resetLuminosityBlockAuxiliary();
		//if (workDirCreated_)
		//  boost::filesystem::remove(workingDirectory_);
		return false;
	}

	assert(eventHeader.version_ > 1);
	formatVersion_ = eventHeader.version_;

	//same lumi, or new lumi detected in file (old mode)
	if (!getLSFromFilename_) {
		//get new lumi from file header
		if (!luminosityBlockAuxiliary()
				|| luminosityBlockAuxiliary()->luminosityBlock()
						!= eventHeader.lumi_) {
			lastOpenedLumi_ = eventHeader.lumi_;
			resetLuminosityBlockAuxiliary();
			timeval tv;
			gettimeofday(&tv, 0);
			edm::Timestamp lsopentime(
					(unsigned long long) tv.tv_sec * 1000000
							+ (unsigned long long) tv.tv_usec);
			edm::LuminosityBlockAuxiliary* luminosityBlockAuxiliary =
					new edm::LuminosityBlockAuxiliary(runAuxiliary()->run(),
							eventHeader.lumi_, lsopentime,
							edm::Timestamp::invalidTimestamp());
			setLuminosityBlockAuxiliary(luminosityBlockAuxiliary);
		}
	} else {
		//new lumi from directory name
		if (!luminosityBlockAuxiliary()
				|| luminosityBlockAuxiliary()->luminosityBlock()
						!= lastOpenedLumi_) {
			resetLuminosityBlockAuxiliary();

			timeval tv;
			gettimeofday(&tv, 0);
			edm::Timestamp lsopentime(
					(unsigned long long) tv.tv_sec * 1000000
							+ (unsigned long long) tv.tv_usec);
			edm::LuminosityBlockAuxiliary* luminosityBlockAuxiliary =
					new edm::LuminosityBlockAuxiliary(runAuxiliary()->run(),
							lastOpenedLumi_, lsopentime,
							edm::Timestamp::invalidTimestamp());
			setLuminosityBlockAuxiliary(luminosityBlockAuxiliary);
		}
	}

	eventID_ = edm::EventID(eventHeader.run_, lastOpenedLumi_,
			eventHeader.event_);
	setEventCached();

	return true;

}

void FedRawDataInputSource::read(
		edm::EventPrincipal& eventPrincipal) {
	//std::cout << ">>>>>>>>>>>>>>>> Reading next event" << std::endl;
	std::auto_ptr<FEDRawDataCollection> rawData(new FEDRawDataCollection);
	edm::Timestamp tstamp = fillFEDRawDataCollection(rawData);

	edm::EventAuxiliary aux(eventID_, processGUID(), tstamp, true,
			edm::EventAuxiliary::PhysicsTrigger);

	makeEvent(eventPrincipal, aux);

	edm::WrapperOwningHolder edp(
			new edm::Wrapper<FEDRawDataCollection>(rawData),
			edm::Wrapper<FEDRawDataCollection>::getInterface());
	eventPrincipal.put(daqProvenanceHelper_.constBranchDescription_, edp,
		           daqProvenanceHelper_.dummyProvenance_);
	return;
}

bool FedRawDataInputSource::eofReached() const {
	if (fileStream_ == 0)
		return true;

	int c;
	c = fgetc(fileStream_);
	ungetc(c, fileStream_);

	return (c == EOF);
}

edm::Timestamp FedRawDataInputSource::fillFEDRawDataCollection(
		std::auto_ptr<FEDRawDataCollection>& rawData) {
	edm::Timestamp tstamp;
	uint32_t eventSize = 0;
	uint32_t paddingSize = 0;
	if (formatVersion_ >= 3) {
		eventSize = getEventSizeFromBuffer();
		paddingSize = getPaddingSizeFromBuffer();
	}
	uint32_t fedSizes[1024];
	eventSize += fillFedSizesFromBuffer(fedSizes);

	/*
	 if (formatVersion_ < 3) {
	 for (unsigned int i = 0; i < 1024; i++)
	 eventSize += fedSizes[i];
	 }
	 */

	unsigned int gtpevmsize = fedSizes[FEDNumbering::MINTriggerGTPFEDID];
	if (gtpevmsize > 0)
		evf::evtn::evm_board_setformat(gtpevmsize);

	//todo put this in a separate function
	if (buffer_left < eventSize)
		checkIfBuffered();
	char* event = (char *) (data_buffer + buffer_cursor);
	buffer_left -= eventSize;
	buffer_cursor += eventSize;

	while (eventSize > 0) {
		eventSize -= sizeof(fedt_t);
		const fedt_t* fedTrailer = (fedt_t*) (event + eventSize);
		const uint32_t fedSize = FED_EVSZ_EXTRACT(fedTrailer->eventsize) << 3; //trailer length counts in 8 bytes
		eventSize -= (fedSize - sizeof(fedh_t));
		const fedh_t* fedHeader = (fedh_t *) (event + eventSize);
		const uint16_t fedId = FED_SOID_EXTRACT(fedHeader->sourceid);
		if (fedId == FEDNumbering::MINTriggerGTPFEDID) {
			const uint64_t gpsl = evf::evtn::getgpslow(
					(unsigned char*) fedHeader);
			const uint64_t gpsh = evf::evtn::getgpshigh(
					(unsigned char*) fedHeader);
			tstamp = edm::Timestamp(
					static_cast<edm::TimeValue_t> ((gpsh << 32) + gpsl));
		}
		FEDRawData& fedData = rawData->FEDData(fedId);
		fedData.resize(fedSize);
		memcpy(fedData.data(), event + eventSize, fedSize);
	}
	assert(eventSize == 0);
	buffer_left -= paddingSize;
	buffer_cursor += paddingSize;
	//	delete event;
	return tstamp;
}

bool FedRawDataInputSource::openNextFile() {
	if (!workDirCreated_)
		createWorkingDirectory();
	//boost::filesystem::path nextFile = workingDirectory_;
	boost::filesystem::path nextFile = buRunDirectory_;
	std::ostringstream fileName;
	char thishost[256];
	gethostname(thishost, 255);
	fileName << std::setfill('0') << std::setw(16) << fileIndex_++ << "_"
			<< thishost << "_" << getpid() << ".raw";
	nextFile /= fileName.str();

	openFile(nextFile);//closes previous file


	while (!searchForNextFile(nextFile) && !eorFileSeen_) {
		std::cout << "No file for me... sleep and try again..." << std::endl;
		usleep(250000);
	}

	return (fileStream_ != 0 || !eorFileSeen_);
}

void FedRawDataInputSource::openFile(boost::filesystem::path const& nextFile) {
	if (fileStream_) {
		fclose(fileStream_);
		fileStream_ = 0;

		if (!testModeNoBuilderUnit_) {
			boost::filesystem::remove(openFile_); // wont work in case of forked children
		} else {
			boost::filesystem::path fileToRename(openFile_);
			unsigned int jumpLS =
					edm::Service<evf::EvFDaqDirector>()->getJumpLS();
			unsigned int jumpIndex =
					edm::Service<evf::EvFDaqDirector>()->getJumpIndex();
			stringstream ss;
			ss << buRunDirectory_.string() << "/ls" << std::setfill('0')
					<< std::setw(4) << jumpLS << "_index" << std::setfill('0')
					<< std::setw(6) << jumpIndex << ".raw";
			string path = ss.str();
			std::cout << "Instead of delete, RENAME: " << openFile_ << " to: "
					<< path << std::endl;
			int rc = rename(openFile_.string().c_str(), path.c_str());
			if (rc != 0) {
				std::cout << "RENAME RAW FAILED!" << std::endl;
			}

			std::cout << "Also rename json: " << openFile_ << " to: " << path
					<< std::endl;
			ss.str("");
			ss << buRunDirectory_.string() << "/ls" << std::setfill('0')
					<< std::setw(4) << jumpLS - 2 << "_index" << std::setfill(
					'0') << std::setw(6) << jumpIndex << ".jsn";
			string sourceJson = ss.str();
			ss.str("");
			ss << buRunDirectory_.string() << "/ls" << std::setfill('0')
					<< std::setw(4) << jumpLS << "_index" << std::setfill('0')
					<< std::setw(6) << jumpIndex << ".jsn";
			string destJson = ss.str();
			rc = rename(sourceJson.c_str(), destJson.c_str());
			if (rc != 0) {
				std::cout << "RENAME JSON FAILED!" << std::endl;
			}
		}

	}

	const int fileDescriptor = open(nextFile.c_str(), O_RDONLY);
	if (fileDescriptor != -1) {
		fileStream_ = fdopen(fileDescriptor, "rb");
		openFile_ = nextFile;
	}
	std::cout << " tried to open file.. " << nextFile << " fd:"
			<< fileDescriptor << std::endl;
}

bool FedRawDataInputSource::searchForNextFile(
		boost::filesystem::path const& nextFile) {
	bool fileIsOKToGrab = false;
	std::stringstream ss;
	unsigned int ls = lastOpenedLumi_;
	unsigned int index;
	unsigned int initialLS = ls;

	std::cout << "Asking for next file... to the DaqDirector" << std::endl;
	evf::FastMonitoringService
			*fms = (evf::FastMonitoringService *) (edm::Service<
					evf::MicroStateService>().operator->());
	fms->startedLookingForFile();
	fileIsOKToGrab = edm::Service<evf::EvFDaqDirector>()->updateFuLock(ls,
			index, eorFileSeen_);

	if (fileIsOKToGrab) {
		ss << buRunDirectory_.string() << "/ls" << std::setfill('0')
				<< std::setw(4) << ls << "_index" << std::setfill('0')
				<< std::setw(6) << index << ".raw";
		string path = ss.str();
		std::cout << "The director says to grab: " << path << std::endl;
		fms->stoppedLookingForFile();
		std::cout << "grabbin next file, setting last seen lumi to LS = " << ls
				<< std::endl;
		boost::filesystem::path theFileToGrab(ss.str());
		if (grabNextFile(theFileToGrab, nextFile)) {
			if (getLSFromFilename_)
				lastOpenedLumi_ = ls;
			return true;
		} else {
			std::cout << "GRABBING SHOULD NEVER FAIL! THE HORROOOOOOOOOOOOOOR!"
					<< std::endl;
			//return false;
			exit(0);
		}
	} else if (ls > initialLS) {
		// ls was increased, so some EoL jsn files were seen, without new data files
		std::cout << "EoL jsn file(s) seen! Current LS is: " << ls << std::endl;
		resetLuminosityBlockAuxiliary();
		timeval tv;
		gettimeofday(&tv, 0);
		edm::Timestamp lsopentime(
				(unsigned long long) tv.tv_sec * 1000000
						+ (unsigned long long) tv.tv_usec);
		edm::LuminosityBlockAuxiliary* luminosityBlockAuxiliary =
				new edm::LuminosityBlockAuxiliary(runAuxiliary()->run(), ls,
						lsopentime, edm::Timestamp::invalidTimestamp());
		setLuminosityBlockAuxiliary(luminosityBlockAuxiliary);

		return false;

	} else {
		std::cout << "The DAQ Director has nothing for me! " << std::endl;
		if (eorFileSeen_) {
			std::cout << "...and he's seen the end of run file!" << std::endl;
		}
		return false;
	}
}

bool FedRawDataInputSource::grabNextFile(boost::filesystem::path& file,
		boost::filesystem::path const& nextFile) {
	try {
		// assemble json path on /hlt/data
		boost::filesystem::path nextFileJson = workingDirectory_;
		boost::filesystem::path jsonSourcePath(file);
		boost::filesystem::path jsonDestPath(nextFileJson);
		boost::filesystem::path jsonExt(".jsn");
		jsonSourcePath.replace_extension(jsonExt);
		boost::filesystem::path jsonTempPath(jsonDestPath);

		std::ostringstream fileNameWithPID;
		fileNameWithPID << jsonSourcePath.stem().string() << "_pid"
				<< std::setfill('0') << std::setw(5) << getpid() << ".jsn";
		boost::filesystem::path filePathWithPID(fileNameWithPID.str());
		jsonTempPath /= filePathWithPID;
		std::cout << " JSON rename " << jsonSourcePath << " to "
				<< jsonTempPath << std::endl;
		//COPY JSON
		string sysCall = "mv ";
		if (testModeNoBuilderUnit_)
			sysCall = "cp ";
		string mvCmd = sysCall + jsonSourcePath.string() + " "
				+ jsonTempPath.string();
		std::cout << " Running cmd = " << mvCmd << std::endl;
		int rc = system(mvCmd.c_str());
		//std::cout << " return code = " << rc << std::endl;
		if (rc != 0) {
			throw std::runtime_error("Cannot copy JSON file, rc is != 0!");
		}

		//openFile(nextFile);
		openFile(file);

		return true;
	}

	catch (const boost::filesystem::filesystem_error& ex) {
		// Input dir gone?
		std::cout << "BOOST FILESYSTEM ERROR CAUGHT: " << ex.what()
				<< std::endl;
		std::cout
				<< "Maybe the BU run dir disappeared? Ending process with code 0..."
				<< std::endl;
		exit(0);
	} catch (std::runtime_error e) {
		// Another process grabbed the file and NFS did not register this
		std::cout << "Exception text: " << e.what() << std::endl;
	} catch (std::exception e) {
		// BU run directory disappeared?
		std::cout << "SOME OTHER EXCEPTION OCCURED!!!! ->" << e.what()
				<< std::endl;
	}
	return false;
}

bool FedRawDataInputSource::runEnded() const {
	boost::filesystem::path endOfRunMarker = buRunDirectory_;
	endOfRunMarker /= "EndOfRun.jsn";
	return boost::filesystem::exists(endOfRunMarker);
}

void FedRawDataInputSource::preForkReleaseResources() {
}

void FedRawDataInputSource::postForkReacquireResources(
		boost::shared_ptr<edm::multicore::MessageReceiverForSource>) {
	createWorkingDirectory();
	InputSource::rewind();
	setRunAuxiliary(
			new edm::RunAuxiliary(runNumber_, edm::Timestamp::beginOfTime(),
					edm::Timestamp::invalidTimestamp()));
}

void FedRawDataInputSource::rewind_() {
}

void FedRawDataInputSource::createWorkingDirectory() {

	char thishost[256];
	gethostname(thishost, 255);
	std::ostringstream myDir;

	workingDirectory_ = localRunBaseDirectory_;

	boost::filesystem::directory_iterator itEnd;
	bool openDirFound = false;
	for (boost::filesystem::directory_iterator it(localRunBaseDirectory_); it
			!= itEnd; ++it) {
		if (boost::filesystem::is_directory(it->path())
				&& it->path().string().find("/open") != std::string::npos)
			openDirFound = true;
	}

	// workingDirectory_ /= "open";

	if (!openDirFound) {
		boost::filesystem::create_directories(workingDirectory_);
	}

	workDirCreated_ = true;

	// also create MON directory

	boost::filesystem::path monDirectory = localRunBaseDirectory_;
	monDirectory /= "mon";

	bool foundMonDir = false;
	if (boost::filesystem::is_directory(monDirectory))
		foundMonDir = true;
	if (!foundMonDir) {
		std::cout << "<mon> DIR NOT FOUND! CREATING!!!" << std::endl;
		boost::filesystem::create_directories(monDirectory);
	}

}

uint32_t FedRawDataInputSource::getEventSizeFromBuffer() {
	if (buffer_left < sizeof(uint32_t))
		checkIfBuffered();
	uint32_t retval = *(uint32_t*) (data_buffer + buffer_cursor);
	buffer_left -= sizeof(uint32_t);
	buffer_cursor += sizeof(uint32_t);
	return retval;
}

uint32_t FedRawDataInputSource::getPaddingSizeFromBuffer() {
	if (buffer_left < sizeof(uint32_t))
		checkIfBuffered();
	uint32_t retval = *(uint32_t*) (data_buffer + buffer_cursor);
	buffer_left -= sizeof(uint32_t);
	buffer_cursor += sizeof(uint32_t);
	return retval;
}

uint32_t FedRawDataInputSource::fillFedSizesFromBuffer(uint32_t *fedSizes) {
	if (buffer_left < sizeof(uint32_t) * 1024)
		checkIfBuffered();
	memcpy((void*) fedSizes, (void*) (data_buffer + buffer_cursor),
			sizeof(uint32_t) * 1024);
	uint32_t eventSize = 0;
	if (formatVersion_ < 3) {
		for (unsigned int i = 0; i < 1024; i++)
			eventSize += fedSizes[i];
	}
	buffer_left -= sizeof(uint32_t) * 1024;
	buffer_cursor += sizeof(uint32_t) * 1024;
	return eventSize;
}

bool FedRawDataInputSource::getEventHeaderFromBuffer(
		FRDEventHeader_V2 *eventHeader) {
	if (buffer_left < sizeof(uint32_t) * 4)
		if (!checkIfBuffered())
			return false;
	memcpy((void*) eventHeader, (void*) (data_buffer + buffer_cursor),
			sizeof(uint32_t) * 4);
	assert(eventHeader->version_ > 1);
	formatVersion_ = eventHeader->version_;
	buffer_left -= sizeof(uint32_t) * 4;
	buffer_cursor += sizeof(uint32_t) * 4;
	return true;
}

bool FedRawDataInputSource::checkIfBuffered() {
	if (eofReached() && !openNextFile())
		return false;
	if (buffer_left == 0) {
		uint32_t chunksize = 1024 * 1024 * eventChunkSize_;
		buffer_left = fread((void*) data_buffer, sizeof(unsigned char),
				chunksize, fileStream_);
	} else {
		uint32_t chunksize = 1024 * 1024 * eventChunkSize_ - buffer_left;
		memcpy((void*) data_buffer, data_buffer + buffer_cursor, buffer_left);
		buffer_left += fread((void*) (data_buffer + buffer_left),
				sizeof(unsigned char), chunksize, fileStream_);
	}
	buffer_cursor = 0;
	return (buffer_left != 0);
}

// define this class as an input source
DEFINE_FWK_INPUT_SOURCE( FedRawDataInputSource);

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
