////////////////////////////////////////////////////////////////////////////////
//
// SharedResources.h
// -------
//
// Resources shared between FSM states.
//
// Created on: Sep 21, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/ResourceBroker/interface/SharedResources.h"

#include <signal.h>
#include <iostream>

using std::string;
using std::vector;
using std::cout;
using std::endl;

using namespace evf::rb_statemachine;

SharedResources::SharedResources(Logger log) :
			wlMonitoring_(0),
			asMonitoring_(0),
			wlWatching_(0),
			asWatching_(0),
			wlSendData_(0),
			asSendData_(0),
			wlSendDqm_(0),
			asSendDqm_(0),
			wlDiscard_(0),
			asDiscard_(0),
			gui_(0),
			commands_(CommandQueue()),
			log_(log),
			bu_(0),
			sm_(0),
			i2oPool_(0),
			ipcManager_(0),
			resourceStructure_(0),
			runNumber_(0),
			deltaT_(0.0),
			deltaN_(0),
			deltaSumOfSquares_(0),
			deltaSumOfSizes_(0),
			throughput_(0.0),
			rate_(0.0),
			average_(0.0),
			rms_(0.0),
			nbAllocatedEvents_(0),
			nbPendingRequests_(0),
			nbReceivedEvents_(0),
			nbSentEvents_(0),
			nbSentDqmEvents_(0),
			nbSentErrorEvents_(0),
			nbPendingSMDiscards_(0),
			nbPendingSMDqmDiscards_(0),
			nbDiscardedEvents_(0),
			// UPDATED
			nbReceivedEol_(0),
			highestEolReceived_(0),
			nbEolPosted_(0),
			nbEolDiscarded_(0),
			nbLostEvents_(0),
			nbDataErrors_(0),
			nbCrcErrors_(0),
			nbTimeoutsWithEvent_(0),
			nbTimeoutsWithoutEvent_(0),
			dataErrorFlag_(0),
			segmentationMode_(false),
			useMessageQueueIPC_(false),
			nbClients_(0),
			clientPrcIds_(""),
			nbRawCells_(16),
			nbRecoCells_(8),
			nbDqmCells_(8),
			rawCellSize_(0x400000) // 4MB
			,
			recoCellSize_(0x800000) // 8MB
			,
			dqmCellSize_(0x800000) // 8MB
			// at least nbRawCells / 2 free resources to send allocate
			, freeResRequiredForAllocate_(-1), doDropEvents_(false),
			doFedIdCheck_(true), doCrcCheck_(1), doDumpEvents_(0),
			buClassName_("BU"), buInstance_(0), smClassName_("StorageManager"),
			smInstance_(0), resourceStructureTimeout_(200000), monSleepSec_(2),
			watchSleepSec_(10), timeOutSec_(30), processKillerEnabled_(true),
			useEvmBoard_(true), reasonForFailed_(""), nbAllocateSent_(0),
			nbTakeReceived_(0), nbDataDiscardReceived_(0),
			nbDqmDiscardReceived_(0), nbSentLast_(0), sumOfSquaresLast_(0),
			sumOfSizesLast_(0), frb_(0), shmInconsistent_(false),
			allowI2ODiscards_(true) {

	sem_init(&lock_, 0, 1);
	sem_init(&accessToResourceStructureLock_, 0, 1);

}

SharedResources::~SharedResources() {

}

//______________________________________________________________________________
void SharedResources::configureResources(xdaq::Application* app) {

	ipcManager_ = new IPCManager(useMessageQueueIPC_);

	ipcManager_->initialise(segmentationMode_.value_, nbRawCells_.value_,
			nbRecoCells_.value_, nbDqmCells_.value_, rawCellSize_.value_,
			recoCellSize_.value_, dqmCellSize_.value_,
			freeResRequiredForAllocate_, bu_, sm_, log_,
			resourceStructureTimeout_.value_, frb_, app);

	resourceStructure_ = ipcManager_->ipc();

	FUResource::doFedIdCheck(doFedIdCheck_);
	FUResource::useEvmBoard(useEvmBoard_);
	resourceStructure_->setDoCrcCheck(doCrcCheck_);
	resourceStructure_->setDoDumpEvents(doDumpEvents_);
	reset();
	shmInconsistent_ = false;

	// XXX shmInconsistent check
	if (resourceStructure_->nbResources() != nbRawCells_.value_
			|| resourceStructure_->nbFreeSlots() != nbRawCells_.value_)
		shmInconsistent_ = true;
}

//______________________________________________________________________________
void SharedResources::reset() {

	gui_->resetCounters();

	deltaT_ = 0.0;
	deltaN_ = 0;
	deltaSumOfSquares_ = 0.0;
	deltaSumOfSizes_ = 0;

	throughput_ = 0.0;
	rate_ = 0.0;
	average_ = 0.0;
	rms_ = 0.0;

	nbSentLast_ = 0;
	sumOfSquaresLast_ = 0;
	sumOfSizesLast_ = 0;
}

//______________________________________________________________________________
void SharedResources::cancelAllWorkloops() {
	if (wlSendData_) {
		wlSendData_->cancel();
		toolbox::task::getWorkLoopFactory()->removeWorkLoop("SendData",
				"waiting");
	}
	if (wlSendDqm_) {
		wlSendDqm_->cancel();
		toolbox::task::getWorkLoopFactory()->removeWorkLoop("SendDqm",
				"waiting");
	}
	if (wlDiscard_) {
		wlDiscard_->cancel();
		toolbox::task::getWorkLoopFactory()->removeWorkLoop("Discard",
				"waiting");
	}

	if (wlMonitoring_) {
		wlMonitoring_->cancel();
		toolbox::task::getWorkLoopFactory()->removeWorkLoop("Monitoring",
				"waiting");
	}
	if (wlWatching_) {
		wlWatching_->cancel();
		toolbox::task::getWorkLoopFactory()->removeWorkLoop("Watching",
				"waiting");
	}
}

//______________________________________________________________________________
void SharedResources::startMonitoringWorkLoop() throw (evf::Exception) {

	struct timezone timezone;
	gettimeofday(&monStartTime_, &timezone);

	try {
		wlMonitoring_ = toolbox::task::getWorkLoopFactory()->getWorkLoop(
				sourceId_ + "Monitoring", "waiting");
		if (!wlMonitoring_->isActive())
			wlMonitoring_->activate();
		asMonitoring_ = toolbox::task::bind(this, &SharedResources::monitoring,
				sourceId_ + "Monitoring");
		wlMonitoring_->submit(asMonitoring_);
	} catch (xcept::Exception& e) {
		string msg = "Failed to start workloop 'Monitoring'.";
		XCEPT_RETHROW(evf::Exception, msg, e);
	}
}

//______________________________________________________________________________
bool SharedResources::monitoring(toolbox::task::WorkLoop*) {

	unsigned int nbSent;
	uint64_t sumOfSquares;
	unsigned int sumOfSizes;
	uint64_t deltaSumOfSquares;

	lock();
	if (0 == resourceStructure_) {
		deltaT_.value_ = 0.0;
		deltaN_.value_ = 0;
		deltaSumOfSquares_.value_ = 0.0;
		deltaSumOfSizes_.value_ = 0;
		throughput_ = 0.0;
		rate_ = 0.0;
		average_ = 0.0;
		rms_ = 0.0;
		unlock();
		return false;
	} else {
		nbSent = resourceStructure_->nbSent();
		sumOfSquares = resourceStructure_->sumOfSquares();
		sumOfSizes = resourceStructure_->sumOfSizes();
	}
	unlock();

	struct timeval monEndTime;
	struct timezone timezone;

	gettimeofday(&monEndTime, &timezone);

	xdata::getInfoSpaceFactory()->lock();
	gui_->monInfoSpace()->lock();

	deltaT_.value_ = deltaT(&monStartTime_, &monEndTime);
	monStartTime_ = monEndTime;

	deltaN_.value_ = nbSent - nbSentLast_;
	nbSentLast_ = nbSent;

	deltaSumOfSquares = sumOfSquares - sumOfSquaresLast_;
	deltaSumOfSquares_.value_ = (double) deltaSumOfSquares;
	sumOfSquaresLast_ = sumOfSquares;

	deltaSumOfSizes_.value_ = sumOfSizes - sumOfSizesLast_;
	sumOfSizesLast_ = sumOfSizes;

	if (deltaT_.value_ != 0) {
		throughput_ = deltaSumOfSizes_.value_ / deltaT_.value_;
		rate_ = deltaN_.value_ / deltaT_.value_;
	} else {
		throughput_ = 0.0;
		rate_ = 0.0;
	}

	double meanOfSquares, mean, squareOfMean, variance;

	if (deltaN_.value_ != 0) {
		meanOfSquares = deltaSumOfSquares_.value_ / ((double) (deltaN_.value_));
		mean = ((double) (deltaSumOfSizes_.value_))
				/ ((double) (deltaN_.value_));
		squareOfMean = mean * mean;
		variance = meanOfSquares - squareOfMean;
		if (variance < 0.0)
			variance = 0.0;

		average_ = deltaSumOfSizes_.value_ / deltaN_.value_;
		rms_ = std::sqrt(variance);
	} else {
		average_ = 0.0;
		rms_ = 0.0;
	}

	gui_->monInfoSpace()->unlock();
	xdata::getInfoSpaceFactory()->unlock();

	::sleep(monSleepSec_.value_);

	return true;
}

//______________________________________________________________________________
void SharedResources::startWatchingWorkLoop() throw (evf::Exception) {
	try {
		wlWatching_ = toolbox::task::getWorkLoopFactory()->getWorkLoop(
				sourceId_ + "Watching", "waiting");
		if (!wlWatching_->isActive())
			wlWatching_->activate();
		asWatching_ = toolbox::task::bind(this, &SharedResources::watching,
				sourceId_ + "Watching");
		wlWatching_->submit(asWatching_);
	} catch (xcept::Exception& e) {
		string msg = "Failed to start workloop 'Watching'.";
		XCEPT_RETHROW(evf::Exception, msg, e);
	}
}

//______________________________________________________________________________
bool SharedResources::watching(toolbox::task::WorkLoop*) {
	lock();
	if (0 == resourceStructure_) {
		unlock();
		return false;
	}

	vector<pid_t> evt_prcids;
	vector<UInt_t> evt_numbers;
	vector<time_t> evt_tstamps;
	try {
		evt_prcids = resourceStructure_->cellPrcIds();
		evt_numbers = resourceStructure_->cellEvtNumbers();
		evt_tstamps = resourceStructure_->cellTimeStamps();
	} catch (evf::Exception& e) {
		goToFailedState(e);
	}

	time_t tcurr = time(0);
	for (UInt_t i = 0; i < evt_tstamps.size(); i++) {
		pid_t pid = evt_prcids[i];
		UInt_t evt = evt_numbers[i];
		time_t tstamp = evt_tstamps[i];
		if (tstamp == 0)
			continue;
		double tdiff = difftime(tcurr, tstamp);
		if (tdiff > timeOutSec_) {
			if (processKillerEnabled_) {
				// UPDATED
				kill(pid, 9);
				nbTimeoutsWithEvent_++;
			}
			LOG4CPLUS_ERROR(
					log_,
					"evt " << evt << " under processing for more than "
							<< timeOutSec_ << "sec for process " << pid);

		}
	}

	vector<pid_t> prcids;
	try {   
		auto lk = resourceStructure_->lockCrashHandler();
		prcids = resourceStructure_->clientPrcIds();
		for (UInt_t i = 0; i < prcids.size(); i++) {
			pid_t pid = prcids[i];
			int status = kill(pid, 0);
			if (status != 0) {
				LOG4CPLUS_ERROR(
						log_,
						"EP prc " << pid
								<< " died, send to error stream if processing.");
				if (!resourceStructure_->handleCrashedEP(runNumber_, pid))
					nbTimeoutsWithoutEvent_++;
			}
		}

	} catch (evf::Exception& e) {
		goToFailedState(e);
	}

	try {
		if ((resourceStructure_->nbResources() != nbRawCells_.value_)
				&& !shmInconsistent_) {
			std::ostringstream ost;
			ost << "Watchdog spotted inconsistency in ResourceTable - nbRaw="
					<< nbRawCells_.value_ << " but nbResources="
					<< resourceStructure_->nbResources() << " and nbFreeSlots="
					<< resourceStructure_->nbFreeSlots();
			XCEPT_DECLARE(evf::Exception, sentinelException, ost.str());
			fsm_->getApp()->notifyQualified("error", sentinelException);

			// XXX shmInconsistent
			shmInconsistent_ = true;
		}
	} catch (evf::Exception& e) {
		goToFailedState(e);
	}

	unlock();

	::sleep(watchSleepSec_.value_);
	return true;
}

//______________________________________________________________________________
double SharedResources::deltaT(const struct timeval *start,
		const struct timeval *end) {
	unsigned int sec;
	unsigned int usec;

	sec = end->tv_sec - start->tv_sec;

	if (end->tv_usec > start->tv_usec) {
		usec = end->tv_usec - start->tv_usec;
	} else {
		sec--;
		usec = 1000000 - ((unsigned int) (start->tv_usec - end->tv_usec));
	}

	return ((double) sec) + ((double) usec) / 1000000.0;
}

// sendData workloop STARTER
//______________________________________________________________________________
void SharedResources::startSendDataWorkLoop() throw (evf::Exception) {
	try {
		LOG4CPLUS_INFO(log_, "Start 'send data' workloop.");
		wlSendData_ = toolbox::task::getWorkLoopFactory()->getWorkLoop(
				"SendData", "waiting");
		if (!wlSendData_->isActive())
			wlSendData_->activate();
		asSendData_ = toolbox::task::bind(this, &SharedResources::sendData,
				"SendData");
		wlSendData_->submit(asSendData_);

	} catch (xcept::Exception& e) {
		string msg = "Failed to start workloop 'SendData'.";
		XCEPT_RETHROW(evf::Exception, msg, e);
	}
}

//  sendData workloop DISPATCHING SIGNATURE
bool SharedResources::sendData(toolbox::task::WorkLoop*) {
	int currentStateID = -1;
	bool reschedule = true;

	fsm_->transitionReadLock();
	currentStateID = fsm_->getCurrentState().stateID();
	fsm_->transitionUnlock();

	try {
		switch (currentStateID) {
		case rb_statemachine::RUNNING:
			reschedule = resourceStructure_->sendData();
			break;
		case rb_statemachine::STOPPING:
			reschedule = resourceStructure_->sendData();
			break;
		case rb_statemachine::HALTING:
			reschedule = resourceStructure_->sendDataWhileHalting();
			break;
		case rb_statemachine::FAILED:
			// workloop must be exited in this state
			return false;
		default:
			cout << "RBStateMachine: current state: " << currentStateID
					<< " does not support action: >>sendData<<" << endl;
			::usleep(50000);
			reschedule = true;
		}
	} catch (evf::Exception& e) {
		goToFailedState(e);
	}

	return reschedule;
}

// sendDqm workloop STARTER
//______________________________________________________________________________
void SharedResources::startSendDqmWorkLoop() throw (evf::Exception) {
	try {
		LOG4CPLUS_INFO(log_, "Start 'send dqm' workloop.");
		wlSendDqm_ = toolbox::task::getWorkLoopFactory()->getWorkLoop(
				"SendDqm", "waiting");
		if (!wlSendDqm_->isActive())
			wlSendDqm_->activate();
		asSendDqm_ = toolbox::task::bind(this, &SharedResources::sendDqm,
				"SendDqm");
		wlSendDqm_->submit(asSendDqm_);

	} catch (xcept::Exception& e) {
		string msg = "Failed to start workloop 'SendDqm'.";
		XCEPT_RETHROW(evf::Exception, msg, e);
	}
}

//  sendDqm workloop DISPATCHING SIGNATURE
bool SharedResources::sendDqm(toolbox::task::WorkLoop*) {
	int currentStateID = -1;
	bool reschedule = true;

	fsm_->transitionReadLock();
	currentStateID = fsm_->getCurrentState().stateID();
	fsm_->transitionUnlock();

	try {
		switch (currentStateID) {
		case rb_statemachine::RUNNING:
			reschedule = resourceStructure_->sendDqm();
			break;
		case rb_statemachine::STOPPING:
			reschedule = resourceStructure_->sendDqm();
			break;
		case rb_statemachine::HALTING:
			reschedule = resourceStructure_->sendDqmWhileHalting();
			break;
		case rb_statemachine::FAILED:
			// workloop must be exited in this state
			return false;
		default:
			cout << "RBStateMachine: current state: " << currentStateID
					<< " does not support action: >>sendDqm<<" << endl;
			::usleep(50000);
			reschedule = true;
		}
	} catch (evf::Exception& e) {
		goToFailedState(e);
	}

	return reschedule;
}

// discard workloop STARTER
//______________________________________________________________________________
void SharedResources::startDiscardWorkLoop() throw (evf::Exception) {
	try {
		LOG4CPLUS_INFO(log_, "Start 'discard' workloop.");
		wlDiscard_ = toolbox::task::getWorkLoopFactory()->getWorkLoop(
				"Discard", "waiting");
		if (!wlDiscard_->isActive())
			wlDiscard_->activate();
		asDiscard_ = toolbox::task::bind(this, &SharedResources::discard,
				"Discard");
		wlDiscard_->submit(asDiscard_);
		resourceStructure_->setActive(true);

	} catch (xcept::Exception& e) {
		string msg = "Failed to start workloop 'Discard'.";
		XCEPT_RETHROW(evf::Exception, msg, e);
	}
	resourceStructure_->setReadyToShutDown(false);
}

//  discard workloop DISPATCHING SIGNATURE
bool SharedResources::discard(toolbox::task::WorkLoop*) {
	int currentStateID = -1;
	bool reschedule = true;

	fsm_->transitionReadLock();
	currentStateID = fsm_->getCurrentState().stateID();
	fsm_->transitionUnlock();
	try {
		switch (currentStateID) {
		case rb_statemachine::RUNNING:
			reschedule = resourceStructure_->discard();
			break;
		case rb_statemachine::STOPPING:
			// XXX: communication with BU after stop!
			reschedule = resourceStructure_->discardWhileHalting(true);
			break;
		case rb_statemachine::HALTING:
			// XXX: no more communication with BU after halt!
			reschedule = resourceStructure_->discardWhileHalting(false);
			break;
		case rb_statemachine::FAILED:
			// workloop must be exited in this state
			return false;
		default:
			cout << "RBStateMachine: current state: " << currentStateID
					<< " does not support action: >>discard<<" << endl;
			::usleep(50000);
			reschedule = true;
		}
	} catch (evf::Exception& e) {
		goToFailedState(e);
	}

	return reschedule;
}

//______________________________________________________________________________
void SharedResources::printWorkLoopStatus() {
	cout << "Workloop status===============" << endl;
	cout << "==============================" << endl;
	if (wlSendData_ != 0)
		cout << "SendData -> " << wlSendData_->isActive() << endl;
	if (wlSendDqm_ != 0)
		cout << "SendDqm  -> " << wlSendDqm_->isActive() << endl;
	if (wlDiscard_ != 0)
		cout << "Discard  -> " << wlDiscard_->isActive() << endl;
	//cout << "Workloops Active  -> " << isActive_ << endl;
}

//______________________________________________________________________________
void SharedResources::goToFailedState(evf::Exception& exception) {
	reasonForFailed_ = exception.what();
	LOG4CPLUS_FATAL(log_,
			"Moving to FAILED state! Reason: " << exception.what());
	EventPtr fail(new Fail());
	commands_.enqEvent(fail);
}
