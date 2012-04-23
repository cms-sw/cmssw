////////////////////////////////////////////////////////////////////////////////
//
// RawCache.cc
// -------
//
// Backup for RawMsgBuf messages containing raw FED data.
//
//  Created on: Nov 16, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/ResourceBroker/interface/RawCache.h"
#include "EventFilter/ResourceBroker/interface/msq_constants.h"
#include <iostream>
#include <cstdlib>
#include <unistd.h>

using namespace evf;
using std::cout;
using std::endl;

RawCache* RawCache::instance_ = 0;
//______________________________________________________________________________

RawCache* RawCache::getInstance() {
	if (instance_ == 0) {
		instance_ = new RawCache();
	}
	return instance_;
}

void RawCache::initialise(unsigned int nMsgs, unsigned int cellSize) {
	if (!initialised_) {
		// set the desired size of the cache
		nMsgs_ = nMsgs;
		//set the cell size of all cells
		cellSize_ = cellSize;

		msgs_ = new RawMsgBuf*[nMsgs_];
		// initialise array with flags regarding cache slot usage
		slotUsage_ = new bool[nMsgs_];

		// set all slots free and initialise
		for (unsigned int i = 0; i < nMsgs_; i++) {
			// not contiguous memory!!
			msgs_[i] = new RawMsgBuf(MAX_MSG_SIZE, RAW_MESSAGE_TYPE);
			msgs_[i]->initialise(cellSize_);

			slotUsage_[i] = false;
		}
		initialised_ = true;
	}

}

RawCache::~RawCache() {
	delete[] slotUsage_;
	delete[] msgs_;
}

RawMsgBuf* RawCache::getMsgToWrite() {

	int freeSlot = getFreeSlot();

	while (freeSlot == -1) {
		cout << "NO FREE SLOTS IN CACHE!" << endl;
		//improve remove print usage
		printUsage();
		::sleep(1);
	}

	slotUsage_[freeSlot] = true;
	return msgs_[freeSlot];

}

void RawCache::releaseMsg(unsigned int fuResourceId) {
	RawMsgBuf* found = 0;

	for (unsigned int i = 0; i < nMsgs_; i++)
		if (slotUsage_[i])
			if (msgs_[i]->rawCell()->fuResourceId() == fuResourceId)
				found = msgs_[i];

	if (found != 0) {
		found->rawCell()->clear();
		setFree(found);
		//printUsage();
	} else
		cout << "RAW MSG BUF corresponding to fuResourceId = " << fuResourceId
				<< "not found in internal allocation list!" << endl;
}

RawCache::RawCache() :
	initialised_(false), nMsgs_(0) {

}

int RawCache::getFreeSlot() const {
	for (unsigned int i = 0; i < nMsgs_; i++)
		if (!slotUsage_[i])
			return i;
	return -1;
}

void RawCache::setFree(RawMsgBuf* rmb) {
	for (unsigned int i = 0; i < nMsgs_; i++)
		if (slotUsage_[i])
			if (msgs_[i] == rmb) {
				slotUsage_[i] = false;
				return;
			}
	cout << "ERROR: Raw Message Buffer to free at address: " << rmb
			<< " not found in internal allocation list!" << endl;
}

void RawCache::printUsage() const {
	cout << "Raw Cache usage: ";
	for (unsigned int i = 0; i < nMsgs_; i++)
		cout << slotUsage_[i] << " ";
	cout << endl;
}
