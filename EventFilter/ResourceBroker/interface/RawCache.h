////////////////////////////////////////////////////////////////////////////////
//
// RawCache.h
// -------
//
// Backup for RawMsgBuf messages containing raw FED data.
//
//  Created on: Nov 16, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////

#ifndef RAWCACHE_H_
#define RAWCACHE_H_

#include "EventFilter/ResourceBroker/interface/RawMsgBuf.h"

namespace evf {

/**
 * Contains an array of RawMsgBuffer objects, as a backup for raw messages
 * being sent over the message queue. In case event processors crash, these
 * message buffers will be sent again.
 *
 * $Author: aspataru $
 *
 */

class RawCache {
public:

	/**
	 * Returns the unique instance of this object.
	 */
	static RawCache* getInstance();

	/**
	 * Initialize the cache with the given number of message buffers,
	 * and the given raw cell size for each one.
	 */
	void initialise(unsigned int nMsgs, unsigned int cellSize);
	virtual ~RawCache();

	/**
	 * Returns a pointer to a message buffer ready to be written
	 * with raw data.
	 */
	RawMsgBuf* getMsgToWrite();

	/**
	 * Releases a message buffer in the cache.
	 */
	void releaseMsg(unsigned int fuResourceId);

	/**
	 * Prints the current slot usage in the cache.
	 */
	void printUsage() const;

private:
	static RawCache* instance_;
	RawCache();
	int getFreeSlot() const;
	void setFree(RawMsgBuf* rawCell);

private:
	bool initialised_;
	RawMsgBuf** msgs_;
	unsigned int nMsgs_, cellSize_;
	bool* slotUsage_;
};

}

#endif /* RAWCACHE_H_ */
