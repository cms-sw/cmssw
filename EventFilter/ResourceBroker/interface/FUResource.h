////////////////////////////////////////////////////////////////////////////////
//
// FUResource
// ----------
//
//            12/10/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
//            20/01/2012 Andrei Spataru <aspataru@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#ifndef FURESOURCE_H
#define FURESOURCE_H 1

#include "EventFilter/ShmBuffer/interface/FUShmRawCell.h"
#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/Utilities/interface/Exception.h"

#include "log4cplus/logger.h"

#include <vector>

#define FED_HCTRLID    0x50000000
#define FED_TCTRLID    0xa0000000
#define REAL_SOID_MASK 0x0003FF00
#define FED_RBIT_MASK  0x0000C004

//#define DEBUG_FURESOURCE_H

namespace xdaq {
class Application;
}

namespace evf {

class EvffedFillerRB;

/**
 * Object representing a managed resource.
 *
 * $Author: aspataru $
 *
 */

class FUResource {
public:
	//
	// construction/destruction
	//
	FUResource(UInt_t fuResourceId, log4cplus::Logger, EvffedFillerRB *,
			xdaq::Application *);
	virtual ~FUResource();

	//
	// member functions
	//

	/**
	 * Associate a ShmRawCell to the resource
	 */

	void allocate(FUShmRawCell* shmCell);
	/**
	 * Process resource
	 */
	void process(MemRef_t* bufRef);

	/**
	 * Release resource
	 */
	void release(bool detachResource);

	/**
	 * Append I2O block to super fragment
	 */
	void appendBlockToSuperFrag(MemRef_t* bufRef);

	/**
	 * Remove last appended I2O block from super fragment
	 */
	void removeLastAppendedBlockFromSuperFrag();

	/**
	 * Returns the size of the current super fragment
	 */
	void superFragSize() throw (evf::Exception);

	/**
	 * Fills super fragment payload with I2O blocks
	 */
	void fillSuperFragPayload() throw (evf::Exception);

	/**
	 * Releases the current super fragment and memory used by it
	 */
	void releaseSuperFrag();

	static
	void doFedIdCheck(bool doFedIdCheck) {
		doFedIdCheck_ = doFedIdCheck;
	}
	static
	void useEvmBoard(bool useEvmBoard) {
		useEvmBoard_ = useEvmBoard;
	}
	void doCrcCheck(bool doCrcCheck) {
		doCrcCheck_ = doCrcCheck;
	}
	bool crcBeingChecked() {
		return doCrcCheck_;
	}
	bool fatalError() const {
		return fatalError_;
	}
	bool isAllocated() const {
		return 0 != shmCell_;
	}
	bool isComplete() const;

	UInt_t fuResourceId() const {
		return fuResourceId_;
	}
	UInt_t buResourceId() const {
		return buResourceId_;
	}
	UInt_t evtNumber() const {
		return evtNumber_;
	}
	UInt_t nbSent() const {
		return nbSent_;
	}

	/**
	 * Increment the number of sent resources
	 */
	void incNbSent() {
		nbSent_++;
	}

	UInt_t nbErrors(bool reset = true);
	UInt_t nbCrcErrors(bool reset = true);
	UInt_t nbBytes(bool reset = true);

	/**
	 * Returns a pointer to the Shm Raw cell associated to this resource
	 */
	evf::FUShmRawCell* shmCell() {
		return shmCell_;
	}
	void scheduleCRCError() {
		nextEventWillHaveCRCError_ = true;
	}

private:
	//
	// member data
	//
	log4cplus::Logger log_;

	static
	bool doFedIdCheck_;
	static
	bool useEvmBoard_;
	bool doCrcCheck_;
	bool fatalError_;

	UInt_t fuResourceId_;
	UInt_t buResourceId_;
	UInt_t evtNumber_;

	MemRef_t* superFragHead_;
	MemRef_t* superFragTail_;

	UInt_t eventPayloadSize_;
	UInt_t nFedMax_;
	UInt_t nSuperFragMax_;

	UInt_t iBlock_;
	UInt_t nBlock_;
	UInt_t iSuperFrag_;
	UInt_t nSuperFrag_;

	UInt_t nbSent_;

	UInt_t nbErrors_;
	UInt_t nbCrcErrors_;
	UInt_t nbBytes_;

	UInt_t fedSize_[1024];
	UInt_t superFragSize_;
	UInt_t eventSize_;

	evf::FUShmRawCell* shmCell_;
	EvffedFillerRB *frb_;

	xdaq::Application *app_;

	bool nextEventWillHaveCRCError_;

	static unsigned int gtpDaqId_;
	static unsigned int gtpEvmId_;
	static unsigned int gtpeId_;

	friend class ResourceChecker;

};

//
// typedefs
//
typedef std::vector<FUResource*> FUResourceVec_t;

} // namespace evf

//
// implementation of inline functions
//

/**
 * Checks if all I2O blocks have been received for this resource
 */
inline
bool evf::FUResource::isComplete() const {

#ifdef DEBUG_FURESOURCE_H
	cout << "------------------------------------------------------"<< endl;
	cout << "nBlock " << nBlock_
	<< " iBlock " << iBlock_
	<< " nSuperFrag " << nSuperFrag_
	<< " iSuperFrag " << iSuperFrag_
	<< endl;
#endif
	return (nBlock_ && nSuperFrag_ && (iSuperFrag_ == nSuperFrag_) && (iBlock_
			== nBlock_));
}

#endif
