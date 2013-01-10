/*
 * A message buffer that holds
 *
 * 	Run
 * 	Lumisection
 * 	Evt number
 * 	[FED sizes]
 * 	[FED raw data segments]
 *
 *	author: aspataru@cern.ch
 */

#ifndef EVENTFILTER_UTILITIES_SIMPLE_MSG_BUF_H
#define EVENTFILTER_UTILITIES_SIMPLE_MSG_BUF_H

#include "EventFilter/Utilities/interface/queue_defs.h"
#include "EventFilter/Utilities/interface/MsgBuf.h"

namespace evf {

#define N_FEDS 1024

class SimpleMsgBuf : public MsgBuf{

public:
	SimpleMsgBuf();
	SimpleMsgBuf(unsigned int size, unsigned int type);
	SimpleMsgBuf(const SimpleMsgBuf &b);
	virtual ~SimpleMsgBuf();

	/*
	 * event buffer fields access
	 */
	void setRLE(unsigned int run, unsigned int lumi, unsigned int evt);
	void addFedSize(unsigned int fedSize);
	void addFedData(const unsigned char* fedRawData, unsigned int segmentSize);

	void getRun(unsigned int&) const;
	void getLumi(unsigned int&) const;
	void getEvt(unsigned int&) const;
	char* getFedSizeStart();
	char* getFedDataStart();
	char* getFedDataEnd();

	void reset() { setBufferPointers(); }

private:
	void setBufferPointers();

private:
	size_t usedSize_;
	friend class MasterQueue;

	/*
	 * pointers to different buffer positions
	 */
	char* pRun_;
	char* pLumi_;
	char* pEvt_;
	char* pFedSizes_;
	char* pFedRawData_;

	// helper pointers
	char* pVarSize_;
	char* pVarData_;
};

}

#endif
