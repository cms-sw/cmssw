/*
 * DataPoint.h
 *
 *  Created on: Sep 24, 2012
 *      Author: aspataru
 */

#ifndef DATAPOINT_H_
#define DATAPOINT_H_

#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/JsonSerializable.h"

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <stdint.h>
#include <assert.h>

//synchronization level between streams/threads for atomic updates
//#define ATOMIC_LEVEL 2 //assume postEvent and postLumi are not synchronized (each invocation can run in different thread)
//#define ATOMIC_LEVEL 1 //assume postEvent can run in different threads but endLumi still sees all memory writes properly
#define ATOMIC_LEVEL 0 //assume data is synchronized

namespace jsoncollector {

#if ATOMIC_LEVEL>0
typedef std::atomic<unsigned int> AtomicMonUInt;
#else
typedef unsigned int AtomicMonUInt;
#endif

typedef std::map<unsigned int,JsonMonPtr> MonPtrMap;

class DataPoint: public JsonSerializable {

public:

	DataPoint() { }

	DataPoint(std::string const& source, std::string const& definition, bool fast=false) :
                 source_(source), definition_(definition), isFastOnly_(fast) { }

	~DataPoint();

	/**
	 * JSON serialization procedure for this class
	 */

	virtual void serialize(Json::Value& root) const;

	/**
	 * JSON deserialization procedure for this class
	 */
	virtual void deserialize(Json::Value& root);

	std::vector<std::string>& getData() {return data_;}

	/**
	 * Functions specific to new monitoring implementation
	 */

	void serialize(Json::Value& root, bool rootInit, std::string const& input) const;

	//take new update for lumi
	void snap(unsigned int lumi);
	void snapGlobal(unsigned int lumi);
	void snapStreamAtomic(unsigned int streamID, unsigned int lumi);

	//set to track a variable
	void trackMonitorable(JsonMonitorable *monitorable,bool NAifZeroUpdates);

	//set to track a vector of variables
	void trackVectorUInt(std::string const& name, std::vector<unsigned int> *monvec, bool NAifZeroUpdates);

	//set to track a vector of atomic variables with guaranteed collection
	void trackVectorUIntAtomic(std::string const& name, std::vector<AtomicMonUInt*> *monvec, bool NAifZeroUpdates);

	//variable not found by the service, but want to output something to JSON
	void trackDummy(std::string const& name, bool setNAifZeroUpdates)
	{
		name_ = name;
		isDummy_=true;
		NAifZeroUpdates_=true;
	}

	void makeStreamLumiMap(unsigned int size);

	//sets which operation is going to be performed on data (specified by JSON def)
        void setOperation(OperationType op) {
    		opType_=op;
	}

	//only used if per-stream DP (should use non-atomic vector here)
	void setStreamLumiPtr(std::vector<unsigned int> *streamLumiPtr) {
	  streamLumisPtr_=streamLumiPtr;
	}

	//fastpath (not implemented now)
	std::string fastOutCSV();

	//pointed object should be available until discard
	JsonMonitorable * mergeAndRetrieveValue(unsigned int forLumi);

	//get everything collected prepared for output
	void mergeAndSerialize(Json::Value& jsonRoot, unsigned int lumi, bool initJsonValue);

	//cleanup lumi
	void discardCollected(unsigned int forLumi);

	//this parameter sets location where we can find hwo many bins are needed for histogram 
	void setNBins(unsigned int *nBins) {nBinsPtr_ = nBins;}

	std::string const& getName() {return name_;}

	// JSON field names
	static const std::string SOURCE;
	static const std::string DEFINITION;
	static const std::string DATA;

protected:
	//for simple usage
	std::string source_;
	std::string definition_;
	std::vector<std::string> data_;

	std::vector<MonPtrMap> streamDataMaps_;
	MonPtrMap globalDataMap_;
	void *tracked_ = nullptr;

        //stream lumi block position
	std::vector<unsigned int> *streamLumisPtr_ = nullptr;

	bool isStream_ = false;
	bool isAtomic_ = false;
	bool isDummy_ = false;
	bool NAifZeroUpdates_ = false;
        bool isFastOnly_;
	
	MonType monType_;
	OperationType opType_;
	std::string name_;

	//helpers
	uint32_t *buf_ = nullptr;
	unsigned int bufLen_ =0;

	unsigned int * nBinsPtr_ = nullptr;
	int cacheI_;//int cache
	bool isCached_=0;

	unsigned int fastIndex_ = 0;


};
}

#endif /* DATAPOINT_H_ */
