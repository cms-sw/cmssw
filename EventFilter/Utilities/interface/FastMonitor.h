/*
 * FastMonitor.h
 *
 *  Created on: Nov 27, 2012
 *      Author: aspataru
 */

#ifndef FASTMONITOR_H_
#define FASTMONITOR_H_

#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/DataPointDefinition.h"
#include "EventFilter/Utilities/interface/DataPoint.h"

#include <tbb/concurrent_queue.h>

namespace jsoncollector {

class FastMonitor {

public:

	FastMonitor(const std::vector<JsonMonitorable*>& monitorableVariables, 
			const std::vector<JsonMonConfig>& varsConfig,
			std::string const& defPath, unsigned int numberOfStreams = 1, bool startImmediately = true);

	virtual ~FastMonitor();

	void setNStreams(unsigned int nStreams);

	//register global monitorable
	void registerGlobalMonitorable(JsonMonitorable *newMonitorable, bool NAifZeroUpdates);

	//register per-stream monitores vector (unsigned int)
	void registerStreamMonitorableUIntVec(std::string &name, std::vector<unsigned int> *vec, bool NAifZeroUpdates ,unsigned int);

	//NOT implemented yet
	//void registerStreamMonitorableIntVec(std::string &name, std::vector<unsigned int>,true,0);
	//void registerStreamMonitorableDoubleVec(std::string &name, std::vector<unsigned int>,true,0);
	//void registerStreamMonitorableStringVec(std::string &name, std::vector<std::string>,true,0);

	//take vector used to track stream lumis and finish initialization
	void commit(std::vector<std::atomic<unsigned int>> *streamLumi);

	// fetches new snapshot and outputs one-line CSV if set
	void snap(bool outputCSVFile, std::string const& path, unsigned int forLumi);

	// merges and outputs everything collected for the given stream to JSON file
	void outputFullJSON(std::string const& path, unsigned int forLumi, bool addHostAndPID=true);

	//discard what was collected for a lumisection
        void discardCollected(unsigned int lumi);

	//this is added to the JSON file
	void getHostAndPID(std::string& sHPid);

private:

	std::string defPath_;
	unsigned int nStreams_;
	DataPointDefinition dpd_;

	JsonMonConfig monConfig_;

	std::vector<DataPointCollector> monitored_;//each var is one vector entry

	//	std::vector<JsonMonitorable*> monitorableVars_;
	std::vector<std::vector<JsonMonitorable*>> monitoredVars_; //per stream
	std::vector<tbb::concurrent_queue<DataPoint*>> accDpQueues_;//per stream tbb queues
	std::string sourceInfo_;
};

}

#endif /* FASTMONITOR_H_ */
