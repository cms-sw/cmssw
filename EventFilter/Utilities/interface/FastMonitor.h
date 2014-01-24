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

class tbb::concurrent_queue;

namespace jsoncollector {

class FastMonitor {

public:

	FastMonitor(const std::vector<JsonMonitorable*>& monitorableVariables, 
			const std::vector<JsonMonConfig>& varsConfig,
			std::string const& defPath, unsigned int numberOfStreams = 1, bool startImmediately = true);

	virtual ~FastMonitor();

	void snap(unsigned int streamID = 0);

	// updates internal HistoDataPoint and prints one-line CSV if bool param is true
	void snap(bool outputCSVFile, std::string const& path, unsigned int streamID = 0);


	// outputs everything for the given stream he contents of the internal histoDataPoint, at the end of lumi
	void outputFullHistoDataPoint(std::string const& path, unsigned int streamID = 0, bool clear = true);

	void getHostAndPID(std::string& sHPid);

	//TODO:call for merging across stream functions

private:

	std::string defPath_;
	unsigned int nStreams_;
	DataPointDefinition dpd_;
	JsonMonConfig monConfig_;
//	std::vector<JsonMonitorable*> monitorableVars_;
	std::vector<std::vector<JsonMonitorable*>> monitoredVars_; //per stream
	std::vector<tbb::concurrent_queue<DataPoint*>> accDpQueues_;//per stream tbb queues
	std::string sourceInfo_;
};

}

#endif /* FASTMONITOR_H_ */
