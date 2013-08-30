/*
 * FastMonitor.h
 *
 *  Created on: Nov 27, 2012
 *      Author: aspataru
 */

#ifndef FASTMONITOR_H_
#define FASTMONITOR_H_

#include "JsonMonitorable.h"
#include "DataPointDefinition.h"
#include "DataPoint.h"

namespace jsoncollector {

class FastMonitor {

public:
	FastMonitor(std::vector<JsonMonitorable*> monitorableVariables,
			std::string defPath);
	virtual ~FastMonitor();

	// updates internal HistoDataPoint and prints one-line CSV if param == true
	void snap(bool outputCSVFile, std::string path);

	// outputs the contents of the internal histoDataPoint, at the end of lumi
	void outputFullHistoDataPoint(std::string path);

private:
	JsonMonitorable* getVarForName(string name) const;

	bool snappedOnce_;
	DataPointDefinition dpd_;
	std::vector<JsonMonitorable*> monitorableVars_;
	std::vector<JsonMonitorable*> monitoredVars_;
	std::vector<string> accumulatedCSV_;
	std::string defPath_;
};

}

#endif /* FASTMONITOR_H_ */
