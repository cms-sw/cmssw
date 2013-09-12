/*
 * DataPointMonitor.h
 *
 *  Created on: Nov 19, 2012
 *      Author: aspataru
 */

#ifndef DATAPOINTMONITOR_H_
#define DATAPOINTMONITOR_H_

#include <vector>
#include <string>
#include "JsonMonitorable.h"
#include "DataPointDefinition.h"
#include "LegendItem.h"
#include "DataPoint.h"

using std::vector;
using std::string;

namespace jsoncollector {

class DataPointMonitor {

public:
	DataPointMonitor(vector<JsonMonitorable*> monitorableVariables,
			string defPath);
	virtual ~DataPointMonitor();

	void snap(DataPoint& outputDataPoint);

private:

	bool isStringMonitorable(string key) const;
	JsonMonitorable* getVarForName(string name) const;

	vector<JsonMonitorable*> monitorableVars_;
	vector<string> toBeMonitored_;
	vector<JsonMonitorable*> monitoredVars_;
	DataPointDefinition dpd_;
	string defPath_;

};
}

#endif /* DATAPOINTMONITOR_H_ */
