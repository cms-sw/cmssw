/*
 * DataPointMonitor.cc
 *
 *  Created on: Oct 29, 2012
 *      Author: aspataru
 */

#include "../interface/DataPointMonitor.h"
#include "../interface/ObjectMerger.h"

using namespace jsoncollector;

DataPointMonitor::DataPointMonitor(
		vector<JsonMonitorable*> monitorableVariables, string defPath) :
	monitorableVars_(monitorableVariables) {
	defPath_ = defPath;
	ObjectMerger::getDataPointDefinitionFor(defPath_, dpd_);
	if (dpd_.isPopulated()) {
		for (unsigned int i = 0; i < dpd_.getLegend().size(); i++) {
			string toBeMonitored = dpd_.getLegend()[i].getName();
			monitoredVars_.push_back(getVarForName(toBeMonitored));
		}
	}
}

DataPointMonitor::~DataPointMonitor() {
}

void DataPointMonitor::snap(DataPoint& outputDataPoint) {
	outputDataPoint.resetData();
	for (unsigned int i = 0; i < monitoredVars_.size(); i++)
		outputDataPoint.addToData(monitoredVars_[i]->toString());
	outputDataPoint.setSource("");
	outputDataPoint.setDefinition(defPath_);
}

bool DataPointMonitor::isStringMonitorable(string key) const {
	for (unsigned int i = 0; i < toBeMonitored_.size(); i++)
		if (key.compare(toBeMonitored_[i]) == 0)
			return true;
	return false;
}

JsonMonitorable* DataPointMonitor::getVarForName(string name) const {
	for (unsigned int i = 0; i < monitorableVars_.size(); i++)
		if (name.compare(monitorableVars_[i]->getName()) == 0)
			return monitorableVars_[i];
	return 0;
}

