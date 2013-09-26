/*
 * FastMonitor.cc
 *
 *  Created on: Nov 27, 2012
 *      Author: aspataru
 */

#include "../interface/FastMonitor.h"
#include "../interface/ObjectMerger.h"
#include "../interface/JSONSerializer.h"
#include "../interface/FileIO.h"
#include "../interface/Utils.h"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace jsoncollector;
using std::string;
using std::vector;
using std::ofstream;
using std::fstream;
using std::endl;

FastMonitor::FastMonitor(vector<JsonMonitorable*> monitorableVariables,
		string defPath) :
	snappedOnce_(false), monitorableVars_(monitorableVariables),
			defPath_(defPath) {

	ObjectMerger::getDataPointDefinitionFor(defPath_, dpd_);
	if (dpd_.isPopulated()) {
		for (unsigned int i = 0; i < dpd_.getLegend().size(); i++) {
			string toBeMonitored = dpd_.getLegend()[i].getName();
			monitoredVars_.push_back(getVarForName(toBeMonitored));
		}
	}
}

FastMonitor::~FastMonitor() {
}

void FastMonitor::snap(bool outputCSVFile, string path) {
	std::stringstream ss;
	for (unsigned int i = 0; i < monitoredVars_.size(); i++) {
		if (i == monitoredVars_.size() - 1) {
			ss << monitoredVars_[i]->toString();
			break;
		}
		ss << monitoredVars_[i]->toString() << ",";
	}

	if (outputCSVFile) {
		ofstream outputFile;
		outputFile.open(path.c_str(), fstream::out | fstream::trunc);
		outputFile << defPath_ << endl;
		outputFile << ss.str();
		outputFile << endl;
		outputFile.close();
	}

	string inputStringCSV = ss.str();
	accumulatedCSV_.push_back(inputStringCSV);
}

void FastMonitor::outputFullHistoDataPoint(string path) {
	if (accumulatedCSV_.size() > 0) {

		vector<DataPoint*> dpToMerge;

		for (unsigned int i = 0; i < accumulatedCSV_.size(); i++) {
			string currentCSV = accumulatedCSV_[i];
			DataPoint* currentDP = ObjectMerger::csvToJson(currentCSV, &dpd_,
					defPath_);
			string hpid;
			Utils::getHostAndPID(hpid);
			currentDP->setSource(hpid);
			dpToMerge.push_back(currentDP);
		}

		string outputJSONAsString;
		string msg;

		DataPoint* mergedDP = ObjectMerger::merge(dpToMerge, msg, true);
		mergedDP->setSource(dpToMerge[0]->getSource());

		for (unsigned int i = 0; i < dpToMerge.size(); i++)
			delete dpToMerge[i];

		JSONSerializer::serialize(mergedDP, outputJSONAsString);
		FileIO::writeStringToFile(path, outputJSONAsString);

		accumulatedCSV_.clear();
		snappedOnce_ = false;

	}
}

JsonMonitorable* FastMonitor::getVarForName(string name) const {
	for (unsigned int i = 0; i < monitorableVars_.size(); i++)
		if (name.compare(monitorableVars_[i]->getName()) == 0)
			return monitorableVars_[i];
	return 0;
}
