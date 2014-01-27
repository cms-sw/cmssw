/*
 * FastMonitor.cc
 *
 *  Created on: Nov 27, 2012
 *      Author: aspataru
 */

#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "EventFilter/Utilities/interface/Utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <assert.h>

using namespace jsoncollector;

FastMonitor::FastMonitor(std::string const& defPath) :
	defPath_(defPath),
{

	//get host and PID info
	getHostAndPID(sourceInfo_);

	//TODO:strict checking

	//first load definition file
	DataPointDefinition::getDataPointDefinitionFor(defPath_, dpd_);

	for (int i=0;i<dpd_.getNumberOfElements();i++) {
	  jsonPtrAtIndex_.push_back(nullptr);
	}

}

void FastMonitor::registerGlobalMonitorable(JsonMonitorable *newMonitorable, bool NAifZeroUpdates) {

	unsigned int jsonIndex;
	dpd.isValid(newMonitorable,jsonIndex);
	DataPoint *dp = new DataPoint(dpd_,defPath_,sourceInfo_,newMonitorable);//DP detects everything else
	dataPoints.push_back(dp);
}

	dpd_.populateMonConfig(monConfig_);
//	monConfig_.setSourceInfoPtr(&sourceInfo_);
//	monConfig_.setDefinitionPtr(&defPath_)

	//populate vectors that provide per-stream measurement
	for (unsigned int s=0;s<nStreams_;s++) {
		std::vector<JsonMonitorable*> sv;
		monitoredVars_.push_back(sv);
		for (unsigned int j=0;j<monConfig_.size();j++)
			monitoredVars_.at(s).push_back(nullptr);
	}

	//TODO:allow per-process (global) measurements..

	//populate it also with variable information from module/service
	for (auto i : monConfig_) {
		std::string & toBeMonitored = i.getName();
		assert(monitorableVariables.size()==varsConfig.size();
		for (unsigned int j=0;j< monitorableVars_.size();j++) 
		{
			if (i.getVarName()==varsConfig[j].getVarName())
			{
				for (unsigned int s=0;s<nStreams_;s++)
					monitoredVars_.at(s).at(i) = monitorableVars[j];
				i.setMonType(varsConfig[j].getMonType());
				i.setNAifZero(varsConfig[j].getNAifZero());
				i.setNBins(varsConfig[j].getNBins());
			}
		}
	}
	//TODO:correctness checks

	//create data point which also accumulates stuff
//	todo: create in begin lumi 
//
	for (unsigned int s=0;s<nStreams_;s++) {
		lumitracker_[i]=0;
		accDpQueues_.push_back(tbb::concurrent_queue<DataPoint> cq);
		if (startImmediately_) {//start measuring first set immediately
			accDpQueues_[s].push(new DataPoint(monitoredVars_[i], &monConfig_ , 50U, 100U,1));
			lumiTracker_[i]=1;
		}

	}
	//accumulatedDp_.reset(new DataPoint("", defPath_, monitoredVars_[i], &MonConfig , 50U, 100U));//some limits...
}

FastMonitor::~FastMonitor() {
}

void FastMonitor::snap(bool outputCSVFile, std::string const& path, unsigned int streamID, unsigned int lumi) {

	//check if luminosity increased and queue another
	if (lumiTracker_[streamID]<lumi) push(new DataPoint)
	if (outputCSVFile) {
		//old style (should only be done in case of outputCSVFile=true)
		std::stringstream ss;
		for (unsigned int i = 0; i < monitoredVars_.size(); i++) {
			if (i == monitoredVars_.size() - 1) {
				ss << monitoredVars_[i]->toString();
				break;
			}
			ss << monitoredVars_[i]->toString() << ",";
		}

		std::ofstream outputFile;
		outputFile.open(path.c_str(), std::fstream::out | std::fstream::trunc);
		outputFile << defPath_ << std::endl;
		outputFile << ss.str();
		outputFile << std::endl;
		outputFile.close();
	}

	accumulatedDp_->snap();
}

void FastMonitor::outputFullHistoDataPoint(std::string const& path, unsigned int streamID) {

	std::cout << "SNAP updates: " <<  accumulatedDp_->getUpdates() << std::endl;

	//this should really be removed
	//assert(accumulatedDp_.getUpdates()<100);

	//prepare some machine and process info
	std::string hpid;
	Utils::getHostAndPID(hpid);
	accumulatedDp_->setSource(hpid);

	std::string outputJSONAsString;
	//JSONSerializer::serialize(&*accumulatedDp_, outputJSONAsString);

	accumulatedDp_->resetAccumulator();
	FileIO::writeStringToFile(path, outputJSONAsString);
}

void FastMonitor::getHostAndPID(std::string& sHPid)
{
	std::stringstream hpid;
	int pid = (int) getpid();
	char hostname[128];
	gethostname(hostname, sizeof hostname);
	hpid << hostname << "_" << pid;
	sHPid = hpid.str();
}


/* @SM:PROBABLY WON't NEED THIS ANYMORE
DataPoint* ObjectMerger::csvToJson(string& olCSV, DataPointDefinition* dpd,
		string defPath) {

	DataPoint* dp = new DataPoint();
	dp->setDefinition(defPath);

	vector<string> tokens;
	std::istringstream ss(olCSV);
	while (!ss.eof()) {
		string field;
		getline(ss, field, ',');
		tokens.push_back(field);
	}

	dp->resetData();

	for (unsigned int i = 0; i < tokens.size(); i++) {
		string currentOpName = dpd->getLegendFor(i).getOperation();
		int index = atoi(tokens[i].c_str());
		if (currentOpName.compare(Operations::HISTO) == 0) {
			vector<int> histo;
			Utils::bumpIndex(histo, index);
			string histoStr;
			Utils::valueArrayToString<int>(histo, histoStr);
			dp->addToData(histoStr);
		} else
			dp->addToData(tokens[i]);
	}

	return dp;
}
*/

