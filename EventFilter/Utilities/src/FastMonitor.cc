/*
 * FastMonitor.cc
 *
 *  Created on: Nov 27, 2012
 *      Author: aspataru
 */

#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/JsonSerializable.h"
#include "EventFilter/Utilities/interface/FileIO.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <assert.h>

using namespace jsoncollector;

FastMonitor::FastMonitor(std::string const& defPath, bool strictChecking) : //strictChecking-->false
	defPath_(defPath),strictChecking_(strictChecking),nStreams_(1)
{

	//get host and PID info
	getHostAndPID(sourceInfo_);

	//TODO:strict checking

	//first load definition file
	DataPointDefinition::getDataPointDefinitionFor(defPath_, sourceInfo_, dpd_);

	for (int i=0;i<dpd_.getNumberOfElements();i++) {
	  jsonPtrAtIndex_.push_back(nullptr);
	}
}

//maybe should destroy datapoints here
FastMonitor::~FastMonitor() {
}

void FastMonitor::registerGlobalMonitorable(JsonMonitorable *newMonitorable, bool NAifZeroUpdates)
{
	unsigned int jsonIndex;
	DataPoint *dp = new DataPoint(dpd_,defPath_,sourceInfo_);//(do we need defpath & sourceInfo here?)
	DataPoint dp->trackMonitorable(newMonitorable,NAifZeroUpdates);
	dataPoints_.push_back(dp);
}

void FastMonitor::registerStreamMonitorableUIntVec(std::string const& name, 
		std::vector<unsigned int> *inputsPtr, bool NAifZeroUpdates)
{
	DataPoint *dp = new DataPoint(dpd_,defPath_,sourceInfo_);
	DataPoint dp->trackVectorUInt(name,inputsPtr,NAifZeroUpdates);
        dataPoints_.push_back(dp);
}


//atomic variables with guaranteed updates at the time of reading
void FastMonitor::registerStreamMonitorableUIntVecAtomic(std::string const& name, 
		std::vector<std::atomic<unsigned int>> *inputsRef_, bool NAifZeroUpdates)
{
	DataPoint *dp = new DataPoint(dpd_,defPath_,sourceInfo_);
	DataPoint dp->trackVectorUintAtomic(std::string const& name, std::vector<std::atomic<unsigned int>> *inputsRef_,bool NAifZeroUpdates);
        dataPoints_.push_back(dp);
}



void FastMonitor::commit(std::vector<std::atomic<unsigned int>> *streamLumisPtr)
{
	streamLumi_=streamLumi;

	std::vector<std::string> & jsonNames= dpd_.getNames();
	regDpCount_ = dataPoints_.size();
	assert(strictChecking_ && jsonNames.size()==regDpCount_);

	std::map<unsigned int,bool> hasJson;
	for (unsigned int i=0;i<jsonNames.size();i++)
	{
	  bool notFoundVar=true;
	  for (unsigned int j=0;j<regDpCount;j++) {
	    if (dataPoints_[j]->getName()==jsonNames(i))
	    {
	      dataPoints_[j]->setOperation(dpd_.getOperationFor(i));
	      jsonDpIndex_.push_back(j);
              hasJson[j]=true;
	      notFoundVar=false;
	      break;
	    }
	  }
	  if (notFoundVar) {
	    assert(!strictChecking_);
	    //push dummy DP if not registered by the service so that we output required JSON/CSV
	    DataPoint *dummyDp = new DataPoint(dpd_,defPath_,sourceInfo_);
	    dummyDp->trackDummy(jsonNames[i]);
	    dataPoints_.push_back(dummyDp);
	    jsonDpIndex_.push_back(dataPoints_.size()-1);
	  }
	}
	for (unsigned int i=0;i<regDpCount;j++) {
		dataPoints_[i]->setStreamLumiPtr(streamLumisPtr);
	}
}


//update everything
void FastMonitor::snap(bool outputCSVFile, std::string const& path, unsigned int forLumi) {

  recentSnaps_++;
  recentSnapsTimer_++;
  //do this only for real ones, not dummies
  for (unsigned int i=0;i<regDpCount_;i++) {
    dataPoints_[i]->snap(forLumi);
  }
  if (outputCVSFile) otuputCSV(path);
}


//update for global variables as most of them are correct only at global EOL
void FastMonitor::snapGlobal(bool outputCSVFile, std::string const& path, unsigned int forLumi) {

  recentSnaps_++;
  //do this only for real ones, not dummies
  for (unsigned int i=0;i<regDpCount_;i++) {
    dataPoints_[i]->snapGlobal(forLumi);
  }
  if (outputCVSFile) otuputCSV(path);
}

//update atomic per-stream vars(e.g. event counters) not updating time-based measurements (mini/microstate)
void FastMonitor::snapStreamAtomic(bool outputCSV, std::string const& path, unsigned int streamID, unsigned int forLumi)
{
  recentSnaps_++;
  //do this only for real ones, not dummies
  for (unsigned int i=0;i<regDpCount_;i++) {
    dataPoints_[i]->snapStreamAtomic(streamID, forLumi);
  }
  if (outputCVSFile) otuputCSV(path);

}

//todo: check for faliures
void FastMonitor::outputCSV(std::string const& path)
{
    //output what was specified in JSON in the same order (including dummies)
    unsigned int monSize = jsonDpIndex_.size();
    std::stringstream ss;
    if (monSize)
      for (unsigned int j; j< monSize;j++) 
	ss << dataPoints_[jsonDpIndex_[j]]->fastOutLatest();
	if (j<monSize-1) ss < ",";
      }
    std::ofstream outputFile;
    outputFile.open(path.c_str(), std::fstream::out | std::fstream::trunc);
    outputFile << defPath_ << std::endl;
    outputFile << ss.str();
    outputFile << std::endl;
    outputFile.close();

}

//get one variable - use auto_ptr?
JsonMonitorable* FastMonitor::getMergedIntJforLumi(std::string const& name,unsigned int forLumi)
{
  auto it = dpNameMap_[name];
  assert(it!=std::map:end);
  return  dataPoints_[*it]->mergeAndRetrieve(forLumi);
}

//serialization step
bool FastMonitor::outputFullJSON(std::string const& path, unsigned int lumi) {

	std::cout << "SNAP updates: " <<  recentSnaps_ << " (by timer: " << recentSnapsBytimer_ << ") in lumisection" << lumi << std::endl;
        recentSnaps_ = recentSnapsTimer_ = 0;

        Json::Value serializeRoot;
        for (unsigned int j; j< jsonDpIndex_.size();j++) 
          ss << dataPoints_[jsonDpIndex_[j]]->mergeAndSerialize(serializeRoot,lumi,j==0);//merge and serialize

        Json::StyledWriter writer;
        //serialize to file (todo check for failures)
        FileIO::writeStringToFile(path, writer.write(serializeRoot));
	return true;
}

void FastMonitor::discardCollected(unsigned int forLumi) {
	for (dp: dataPoints_) dp->discardCollected(forLumi);
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

