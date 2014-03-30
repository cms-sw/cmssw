/*
 * FastMonitor.cc
 *
 *  Created on: Nov 27, 2012
 *      Author: aspataru
 */

#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/JsonSerializable.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

using namespace jsoncollector;

FastMonitor::FastMonitor(std::string const& defPath, std::string const defGroup, bool strictChecking, bool useSource, bool useDefinition) :
	defPath_(defPath),strictChecking_(strictChecking),useSource_(useSource),useDefinition_(useDefinition),nStreams_(1),deleteDef_(true)
{
  //get host and PID info
  if (useSource)
    getHostAndPID(sourceInfo_);

  //load definition file
  dpd_ = new DataPointDefinition();
  DataPointDefinition::getDataPointDefinitionFor(defPath_, dpd_,&defGroup);

}


FastMonitor::FastMonitor(DataPointDefinition * dpd, bool strictChecking, bool useSource, bool useDefinition) :
	strictChecking_(strictChecking),useSource_(useSource),useDefinition_(useDefinition),nStreams_(1),dpd_(dpd)
{
  //get host and PID info
  if (useSource)
    getHostAndPID(sourceInfo_);
}

FastMonitor::~FastMonitor()
{
  for (auto dp: dataPoints_) delete dp;
  if (deleteDef_) delete dpd_;
  if (deleteDefFast_) delete dpdFast_;
}

void FastMonitor::addFastPathDefinition(std::string const& defPathFast, std::string const defGroupFast, bool strict)
{
  haveFastPath_=true;
  defPathFast_=defPathFast;
  dpdFast_ = new DataPointDefinition();
  DataPointDefinition::getDataPointDefinitionFor(defPathFast_, dpdFast_,&defGroupFast);
  fastPathStrictChecking_=strict;
  deleteDefFast_=true;
}

//per-process variables
void FastMonitor::registerGlobalMonitorable(JsonMonitorable *newMonitorable, bool NAifZeroUpdates, unsigned int *nBins)
{
  DataPoint *dp = new DataPoint(sourceInfo_,defPath_);
  dp->trackMonitorable(newMonitorable,NAifZeroUpdates);
  dp->setNBins(nBins);
  dataPoints_.push_back(dp);
  dpNameMap_[newMonitorable->getName()]=dataPoints_.size()-1;

  //checks if the same name is registered twice
  assert(uids_.insert(newMonitorable->getName()).second);
}

//fast path: no merge operation is performed
void FastMonitor::registerFastGlobalMonitorable(JsonMonitorable *newMonitorable)
{
  DataPoint *dp = new DataPoint(sourceInfo_,defPathFast_,true);
  dp->trackMonitorable(newMonitorable,false);
  dataPointsFastOnly_.push_back(dp);
}

//per-stream variables
void FastMonitor::registerStreamMonitorableUIntVec(std::string const& name, std::vector<unsigned int> *inputs, bool NAifZeroUpdates, unsigned int *nBins)
{
  DataPoint *dp = new DataPoint(sourceInfo_,defPath_);
  dp->trackVectorUInt(name,inputs,NAifZeroUpdates);
  dp->setNBins(nBins);
  dataPoints_.push_back(dp);
  dpNameMap_[name]=dataPoints_.size()-1;
  assert (uids_.insert(name).second);
}


//atomic variables with guaranteed updates at the time of reading
void FastMonitor::registerStreamMonitorableUIntVecAtomic(std::string const& name, std::vector<AtomicMonUInt*> *inputs, bool NAifZeroUpdates, unsigned int *nBins)
{
  std::string definitionToPass;
  if (useDefinition_) definitionToPass=defPath_;
  DataPoint *dp = new DataPoint(definitionToPass,sourceInfo_);
  dp->trackVectorUIntAtomic(name,inputs,NAifZeroUpdates);
  dp->setNBins(nBins);
  dataPoints_.push_back(dp);
  dpNameMap_[name]=dataPoints_.size()-1;
  assert (uids_.insert(name).second);
}



void FastMonitor::commit(std::vector<unsigned int> *streamLumisPtr)
{
  std::vector<std::string> const& jsonNames= dpd_->getNames();
  regDpCount_ = dataPoints_.size();
  if (strictChecking_)
    assert(jsonNames.size()==regDpCount_);

  std::map<unsigned int,bool> hasJson;
  for (unsigned int i=0;i<jsonNames.size();i++)
  {
    bool notFoundVar=true;
    for (unsigned int j=0;j<regDpCount_;j++) {
      if (dataPoints_[j]->getName()==jsonNames[i])
      {
	dataPoints_[j]->setOperation(dpd_->getOperationFor(i));
	jsonDpIndex_.push_back(j);
	hasJson[j]=true;
	notFoundVar=false;
	break;
      }
    }
    if (notFoundVar) {
      assert(!strictChecking_);
      //push dummy DP if not registered by the service so that we output required JSON/CSV
      DataPoint *dummyDp = new DataPoint(sourceInfo_,defPath_);
      dummyDp->trackDummy(jsonNames[i],true);
      dataPoints_.push_back(dummyDp);
      jsonDpIndex_.push_back(dataPoints_.size()-1);
    }
  }
  for (unsigned int i=0;i<regDpCount_;i++) {
    dataPoints_[i]->setStreamLumiPtr(streamLumisPtr);
  }

  //fast path:
  if (haveFastPath_) {
    std::vector<std::string> const& fjsonNames = dpdFast_->getNames();
    fregDpCount_ = dataPointsFastOnly_.size();
    assert(!(fastPathStrictChecking_ && fjsonNames.size()==fregDpCount_));
    std::map<unsigned int,bool> fhasJson;
    for (unsigned int i=0;i<fjsonNames.size();i++)
    {
      bool notFoundVar=true;
      for (unsigned int j=0;j<fregDpCount_;j++) {
	if (dataPointsFastOnly_[j]->getName()==fjsonNames[i])
	{
	  jsonDpIndexFast_.push_back(dataPointsFastOnly_[j]);
	  fhasJson[j]=true;
	  notFoundVar=false;
	  break;
	}
      }
      if (notFoundVar)
      {
        //try to find variable among slow variables

        bool notFoundVarSlow=true;
        for (unsigned int j=0;j<regDpCount_;j++) {
	  if (dataPoints_[j]->getName()==fjsonNames[i])
	  {
	    jsonDpIndexFast_.push_back(dataPoints_[j]);
	    //fhasJson[j]=true;
	    notFoundVarSlow=false;
	    break;
	  }
	}

	assert(!(fastPathStrictChecking_ && !notFoundVarSlow));
	//push dummy DP if not registered by the service so that we output required JSON/CSV
	if (notFoundVarSlow) {
	  DataPoint *dummyDp = new DataPoint(sourceInfo_,defPathFast_);
	  dummyDp->trackDummy(fjsonNames[i],true);
	  dataPointsFastOnly_.push_back(dummyDp);
	  jsonDpIndexFast_.push_back(dummyDp);
        }
      }
    }
  } 
}

//update everything
void FastMonitor::snap(unsigned int ls)
{
  recentSnaps_++;
  recentSnapsTimer_++;
  for (unsigned int i=0;i<regDpCount_;i++) {
    dataPoints_[i]->snap(ls);
  }
}

//update for global variables as most of them are correct only at global EOL
void FastMonitor::snapGlobal(unsigned int ls)
{
  recentSnaps_++;
  for (unsigned int i=0;i<regDpCount_;i++) {
    dataPoints_[i]->snapGlobal(ls);
  }
}

//update atomic per-stream vars(e.g. event counters) not updating time-based measurements (mini/microstate)
void FastMonitor::snapStreamAtomic(unsigned int ls, unsigned int streamID)
{
  recentSnaps_++;
  for (unsigned int i=0;i<regDpCount_;i++) {
    dataPoints_[i]->snapStreamAtomic(ls, streamID);
  }
}

std::string FastMonitor::getCSVString()
{
  //output what was specified in JSON in the same order (including dummies)
  unsigned int monSize = jsonDpIndexFast_.size();
  std::stringstream ss;
  if (monSize) {
    for (unsigned int j=0; j< monSize;j++) { 
      ss << jsonDpIndexFast_[j]->fastOutCSV();
      if (j<monSize-1) ss << ",";
    }
  }
  return ss.str();
}

void FastMonitor::outputCSV(std::string const& path, std::string const& csvString)
{
  std::ofstream outputFile;
  outputFile.open(path.c_str(), std::fstream::out | std::fstream::trunc);
  outputFile << defPathFast_ << std::endl;
  outputFile << csvString << std::endl;
  outputFile.close();
}


//get one variable (caller must delete it later)
JsonMonitorable* FastMonitor::getMergedIntJForLumi(std::string const& name,unsigned int forLumi)
{
  auto it = dpNameMap_.find(name);
  assert(it!=dpNameMap_.end());
  return  dataPoints_[it->second]->mergeAndRetrieveValue(forLumi);
}

bool FastMonitor::outputFullJSON(std::string const& path, unsigned int lumi, bool log)
{
  if (log)
    edm::LogInfo("FastMonitor") << "SNAP updates: " <<  recentSnaps_ << " (by timer: " << recentSnapsTimer_ 
                              << ") in lumisection ";

  recentSnaps_ = recentSnapsTimer_ = 0;

  Json::Value serializeRoot;
  for (unsigned int j=0; j< jsonDpIndex_.size();j++) {
    dataPoints_[jsonDpIndex_[j]]->mergeAndSerialize(serializeRoot,lumi,j==0);
  }

  Json::StyledWriter writer;
  std::string && result = writer.write(serializeRoot);
  FileIO::writeStringToFile(path, result);
  return true;
}

void FastMonitor::discardCollected(unsigned int forLumi)
{
  for (auto dp: dataPoints_) dp->discardCollected(forLumi);
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

