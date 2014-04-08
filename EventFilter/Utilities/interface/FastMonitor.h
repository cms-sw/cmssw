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

#include <unordered_set>

namespace jsoncollector {

class FastMonitor {

public:

  FastMonitor(std::string const& defPath, std::string const defGroup, bool strictChecking, bool useSource=true, bool useDefinition=true);
  FastMonitor(DataPointDefinition * dpd, bool strictChecking, bool useSource=true, bool useDefinition=true);

  virtual ~FastMonitor();

  void addFastPathDefinition(std::string const& defPathFast, std::string const defGroupFast, bool strict);

  void setDefPath(std::string const& dpath) {defPath_=dpath;}

  void setNStreams(unsigned int nStreams) {nStreams_=nStreams;}

  //register global monitorable
  void registerGlobalMonitorable(JsonMonitorable *newMonitorable, bool NAifZeroUpdates, unsigned int *nBins=nullptr);

  //register fastPath global monitorable
  void registerFastGlobalMonitorable(JsonMonitorable *newMonitorable);

  //register per-stream monitores vector (unsigned int)
  void registerStreamMonitorableUIntVec(std::string const& name, 
      std::vector<unsigned int> *inputs, bool NAifZeroUpdates, unsigned int *nBins=nullptr);

  //NOT implemented yet
  //void registerStreamMonitorableIntVec(std::string &name, std::vector<unsigned int>,true,0);
  //void registerStreamMonitorableDoubleVec(std::string &name, std::vector<unsigned int>,true,0);
  //void registerStreamMonitorableStringVec(std::string &name, std::vector<std::string>,true,0);

  void registerStreamMonitorableUIntVecAtomic(std::string const& name,
      std::vector<AtomicMonUInt*> *inputs, bool NAifZeroUpdates, unsigned int *nBins=nullptr);

  //take vector used to track stream lumis and finish initialization
  void commit(std::vector<unsigned int> *streamLumisPtr);

  // fetches new snapshot and outputs one-line CSV if set (timer based)
  void snap(bool outputCSVFile, std::string const& path, unsigned int forLumi);

  //only update global variables (invoked at global EOL)
  void snapGlobal(bool outputCSVFile, std::string const& path, unsigned int forLumi);

  //only updates atomic vectors (for certain stream - at stream EOL)
  void snapStreamAtomic(bool outputCSVFile, std::string const& path, unsigned int streamID, unsigned int forLumi);

  //fastpath CSV
  void outputCSV(std::string const& path);

  //provide merged variable back to user
  JsonMonitorable* getMergedIntJForLumi(std::string const& name,unsigned int forLumi);

  // merges and outputs everything collected for the given stream to JSON file
  bool outputFullJSON(std::string const& path, unsigned int lumi);

  //discard what was collected for a lumisection
  void discardCollected(unsigned int forLumi);

  //this is added to the JSON file
  void getHostAndPID(std::string& sHPid);

private:

  std::string defPath_;
  std::string defPathFast_;
  bool strictChecking_;
  bool fastPathStrictChecking_;
  bool useSource_;
  bool useDefinition_;
  bool haveFastPath_=false;

  unsigned int nStreams_;

  std::string sourceInfo_;
  DataPointDefinition *dpd_;
  DataPointDefinition *dpdFast_;
  bool deleteDef_=false;
  bool deleteDefFast_=false;

  std::vector<DataPoint*> dataPoints_;
  std::vector<DataPoint*> dataPointsFastOnly_;
  std::vector<unsigned int> jsonDpIndex_;
  std::vector<DataPoint*> jsonDpIndexFast_;
  std::vector<DataPoint*> orphanedDps_;
  std::map<std::string,unsigned int> dpNameMap_;

  unsigned int recentSnaps_ = 0;
  unsigned int recentSnapsTimer_ = 0;
  unsigned int regDpCount_ = 0;
  unsigned int fregDpCount_ = 0;

  std::unordered_set<std::string> uids_;

};

}

#endif /* FASTMONITOR_H_ */
