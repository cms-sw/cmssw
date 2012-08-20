#include "DQM/EcalCommon/interface/EcalDQMonitor.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DQM/EcalCommon/interface/DQWorker.h"

#include <sstream>

using namespace ecaldqm;

EcalDQMonitor::EcalDQMonitor(const edm::ParameterSet &_ps) :
  moduleName_(_ps.getUntrackedParameter<std::string>("moduleName")),
  mergeRuns_(_ps.getUntrackedParameter<bool>("mergeRuns")),
  verbosity_(_ps.getUntrackedParameter<int>("verbosity")),
  initialized_(false)
{
  using namespace std;

  vector<string> workerNames(_ps.getUntrackedParameter<vector<string> >("workers"));
  const edm::ParameterSet& allParams(_ps.getUntrackedParameterSet("workerParameters"));
  const edm::ParameterSet& mePaths(_ps.getUntrackedParameterSet("mePaths"));

  WorkerFactory factory(0);

  for(vector<string>::iterator wItr(workerNames.begin()); wItr != workerNames.end(); ++wItr){
    if (!(factory = SetWorker::findFactory(*wItr))) continue;

    if(verbosity_ > 0) cout << moduleName_ << ": Setting up " << *wItr << endl;

    map<string, vector<MEData> >::iterator dItr(DQWorker::meData.find(*wItr));
    if(dItr == DQWorker::meData.end())
      throw cms::Exception("InvalidCall") << "MonitorElement setup data not found for " << *wItr << endl;

    string topDir;
    if(allParams.existsAs<edm::ParameterSet>(*wItr)){
      edm::ParameterSet const& workerParams(allParams.getUntrackedParameterSet(*wItr));
      if(workerParams.existsAs<string>("topDirectory"))
        topDir = workerParams.getUntrackedParameter<string>("topDirectory");
    }
    else
      topDir = allParams.getUntrackedParameterSet("Common").getUntrackedParameter<string>("topDirectory");

    for(vector<MEData>::iterator mItr(dItr->second.begin()); mItr != dItr->second.end(); ++mItr)
      mItr->fullPath = topDir + "/" + mePaths.getUntrackedParameterSet(*wItr).getUntrackedParameter<string>(mItr->pathName);

    DQWorker* worker(factory(allParams));
    if(worker->getName() != *wItr){
      delete worker;

      if(verbosity_ > 0) cout << moduleName_ << ": " << *wItr << " could not be configured" << endl; 
      continue;
    }

    worker->setVerbosity(verbosity_);

    workers_.push_back(worker);
  }
}

EcalDQMonitor::~EcalDQMonitor()
{
}
