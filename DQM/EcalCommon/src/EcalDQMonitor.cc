#include "DQM/EcalCommon/interface/EcalDQMonitor.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DQM/EcalCommon/interface/DQWorker.h"

#include <sstream>

using namespace ecaldqm;

EcalDQMonitor::EcalDQMonitor(const edm::ParameterSet &_ps) :
  moduleName_(_ps.getUntrackedParameter<std::string>("moduleName")),
  mergeRuns_(_ps.getUntrackedParameter<bool>("mergeRuns", false)),
  verbosity_(_ps.getUntrackedParameter<int>("verbosity", 0)),
  initialized_(false)
{
  using namespace std;

  vector<string> workerNames(_ps.getUntrackedParameter<vector<string> >("workers"));
  edm::ParameterSet const& allParams(_ps.getUntrackedParameterSet("workerParameters"));
  edm::ParameterSet const& commonParams(allParams.getUntrackedParameterSet("common"));

  WorkerFactory factory(0);

  for(vector<string>::iterator wItr(workerNames.begin()); wItr != workerNames.end(); ++wItr){
    if (!(factory = WorkerFactoryHelper::findFactory(*wItr))) continue;

    if(verbosity_ > 0) cout << moduleName_ << ": Setting up " << *wItr << endl;

    edm::ParameterSet const& workerParams(allParams.getUntrackedParameterSet(*wItr));

    DQWorker* worker(factory(workerParams, commonParams));
    if(worker->getName() != *wItr){
      delete worker;

      if(verbosity_ > 0) cout << moduleName_ << ": " << *wItr << " could not be configured" << endl; 
      continue;
    }

    worker->setVerbosity(verbosity_);

    workers_.push_back(worker);
  }
}
