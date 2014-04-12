#include "../interface/EcalCondDBWriter.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TObjArray.h"
#include "TPRegexp.h"
#include "TString.h"

void
setBit(int& _bitArray, unsigned _iBit)
{
  _bitArray |= (0x1 << _iBit);
}

bool
getBit(int& _bitArray, unsigned _iBit)
{
  return (_bitArray & (0x1 << _iBit)) != 0;
}

EcalCondDBWriter::EcalCondDBWriter(edm::ParameterSet const& _ps) :
  db_(0),
  location_(_ps.getUntrackedParameter<std::string>("location")),
  runType_(_ps.getUntrackedParameter<std::string>("runType")),
  runGeneralTag_(_ps.getUntrackedParameter<std::string>("runGeneralTag")),
  monRunGeneralTag_(_ps.getUntrackedParameter<std::string>("monRunGeneralTag")),
  inputRootFiles_(_ps.getUntrackedParameter<std::vector<std::string> >("inputRootFiles")),
  summaryWriter_(_ps.getUntrackedParameterSet("workerParams")),
  verbosity_(_ps.getUntrackedParameter<int>("verbosity")),
  executed_(false)
{
  if(inputRootFiles_.size() == 0)
    throw cms::Exception("Configuration") << "No input ROOT file given";

  std::string DBName(_ps.getUntrackedParameter<std::string>("DBName"));
  std::string hostName(_ps.getUntrackedParameter<std::string>("hostName"));
  int hostPort(_ps.getUntrackedParameter<int>("hostPort"));
  std::string userName(_ps.getUntrackedParameter<std::string>("userName"));
  std::string password(_ps.getUntrackedParameter<std::string>("password"));

  std::auto_ptr<EcalCondDBInterface> db(0);

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << "Establishing DB connection";

  try{
    db = std::auto_ptr<EcalCondDBInterface>(new EcalCondDBInterface(DBName, userName, password));
  }
  catch(std::runtime_error& re){
    if(hostName != ""){
      try{
        db = std::auto_ptr<EcalCondDBInterface>(new EcalCondDBInterface(hostName, DBName, userName, password, hostPort));
      }
      catch(std::runtime_error& re2){
        throw cms::Exception("DBError") << re2.what();
      }
    }
    else
      throw cms::Exception("DBError") << re.what();
  }

  db_ = db.release();

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << " Done.";

  edm::ParameterSet const& workerParams(_ps.getUntrackedParameterSet("workerParams"));

  workers_[Integrity] = new ecaldqm::IntegrityWriter(workerParams);
  workers_[Cosmic] = 0;
  workers_[Laser] = new ecaldqm::LaserWriter(workerParams);
  workers_[Pedestal] = new ecaldqm::PedestalWriter(workerParams);
  workers_[Presample] = new ecaldqm::PresampleWriter(workerParams);
  workers_[TestPulse] = new ecaldqm::TestPulseWriter(workerParams);
  workers_[BeamCalo] = 0;
  workers_[BeamHodo] = 0;
  workers_[TriggerPrimitives] = 0;
  workers_[Cluster] = 0;
  workers_[Timing] = new ecaldqm::TimingWriter(workerParams);
  workers_[Led] = new ecaldqm::LedWriter(workerParams);
  workers_[RawData] = 0;
  workers_[Occupancy] = new ecaldqm::OccupancyWriter(workerParams);

  for(unsigned iC(0); iC < nTasks; ++iC)
    if(workers_[iC]) workers_[iC]->setVerbosity(verbosity_);
}

EcalCondDBWriter::~EcalCondDBWriter()
{
  try{
    delete db_;
  }
  catch(std::runtime_error& e){
    throw cms::Exception("DBError") << e.what();
  }

  for(unsigned iC(0); iC < nTasks; ++iC)
    delete workers_[iC];
}

void
EcalCondDBWriter::analyze(edm::Event const&, edm::EventSetup const&)
{
  if(executed_) return;

  /////////////////////// INPUT INITIALIZATION /////////////////////////

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << "Initializing DQMStore from input ROOT files";

  DQMStore& dqmStore(*edm::Service<DQMStore>());

  int runNumber(0);

  for(unsigned iF(0); iF < inputRootFiles_.size(); ++iF){
    std::string& fileName(inputRootFiles_[iF]);

    if(verbosity_ > 1) edm::LogInfo("EcalDQM") << " " << fileName;

    TPRegexp pat("DQM_V[0-9]+(?:|_[0-9a-zA-Z]+)_R([0-9]+)");
    std::auto_ptr<TObjArray> matches(pat.MatchS(fileName.c_str()));
    if(matches->GetEntries() == 0)
      throw cms::Exception("Configuration") << "Input file " << fileName << " is not an DQM output";

    if(iF == 0)
      runNumber = TString(matches->At(1)->GetName()).Atoi();
    else if(TString(matches->At(1)->GetName()).Atoi() != runNumber)
      throw cms::Exception("Configuration") << "Input files disagree in run number";

    dqmStore.open(fileName, false, "", "", DQMStore::StripRunDirs);
  }

  if(verbosity_ > 1) edm::LogInfo("EcalDQM") << " Searching event info";

  uint64_t timeStampInFile(0);
  unsigned processedEvents(0);

  dqmStore.cd();
  std::vector<std::string> dirs(dqmStore.getSubdirs());
  for(unsigned iD(0); iD < dirs.size(); ++iD){
    if(!dqmStore.dirExists(dirs[iD] + "/EventInfo")) continue;

    MonitorElement* timeStampME(dqmStore.get(dirs[iD] + "/EventInfo/runStartTimeStamp"));
    if(timeStampME){
      double timeStampValue(timeStampME->getFloatValue());
      uint64_t seconds(timeStampValue);
      uint64_t microseconds((timeStampValue - seconds) * 1.e6);
      timeStampInFile = (seconds << 32) | microseconds;
    }

    MonitorElement* eventsME(dqmStore.get(dirs[iD] + "/EventInfo/processedEvents"));
    if(eventsME)
      processedEvents = eventsME->getIntValue();

    if(timeStampInFile != 0 && processedEvents != 0){
      if(verbosity_ > 1) edm::LogInfo("EcalDQM") << " Event info found; timestamp=" << timeStampInFile << " processedEvents=" << processedEvents;
      break;
    }
  }

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << " Done.";

  //////////////////////// SOURCE INITIALIZATION //////////////////////////

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << "Setting up source MonitorElements for given run type " << runType_;

  int taskList(0);
  for(unsigned iC(0); iC < nTasks; ++iC){
    if(!workers_[iC] || !workers_[iC]->runsOn(runType_)) continue;

    workers_[iC]->retrieveSource(dqmStore);

    setBit(taskList, iC);
  }

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << " Done.";

  //////////////////////// DB INITIALIZATION //////////////////////////

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << "Initializing DB entry";

  RunIOV runIOV;
  RunTag runTag;
  try{
    runIOV = db_->fetchRunIOV(location_, runNumber);
    runTag = runIOV.getRunTag();
  }
  catch(std::runtime_error& e){
    std::cerr << e.what();

    if(timeStampInFile == 0)
      throw cms::Exception("Initialization") << "Time stamp for the run could not be found";

    LocationDef locationDef;
    locationDef.setLocation(location_);
    RunTypeDef runTypeDef;
    runTypeDef.setRunType(runType_);
    runTag.setLocationDef(locationDef);
    runTag.setRunTypeDef(runTypeDef);
    runTag.setGeneralTag(runGeneralTag_);

    runIOV.setRunStart(Tm(timeStampInFile));
    runIOV.setRunNumber(runNumber);
    runIOV.setRunTag(runTag);

    try{
      db_->insertRunIOV(&runIOV);
      runIOV = db_->fetchRunIOV(&runTag, runNumber);
    }
    catch(std::runtime_error& e){
      throw cms::Exception("DBError") << e.what();
    }
  }

  // No filtering - DAQ definitions change time to time..
//   if(runType_ != runIOV.getRunTag().getRunTypeDef().getRunType())
//     throw cms::Exception("Configuration") << "Given run type " << runType_ << " does not match the run type in DB " << runIOV.getRunTag().getRunTypeDef().getRunType();

  MonVersionDef versionDef;
  versionDef.setMonitoringVersion("test01"); // the only mon_ver in mon_version_def table as of September 2012
  MonRunTag monTag;
  monTag.setMonVersionDef(versionDef);
  monTag.setGeneralTag(monRunGeneralTag_);

  MonRunIOV monIOV;

  try{
    monIOV = db_->fetchMonRunIOV(&runTag, &monTag, runNumber, 1);
  }
  catch(std::runtime_error& e){
    std::cerr << e.what();

    monIOV.setRunIOV(runIOV);
    monIOV.setSubRunNumber(1);
    monIOV.setSubRunStart(runIOV.getRunStart());
    monIOV.setSubRunEnd(runIOV.getRunEnd());
    monIOV.setMonRunTag(monTag);

    try{
      db_->insertMonRunIOV(&monIOV);
      monIOV = db_->fetchMonRunIOV(&runTag, &monTag, runNumber, 1);
    }
    catch(std::runtime_error& e){
      throw cms::Exception("DBError") << e.what();
    }
  }

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << " Done.";

  //////////////////////// DB WRITING //////////////////////////

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << "Writing to DB";

  int outcome(0);
  for(unsigned iC(0); iC < nTasks; ++iC){
    if(!getBit(taskList, iC)) continue;

    if(verbosity_ > 1) edm::LogInfo("EcalDQM") << " " << workers_[iC]->getName();

    if(workers_[iC]->isActive() && workers_[iC]->run(db_, monIOV)) setBit(outcome, iC);
  }

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << " Done.";

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << "Registering the outcome of DB writing";

  summaryWriter_.setTaskList(taskList);
  summaryWriter_.setOutcome(outcome);
  summaryWriter_.setProcessedEvents(processedEvents);
  summaryWriter_.run(db_, monIOV);

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << " Done.";

  executed_ = true;
}

DEFINE_FWK_MODULE(EcalCondDBWriter);
