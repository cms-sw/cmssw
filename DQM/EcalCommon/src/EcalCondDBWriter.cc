#include "DQM/EcalCommon/interface/EcalCondDBWriter.h"

#include <iostream>
#include <set>
#include <fstream>
#include <sstream>
#include <exception>
#include <stdexcept>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/EcalCommon/interface/LogicID.h"

#include "OnlineDB/EcalCondDB/interface/MonRunDat.h"

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
  tagName_(_ps.getUntrackedParameter<std::string>("tagName")),
  location_(_ps.getUntrackedParameter<std::string>("location")),
  runType_(_ps.getUntrackedParameter<std::string>("runType")),
  inputRootFiles_(_ps.getUntrackedParameter<std::vector<std::string> >("inputRootFiles")),
  workerParams_(_ps.getUntrackedParameterSet("workerParams")),
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

  if(verbosity_ > 0) std::cout << "Establishing DB connection" << std::endl;

  if(hostName == ""){
    try{
      db = std::auto_ptr<EcalCondDBInterface>(new EcalCondDBInterface(DBName, userName, password));
    }
    catch(std::runtime_error& re){
      throw cms::Exception("DBError") << re.what();
    }
  }
  else{
    try{
      db = std::auto_ptr<EcalCondDBInterface>(new EcalCondDBInterface(hostName, DBName, userName, password, hostPort));
    }
    catch(std::runtime_error& re){
      throw cms::Exception("DBError") << re.what();
    }
  }

  db_ = db.release();

  if(verbosity_ > 0) std::cout << " Done." << std::endl;

  workers_[Integrity] = new ecaldqm::IntegrityWriter();
  workers_[Cosmic] = 0;
  workers_[Laser] = new ecaldqm::LaserWriter();
  workers_[Pedestal] = new ecaldqm::PedestalWriter();
  workers_[Presample] = new ecaldqm::PresampleWriter();
  workers_[TestPulse] = new ecaldqm::TestPulseWriter();
  workers_[BeamCalo] = 0;
  workers_[BeamHodo] = 0;
  workers_[TriggerPrimitives] = 0;
  workers_[Cluster] = 0;
  workers_[Timing] = new ecaldqm::TimingWriter();
  workers_[Led] = new ecaldqm::LedWriter();
  workers_[RawData] = new ecaldqm::RawDataWriter();
  workers_[Occupancy] = new ecaldqm::OccupancyWriter();

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

  std::set<std::string> runTypes;

  runTypes.insert("COSMIC");
  runTypes.insert("BEAM");
  runTypes.insert("MTCC");
  runTypes.insert("LASER");
  runTypes.insert("TEST_PULSE");
  runTypes.insert("PEDESTAL");
  runTypes.insert("PEDESTAL-OFFSET");
  runTypes.insert("LED");
  runTypes.insert("PHYSICS");
  runTypes.insert("HALO");
  runTypes.insert("CALIB");

  if(runTypes.find(runType_) == runTypes.end())
    throw cms::Exception("Configuration") << "Run type " << runType_ << " not defined";

  std::set<std::string> enabledRunTypes[nTasks];

  enabledRunTypes[Integrity] = runTypes;
  enabledRunTypes[Integrity].erase("HALO");
  enabledRunTypes[Integrity].erase("CALIB");

  enabledRunTypes[Cosmic] = runTypes;
  enabledRunTypes[Cosmic].erase("BEAM");
  enabledRunTypes[Cosmic].erase("PEDESTAL_OFFSET");
  enabledRunTypes[Cosmic].erase("HALO");
  enabledRunTypes[Cosmic].erase("CALIB");

  enabledRunTypes[Laser] = runTypes;
  enabledRunTypes[Laser].erase("PEDESTAL_OFFSET");
  enabledRunTypes[Laser].erase("HALO");
  enabledRunTypes[Laser].erase("CALIB");

  enabledRunTypes[Pedestal] = runTypes;
  enabledRunTypes[Pedestal].erase("PEDESTAL_OFFSET");
  enabledRunTypes[Pedestal].erase("HALO");
  enabledRunTypes[Pedestal].erase("CALIB");

  enabledRunTypes[Presample] = runTypes;
  enabledRunTypes[Presample].erase("PEDESTAL_OFFSET");
  enabledRunTypes[Presample].erase("HALO");
  enabledRunTypes[Presample].erase("CALIB");

  enabledRunTypes[TestPulse] = runTypes;
  enabledRunTypes[TestPulse].erase("PEDESTAL_OFFSET");
  enabledRunTypes[TestPulse].erase("HALO");
  enabledRunTypes[TestPulse].erase("CALIB");

  enabledRunTypes[BeamCalo] = std::set<std::string>();

  enabledRunTypes[BeamHodo] = std::set<std::string>();

  enabledRunTypes[TriggerPrimitives] = runTypes;
  enabledRunTypes[TriggerPrimitives].erase("PEDESTAL_OFFSET");
  enabledRunTypes[TriggerPrimitives].erase("HALO");
  enabledRunTypes[TriggerPrimitives].erase("CALIB");

  enabledRunTypes[Cluster] = runTypes;
  enabledRunTypes[Cluster].erase("PEDESTAL_OFFSET");
  enabledRunTypes[Cluster].erase("HALO");
  enabledRunTypes[Cluster].erase("CALIB");

  enabledRunTypes[Timing] = runTypes;
  enabledRunTypes[Timing].erase("PEDESTAL_OFFSET");
  enabledRunTypes[Timing].erase("HALO");
  enabledRunTypes[Timing].erase("CALIB");

  enabledRunTypes[Led] = runTypes;
  enabledRunTypes[Led].erase("PEDESTAL_OFFSET");
  enabledRunTypes[Led].erase("HALO");
  enabledRunTypes[Led].erase("CALIB");

  enabledRunTypes[RawData] = runTypes;
  enabledRunTypes[RawData].erase("HALO");
  enabledRunTypes[RawData].erase("CALIB");

  enabledRunTypes[Occupancy] = runTypes;
  enabledRunTypes[Occupancy].erase("HALO");
  enabledRunTypes[Occupancy].erase("CALIB");

  /////////////////////// INPUT INITIALIZATION /////////////////////////

  if(verbosity_ > 0) std::cout << "Initializing DQMStore from input ROOT files" << std::endl;

  DQMStore* dqmStore(&(*edm::Service<DQMStore>()));
  if(!dqmStore)
    throw cms::Exception("Service") << "DQMStore not found" << std::endl;

  int runNumber(0);

  for(unsigned iF(0); iF < inputRootFiles_.size(); ++iF){
    std::string& fileName(inputRootFiles_[iF]);

    if(verbosity_ > 1) std::cout << " " << fileName << std::endl;

    TPRegexp pat("DQM_V[0-9]+_[0-9a-zA-Z]+_R([0-9]+).root");
    std::auto_ptr<TObjArray> matches(pat.MatchS(fileName.c_str()));
    if(matches->GetEntries() == 0)
      throw cms::Exception("Configuration") << "Input file " << fileName << " is not an online DQM output";

    if(iF == 0)
      runNumber = TString(matches->At(1)->GetName()).Atoi();
    else if(TString(matches->At(1)->GetName()).Atoi() != runNumber)
      throw cms::Exception("Configuration") << "Input files disagree in run number";

    dqmStore->open(fileName, false, "", "", DQMStore::StripRunDirs);
  }

  if(verbosity_ > 1) std::cout << " Searching event info" << std::endl;

  uint64_t timeStampInFile(0);
  unsigned processedEvents(0);

  dqmStore->cd();
  std::vector<std::string> dirs(dqmStore->getSubdirs());
  for(unsigned iD(0); iD < dirs.size(); ++iD){
    if(!dqmStore->dirExists(dirs[iD] + "/EventInfo")) continue;

    MonitorElement* timeStampME(dqmStore->get(dirs[iD] + "/EventInfo/runStartTimeStamp"));
    if(timeStampME){
      double timeStampValue(timeStampME->getFloatValue());
      uint64_t seconds(timeStampValue);
      uint64_t microseconds((timeStampValue - seconds) * 1.e6);
      timeStampInFile = (seconds << 32) | microseconds;
    }

    MonitorElement* eventsME(dqmStore->get(dirs[iD] + "/EventInfo/processedEvents"));
    if(eventsME)
      processedEvents = eventsME->getIntValue();

    if(timeStampInFile != 0 && processedEvents != 0){
      if(verbosity_ > 1) std::cout << " Event info found; timestamp=" << timeStampInFile << " processedEvents=" << processedEvents << std::endl;
      break;
    }
  }

  if(verbosity_ > 0) std::cout << " Done." << std::endl;

  //////////////////////// SOURCE INITIALIZATION //////////////////////////

  BinService const* binService(&(*(edm::Service<EcalDQMBinningService>())));
  if(!binService)
    throw cms::Exception("Service") << "EcalDQMBinningService not found" << std::endl;

  if(verbosity_ > 0) std::cout << "Setting up source MonitorElements for given run type " << runType_ << std::endl;

  int taskList(0);
  for(unsigned iC(0); iC < nTasks; ++iC){
    if(enabledRunTypes[iC].find(runType_) == enabledRunTypes[iC].end()) continue;

    if(!workers_[iC]) continue;

    workers_[iC]->setup(workerParams_, binService);
    workers_[iC]->retrieveSource();

    setBit(taskList, iC);
  }

  if(verbosity_ > 0) std::cout << " Done." << std::endl;

  //////////////////////// DB INITIALIZATION //////////////////////////

  if(verbosity_ > 0) std::cout << "Initializing DB entry" << std::endl;

  LocationDef locationDef;
  locationDef.setLocation(location_);
  RunTypeDef runTypeDef;
  runTypeDef.setRunType(runType_);
  RunTag runTag;
  runTag.setLocationDef(locationDef);
  runTag.setRunTypeDef(runTypeDef);
  runTag.setGeneralTag(runType_);

  RunIOV runIOV;
  bool runIOVDefined(false);
  try{
    runIOV = db_->fetchRunIOV(location_, runNumber);
    runIOVDefined = true;
  }
  catch(std::runtime_error& e){
    std::cerr << e.what() << std::endl;
  }

  if(!runIOVDefined){
    if(timeStampInFile == 0)
      throw cms::Exception("Initialization") << "Time stamp for the run could not be found";

    runIOV.setRunStart(Tm(timeStampInFile));
    runIOV.setRunNumber(runNumber);
    runIOV.setRunTag(runTag);

    try{
      db_->insertRunIOV(&runIOV);
      runIOV = db_->fetchRunIOV(location_, runNumber);
    }
    catch(std::runtime_error& e){
      throw cms::Exception("DBError") << e.what();
    }
  }

  if(runType_ != runIOV.getRunTag().getRunTypeDef().getRunType())
    throw cms::Exception("Configuration") << "Given run type " << runType_ << " does not match the run type in DB " << runIOV.getRunTag().getRunTypeDef().getRunType();

  MonVersionDef versionDef;
  versionDef.setMonitoringVersion("test01");
  MonRunTag monTag;
  monTag.setMonVersionDef(versionDef);
  monTag.setGeneralTag(tagName_);

  MonRunIOV monIOV;

  bool subRunIOVDefined(false);
  try{
    monIOV = db_->fetchMonRunIOV(&runTag, &monTag, runNumber, 1);
    subRunIOVDefined = true;
  }
  catch(std::runtime_error& e){
    std::cerr << e.what() << std::endl;
  }

  if(!subRunIOVDefined){
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

  if(verbosity_ > 0) std::cout << " Done." << std::endl;

  //////////////////////// DB WRITING //////////////////////////

  if(verbosity_ > 0) std::cout << "Writing to DB" << std::endl;

  int outcome(0);
  for(unsigned iC(0); iC < nTasks; ++iC){
    if(!getBit(taskList, iC)) continue;

    if(verbosity_ > 1) std::cout << " " << workers_[iC]->getName() << std::endl;

    if(workers_[iC]->run(db_, monIOV)) setBit(outcome, iC);
  }

  if(verbosity_ > 0) std::cout << " Done." << std::endl;

  if(verbosity_ > 0) std::cout << "Registering the outcome of DB writing" << std::endl;

  std::map<EcalLogicID, MonRunDat> dataset;
  MonRunDat& ebDat(dataset[LogicID::getEcalLogicID("EB")]);
  MonRunDat& eeDat(dataset[LogicID::getEcalLogicID("EE")]);

  MonRunOutcomeDef outcomeDef;
  outcomeDef.setShortDesc("success");

  ebDat.setNumEvents(processedEvents);
  eeDat.setNumEvents(processedEvents);

  ebDat.setMonRunOutcomeDef(outcomeDef);
  eeDat.setMonRunOutcomeDef(outcomeDef);

  ebDat.setTaskList(taskList);
  eeDat.setTaskList(taskList);

  ebDat.setTaskOutcome(outcome);
  eeDat.setTaskOutcome(outcome);

  try{
    db_->insertDataSet(&dataset, &monIOV);
  }
  catch(std::runtime_error& e){
    throw cms::Exception("DBError") << e.what();
  }

  if(verbosity_ > 0) std::cout << " Done." << std::endl;

  executed_ = true;
}
