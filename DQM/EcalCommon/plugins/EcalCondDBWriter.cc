#include "DQM/EcalCommon/interface/EcalCondDBWriter.h"

#include <iostream>
#include <set>
#include <fstream>
#include <sstream>
#include <exception>
#include <stdexcept>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/EcalCommon/interface/MESetUtils.h"
#include "DQM/EcalCommon/interface/LogicID.h"

#include "OnlineDB/EcalCondDB/interface/MonRunDat.h"

#include "TFile.h"
#include "TObjArray.h"
#include "TPRegexp.h"
#include "TObjString.h"
#include "TKey.h"
#include "TDirectory.h"
#include "TClass.h"

EcalCondDBWriter::EcalCondDBWriter(edm::ParameterSet const& _ps) :
  db_(0),
  tagName_(_ps.getUntrackedParameter<std::string>("tagName")),
  location_(_ps.getUntrackedParameter<std::string>("location")),
  runType_(_ps.getUntrackedParameter<std::string>("runType")),
  inputRootFiles_(_ps.getUntrackedParameter<std::vector<std::string> >("inputRootFiles")),
  executed_(false)
{
  clientNames_[Integrity] = "Integrity";
  clientNames_[Cosmic] = "Cosmic";
  clientNames_[Laser] = "Laser";
  clientNames_[Pedestal] = "Pedestal";
  clientNames_[Presample] = "Presample";
  clientNames_[TestPulse] = "TestPulse";
  clientNames_[BeamCalo] = "BeamCalo";
  clientNames_[BeamHodo] = "BeamHodo";
  clientNames_[TriggerPrimitives] = "TriggerPrimitives";
  clientNames_[Cluster] = "Cluster";
  clientNames_[Timing] = "Timing";
  clientNames_[Led] = "Led";
  clientNames_[RawData] = "RawData";
  clientNames_[Occupancy] = "Occupancy";

  edm::ParameterSet const& MESetParams(_ps.getUntrackedParameterSet("MESetParams"));
  for(unsigned iC(0); iC < nClients; ++iC)
    meSetParams_[iC] = MESetParams.getUntrackedParameterSet(clientNames_[iC]);

  if(inputRootFiles_.size() == 0)
    throw cms::Exception("Configuration") << "No input ROOT file given";

  std::string DBName(_ps.getUntrackedParameter<std::string>("DBName"));
  std::string hostName(_ps.getUntrackedParameter<std::string>("hostName"));
  int hostPort(_ps.getUntrackedParameter<int>("hostPort"));
  std::string userName(_ps.getUntrackedParameter<std::string>("userName"));
  std::string password(_ps.getUntrackedParameter<std::string>("password"));

  std::auto_ptr<EcalCondDBInterface> db(0);

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
}

EcalCondDBWriter::~EcalCondDBWriter()
{
  try{
    delete db_;
  }
  catch(std::runtime_error& e){
    throw cms::Exception("DBError") << e.what();
  }
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

  std::set<std::string> enabledRunTypes[nClients];

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

  DQMStore* dqmStore(&(*edm::Service<DQMStore>()));
  if(!dqmStore)
    throw cms::Exception("Service") << "DQMStore not found" << std::endl;
  BinService const* binService(&(*(edm::Service<EcalDQMBinningService>())));
  if(!binService)
    throw cms::Exception("Service") << "EcalDQMBinningService not found" << std::endl;

  int runNumber(0);

  for(unsigned iF(0); iF < inputRootFiles_.size(); ++iF){
    std::string& fileName(inputRootFiles_[iF]);

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

  uint64_t timeStampInFile(0);
  unsigned processedEvents(0);

  dqmStore->cd();
  std::vector<std::string> dirs(dqmStore->getSubdirs());
  for(unsigned iD(0); iD < dirs.size(); ++iD){
    std::cout << dirs[iD] << std::endl;
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

    if(timeStampInFile != 0 && processedEvents != 0)
      break;
  }

  int taskList(0);
  PtrMap<std::string, ecaldqm::MESet const> meSets[nClients];
  for(unsigned iC(0); iC < nClients; ++iC){
    if(enabledRunTypes[iC].find(runType_) != enabledRunTypes[iC].end()){
      taskList |= (0x1 << iC);

      edm::ParameterSet& meSetParams(meSetParams_[iC]);
      std::vector<std::string> meNames(meSetParams.getParameterNames());
      for(unsigned iP(0); iP < meNames.size(); ++iP){
        std::string& meName(meNames[iP]);
        edm::ParameterSet const& meSetParam(meSetParams.getUntrackedParameterSet(meName));
        ecaldqm::MESet const* meSet(ecaldqm::createMESet(meSetParam, binService, "", dqmStore));
        if(!meSet)
          throw cms::Exception("IOError") << "MonitorElement " << meName << " not found";
        meSets[iC][meName] = meSet;
      }
    }
  }


  //////////////////////// DB INITIALIZATION //////////////////////////

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


  //////////////////////// DB WRITING //////////////////////////

  int outcome(0);
  if(((taskList >> Integrity) & 0x1) && writeIntegrity(meSets[Integrity], monIOV))
    outcome |= (0x1 << Integrity);
  if(((taskList >> Laser) & 0x1) && writeLaser(meSets[Laser], monIOV))
    outcome |= (0x1 << Laser);
  if(((taskList >> Pedestal) & 0x1) && writePedestal(meSets[Pedestal], monIOV))
    outcome |= (0x1 << Pedestal);
  if(((taskList >> Presample) & 0x1) && writePresample(meSets[Presample], monIOV))
    outcome |= (0x1 << Presample);
  if(((taskList >> TestPulse) & 0x1) && writeTestPulse(meSets[TestPulse], monIOV))
    outcome |= (0x1 << TestPulse);
  if(((taskList >> Timing) & 0x1) && writeTiming(meSets[Timing], monIOV))
    outcome |= (0x1 << Timing);
  if(((taskList >> Led) & 0x1) && writeLed(meSets[Led], monIOV))
    outcome |= (0x1 << Led);
  if(((taskList >> RawData) & 0x1) && writeRawData(meSets[RawData], monIOV))
    outcome |= (0x1 << RawData);
  if(((taskList >> Occupancy) & 0x1) && writeOccupancy(meSets[Occupancy], monIOV))
    outcome |= (0x1 << Occupancy);

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

  executed_ = true;
}

DEFINE_FWK_MODULE(EcalCondDBWriter);
