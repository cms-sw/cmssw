#include "DQM/EcalMonitorDbModule/interface/EcalCondDBReader.h"

#include "DQM/EcalCommon/interface/MESetUtils.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalCondDBReader::EcalCondDBReader(edm::ParameterSet const& _ps) :
  db_(nullptr),
  monIOV_(),
  worker_(nullptr),
  formula_(_ps.getUntrackedParameter<std::string>("formula")),
  meSet_(ecaldqm::createMESet(_ps.getUntrackedParameterSet("plot"))),
  verbosity_(_ps.getUntrackedParameter<int>("verbosity"))
{
  std::string table(_ps.getUntrackedParameter<std::string>("table"));
  edm::ParameterSet const& workerParams(_ps.getUntrackedParameterSet("workerParams"));

  if(table == "CrystalConsistency") worker_ = new ecaldqm::CrystalConsistencyReader(workerParams);
  if(table == "TTConsistency") worker_ = new ecaldqm::TTConsistencyReader(workerParams);
  if(table == "MemChConsistency") worker_ = new ecaldqm::MemChConsistencyReader(workerParams);
  if(table == "MemTTConsistency") worker_ = new ecaldqm::MemTTConsistencyReader(workerParams);
  if(table == "LaserBlue") worker_ = new ecaldqm::LaserBlueReader(workerParams);
  if(table == "TimingLaserBlueCrystal") worker_ = new ecaldqm::TimingLaserBlueCrystalReader(workerParams);
  if(table == "PNBlue") worker_ = new ecaldqm::PNBlueReader(workerParams);
  if(table == "LaserGreen") worker_ = new ecaldqm::LaserGreenReader(workerParams);
  if(table == "TimingLaserGreenCrystal") worker_ = new ecaldqm::TimingLaserGreenCrystalReader(workerParams);
  if(table == "PNGreen") worker_ = new ecaldqm::PNGreenReader(workerParams);
  if(table == "LaserIRed") worker_ = new ecaldqm::LaserIRedReader(workerParams);
  if(table == "TimingLaserIRedCrystal") worker_ = new ecaldqm::TimingLaserIRedCrystalReader(workerParams);
  if(table == "PNIRed") worker_ = new ecaldqm::PNIRedReader(workerParams);
  if(table == "LaserRed") worker_ = new ecaldqm::LaserRedReader(workerParams);
  if(table == "TimingLaserRedCrystal") worker_ = new ecaldqm::TimingLaserRedCrystalReader(workerParams);
  if(table == "PNRed") worker_ = new ecaldqm::PNRedReader(workerParams);
  if(table == "Pedestals") worker_ = new ecaldqm::PedestalsReader(workerParams);
  if(table == "PNPed") worker_ = new ecaldqm::PNPedReader(workerParams);
  if(table == "PedestalsOnline") worker_ = new ecaldqm::PedestalsOnlineReader(workerParams);
  if(table == "TestPulse") worker_ = new ecaldqm::TestPulseReader(workerParams);
  if(table == "PulseShape") worker_ = new ecaldqm::PulseShapeReader(workerParams);
  if(table == "PNMGPA") worker_ = new ecaldqm::PNMGPAReader(workerParams);
  if(table == "TimingCrystal") worker_ = new ecaldqm::TimingCrystalReader(workerParams);
  if(table == "Led1") worker_ = new ecaldqm::Led1Reader(workerParams);
  if(table == "TimingLed1Crystal") worker_ = new ecaldqm::TimingLed1CrystalReader(workerParams);
  if(table == "Led2") worker_ = new ecaldqm::Led2Reader(workerParams);
  if(table == "TimingLed2Crystal") worker_ = new ecaldqm::TimingLed2CrystalReader(workerParams);
  if(table == "Occupancy") worker_ = new ecaldqm::OccupancyReader(workerParams);

  if(!worker_)
    throw cms::Exception("Configuration") << "Invalid worker type";

  std::string DBName(_ps.getUntrackedParameter<std::string>("DBName"));
  std::string hostName(_ps.getUntrackedParameter<std::string>("hostName"));
  int hostPort(_ps.getUntrackedParameter<int>("hostPort"));
  std::string userName(_ps.getUntrackedParameter<std::string>("userName"));
  std::string password(_ps.getUntrackedParameter<std::string>("password"));

  std::unique_ptr<EcalCondDBInterface> db(nullptr);

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << "Establishing DB connection";

  try{
    db = std::unique_ptr<EcalCondDBInterface>(new EcalCondDBInterface(DBName, userName, password));
  }
  catch(std::runtime_error& re){
    if(!hostName.empty()){
      try{
        db = std::unique_ptr<EcalCondDBInterface>(new EcalCondDBInterface(hostName, DBName, userName, password, hostPort));
      }
      catch(std::runtime_error& re2){
        throw cms::Exception("DBError") << re2.what();
      }
    }
    else
      throw cms::Exception("DBError") << re.what();
  }

  db_ = db.release();

  std::string location(_ps.getUntrackedParameter<std::string>("location"));
  int runNumber(_ps.getUntrackedParameter<int>("runNumber"));
  std::string monRunGeneralTag(_ps.getUntrackedParameter<std::string>("monRunGeneralTag"));

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << "Initializing DB entry";

  RunTag runTag;

  try{
    runTag = db_->fetchRunIOV(location, runNumber).getRunTag();
  }
  catch(std::exception&){
    edm::LogError("EcalDQM") << "Cannot fetch RunIOV for location=" << location << " runNumber=" << runNumber;
    throw;
  }

  MonVersionDef versionDef;
  versionDef.setMonitoringVersion("test01"); // the only mon_ver in mon_version_def table as of September 2012
  MonRunTag monTag;
  monTag.setMonVersionDef(versionDef);
  monTag.setGeneralTag(monRunGeneralTag);

  try{
    monIOV_ = db_->fetchMonRunIOV(&runTag, &monTag, runNumber, 1);
  }
  catch(std::runtime_error& e){
    edm::LogError("EcalDQM") << "Cannot fetch MonRunIOV for location=" << location << " runNumber=" << runNumber << " monVersion=test01 monRunGeneralTag=" << monRunGeneralTag;
    throw;
  }

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << " Done.";
}

EcalCondDBReader::~EcalCondDBReader()
{
  delete worker_;
  delete meSet_;
}
 
void
EcalCondDBReader::dqmEndJob(DQMStore::IBooker& _ibooker, DQMStore::IGetter&)
{
  meSet_->book(_ibooker);

  std::map<DetId, double> values(worker_->run(db_, monIOV_, formula_));
  for(std::map<DetId, double>::const_iterator vItr(values.begin()); vItr != values.end(); ++vItr)
    meSet_->setBinContent(vItr->first, vItr->second);
}
