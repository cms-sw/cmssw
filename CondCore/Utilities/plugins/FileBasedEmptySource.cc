#include "FWCore/Sources/interface/ProducerSourceBase.h"
#include "CondCore/CondDB/interface/Time.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <string>
#include <fstream>
#include <unistd.h>
namespace cond {
  class FileBasedEmptySource : public edm::ProducerSourceBase {
  public:
    FileBasedEmptySource(edm::ParameterSet const&, edm::InputSourceDescription const&);
    ~FileBasedEmptySource() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::Event& e) override;
    bool setRunAndEventInfo(edm::EventID& id,
                            edm::TimeValue_t& time,
                            edm::EventAuxiliary::ExperimentType& eType) override;
    void initialize(edm::EventID& id, edm::TimeValue_t& time, edm::TimeValue_t& interval) override;

  private:
    unsigned int m_interval;
    unsigned long long m_eventId;
    unsigned int m_eventsPerLumi;
    std::string m_pathForLastLumiFile;
    unsigned int m_currentRun;
    unsigned int m_currentLumi;
    boost::posix_time::ptime m_currentLumiTime;
  };
}  // namespace cond

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
//#include "DataFormats/Provenance/interface/EventID.h"

namespace cond {
  //allowed parameters: firstRun, firstTime, lastRun, lastTime,
  //common paras: timetype,interval
  FileBasedEmptySource::FileBasedEmptySource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc)
      : edm::ProducerSourceBase(pset, desc, true),
        m_interval(pset.getParameter<unsigned int>("interval")),
        m_eventId(0),
        m_eventsPerLumi(pset.getUntrackedParameter<unsigned int>("numberEventsInLuminosityBlock")),
        m_pathForLastLumiFile(pset.getParameter<std::string>("pathForLastLumiFile")),
        m_currentRun(0),
        m_currentLumi(0),
        m_currentLumiTime() {}

  FileBasedEmptySource::~FileBasedEmptySource() {}

  void FileBasedEmptySource::produce(edm::Event&) {}

  bool FileBasedEmptySource::setRunAndEventInfo(edm::EventID& id,
                                                edm::TimeValue_t& time,
                                                edm::EventAuxiliary::ExperimentType&) {
    cond::Time_t lastLumi = cond::time::MIN_VAL;
    {
      std::ifstream lastLumiFile(m_pathForLastLumiFile);
      if (lastLumiFile) {
        lastLumiFile >> lastLumi;
      } else {
        std::cout << "Error: last lumi file can't be read." << std::endl;
        return false;
      }
    }
    auto t = cond::time::unpack(lastLumi);
    unsigned int runId = t.first;
    unsigned int lumiId = t.second;
    //std::cout <<"###### setRunAndEventInfo Run: "<<runId<<" lumi: "<<lumiId<<std::endl;
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    if (runId == m_currentRun && lumiId == m_currentLumi) {
      m_eventId += 1;
      if (m_eventId >= m_eventsPerLumi) {
        return false;
      }
    } else {
      m_currentRun = runId;
      m_currentLumi = lumiId;
      m_currentLumiTime = now;
      m_eventId = 1;
    }
    std::cout << "###### setRunAndEventInfo Run: " << runId << " lumi: " << lumiId << " event id: " << m_eventId
              << " time:" << boost::posix_time::to_simple_string(now) << std::endl;
    time = cond::time::from_boost(now);
    id = edm::EventID(runId, lumiId, m_eventId);
    usleep(20000);
    return true;
  }

  void FileBasedEmptySource::initialize(edm::EventID& id, edm::TimeValue_t& time, edm::TimeValue_t& interval) {
    cond::Time_t lastLumi = cond::time::MIN_VAL;
    {
      std::ifstream lastLumiFile(m_pathForLastLumiFile);
      if (lastLumiFile) {
        lastLumiFile >> lastLumi;
      } else {
        std::cout << "Error: last lumi file can't be read." << std::endl;
        return;
      }
    }
    m_eventId = 0;
    auto t = cond::time::unpack(lastLumi);
    unsigned int runId = t.first;
    unsigned int lumiId = t.second;
    std::cout << "###### initialize Run: " << runId << " lumi: " << lumiId << std::endl;
    m_currentRun = runId;
    m_currentLumi = lumiId;
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    m_currentLumiTime = now;
    time = cond::time::from_boost(now);
    id = edm::EventID(runId, lumiId, m_eventId);
    interval = m_interval;
  }

  void FileBasedEmptySource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("Creates runs, lumis and events containing no products.");
    ProducerSourceBase::fillDescription(desc);

    //desc.add<unsigned int>("firstRunnumber")->setComment("The first run number to use");
    //desc.add<unsigned int>("lastRunnumber")->setComment("The last run number to use");
    //desc.add<unsigned int>("firstLumi")->setComment("The first lumi id to use");
    //desc.add<unsigned int>("lastLumi")->setComment("The last lumi id to use");
    //desc.add<unsigned int>("maxLumiInRun");
    //desc.add<std::string>("startTime");
    //desc.add<std::string>("endTime");
    desc.add<unsigned int>("interval");
    desc.add<unsigned int>("maxEvents");
    desc.add<std::string>("pathForLastLumiFile");
    descriptions.add("source", desc);
  }

}  // namespace cond

#include "FWCore/Framework/interface/InputSourceMacros.h"
using cond::FileBasedEmptySource;

DEFINE_FWK_INPUT_SOURCE(FileBasedEmptySource);
