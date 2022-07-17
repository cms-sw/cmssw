
/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>

#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

namespace edmtest {
  class LumiTestReadAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit LumiTestReadAnalyzer(edm::ParameterSet const& p)
        : theBSToken_(esConsumes()),
          m_processId(p.getUntrackedParameter<std::string>("processId")),
          m_pathForLastLumiFile(p.getUntrackedParameter<std::string>("lastLumiFile", "")),
          m_pathForErrorFile("") {
      std::string pathForErrorFolder = p.getUntrackedParameter<std::string>("pathForErrorFile");
      m_pathForErrorFile = pathForErrorFolder + "/lumi_read_" + m_processId + ".txt";
    }
    explicit LumiTestReadAnalyzer(int i) {}
    virtual ~LumiTestReadAnalyzer() = default;
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    const edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> theBSToken_;
    std::string m_processId;
    std::string m_pathForLastLumiFile;
    std::string m_pathForErrorFile;
  };

  void LumiTestReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    static constexpr const char* const MSGSOURCE = "LumiTestReadAnalyzer:";
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("BeamSpotObjectsRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogError(MSGSOURCE) << "Record \"BeamSpotObjectsRcd\" does not exist ";
    }
    auto const& payload = &context.getData(theBSToken_);
    edm::LogInfo(MSGSOURCE) << "Event " << e.id().event() << " Run " << e.id().run() << " Lumi "
                            << e.id().luminosityBlock() << " Time " << e.time().value() << " LumiTestPayload id "
                            << payload->beamType() << std::endl;
    //cond::Time_t target = cond::time::lumiTime( e.id().run(), e.id().luminosityBlock());
    unsigned int target = e.id().luminosityBlock();
    unsigned int found = payload->beamType();
    if (target != found) {
      boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
      std::stringstream msg;
      msg << "On time " << boost::posix_time::to_iso_extended_string(now) << " Target " << target << "; found "
          << found;
      edm::LogWarning(MSGSOURCE) << msg.str();
      edm::LogPrint("LumiTestReadAnalyzer") << "ERROR ( process " << m_processId << " ) : " << msg.str();
      edm::LogPrint("LumiTestReadAnalyzer") << "### dumping in file " << m_pathForErrorFile;
      {
        std::ofstream errorFile(m_pathForErrorFile, std::ios_base::app);
        errorFile << msg.str() << std::endl;
      }
      //throw std::runtime_error( msg.str() );
    } else {
      edm::LogPrint("LumiTestReadAnalyzer") << "Info: read was ok.";
    }
  }
  DEFINE_FWK_MODULE(LumiTestReadAnalyzer);
}  // namespace edmtest
