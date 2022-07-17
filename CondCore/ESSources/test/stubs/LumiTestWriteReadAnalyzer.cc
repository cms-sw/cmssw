
/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include <fstream>

using namespace std;

namespace edmtest {
  class LumiTestWriteReadAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit LumiTestWriteReadAnalyzer(edm::ParameterSet const& p);
    explicit LumiTestWriteReadAnalyzer(int i) {}
    virtual ~LumiTestWriteReadAnalyzer() {}

    virtual void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
    const edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> m_token;
  };

  LumiTestWriteReadAnalyzer::LumiTestWriteReadAnalyzer(edm::ParameterSet const&)
      : m_token(esConsumes<BeamSpotObjects, BeamSpotObjectsRcd>()) {}

  void LumiTestWriteReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    static constexpr const char* const MSGSOURCE = "LumiTestWriteReadAnalyzer:";
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("BeamSpotObjectsRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogError(MSGSOURCE) << "Record \"BeamSpotObjectsRcd\" does not exist ";
    }
    auto const& payload = context.getData(m_token);
    edm::LogInfo(MSGSOURCE) << "Event " << e.id().event() << " Run " << e.id().run() << " Lumi "
                            << e.id().luminosityBlock() << " Time " << e.time().value() << " LumiTestPayload id "
                            << payload.beamType() << std::endl;
    unsigned int target = e.id().luminosityBlock();
    unsigned int found = payload.beamType();
    if (target != found) {
      boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
      std::stringstream msg;
      msg << "On time " << boost::posix_time::to_iso_extended_string(now) << " Target " << target << "; found "
          << found;
      edm::LogError(MSGSOURCE) << msg.str();
    } else {
      edm::LogInfo(MSGSOURCE) << "Read target payload was ok." << std::endl;
    }
  }
  DEFINE_FWK_MODULE(LumiTestWriteReadAnalyzer);
}  // namespace edmtest
