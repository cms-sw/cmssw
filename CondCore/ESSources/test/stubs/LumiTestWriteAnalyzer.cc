
/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
//#include <map>
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
  class LumiTestWriteAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit LumiTestWriteAnalyzer(edm::ParameterSet const& p)
        : m_connectionString(p.getUntrackedParameter<std::string>("connectionString")),
          m_tagName(p.getUntrackedParameter<std::string>("tagName")),
          m_run(p.getUntrackedParameter<unsigned int>("runNumber")),
          m_NLumi(p.getUntrackedParameter<unsigned int>("numberOfLumis")),
          m_iovSize(p.getUntrackedParameter<unsigned int>("iovSize")) {}
    explicit LumiTestWriteAnalyzer(int i) {}
    virtual ~LumiTestWriteAnalyzer() {}
    virtual void beginJob() override;
    virtual void analyze(edm::Event const&, edm::EventSetup const&) override {}

  private:
    std::string m_connectionString;
    std::string m_tagName;
    unsigned int m_run;
    unsigned int m_NLumi;
    unsigned int m_iovSize;
  };

  void LumiTestWriteAnalyzer::beginJob() {
    cond::persistency::ConnectionPool pool;
    pool.setMessageVerbosity(coral::Debug);
    auto session = pool.createSession(m_connectionString, true);
    session.transaction().start(false);
    cond::persistency::IOVEditor editor;
    if (!session.existsDatabase() || !session.existsIov(m_tagName)) {
      editor = session.createIov<BeamSpotObjects>(m_tagName, cond::lumiid);
      editor.setDescription("Read/Write Test");
    } else {
      editor = session.editIov(m_tagName);
    }
    size_t i = 0;
    for (size_t lumiId = 1; lumiId < m_NLumi; lumiId += m_iovSize) {
      BeamSpotObjects mybeamspot;
      mybeamspot.setPosition(0.053, 0.1, 0.13);
      mybeamspot.setSigmaZ(3.8 + i);
      mybeamspot.setType(int(lumiId));
      auto payloadId = session.storePayload(mybeamspot);
      auto since = cond::time::lumiTime(m_run, lumiId);
      editor.insert(since, payloadId);
      i++;
    }
    editor.flush();
    session.transaction().commit();
  }

  DEFINE_FWK_MODULE(LumiTestWriteAnalyzer);
}  // namespace edmtest
