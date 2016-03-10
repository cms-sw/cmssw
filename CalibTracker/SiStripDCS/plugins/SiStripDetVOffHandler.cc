#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <sstream>

#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "CalibTracker/SiStripDCS/interface/SiStripDetVOffBuilder.h"


class SiStripDetVOffHandler : public edm::EDAnalyzer {
public:
  explicit SiStripDetVOffHandler(const edm::ParameterSet& iConfig );
  virtual ~SiStripDetVOffHandler();
  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void endJob();

private:
  cond::persistency::ConnectionPool m_connectionPool;
  std::string m_condDb;
  std::string m_localCondDbFile;
  std::string m_targetTag;

  bool getLastIOVFromDB_;
  int maxTimeBeforeNewIOV_;

  std::vector< std::pair<SiStripDetVOff*,cond::Time_t> > newPayloads;
  edm::Service<SiStripDetVOffBuilder> modHVBuilder;
};

SiStripDetVOffHandler::SiStripDetVOffHandler(const edm::ParameterSet& iConfig):
    m_connectionPool(),
    m_condDb( iConfig.getParameter< std::string >("conditionDatabase") ),
    m_localCondDbFile( iConfig.getParameter< std::string >("condDbFile") ),
    m_targetTag( iConfig.getParameter< std::string >("targetTag") ),
    getLastIOVFromDB_( iConfig.getUntrackedParameter< bool >("getLastIOVFromDB", true) ),
    maxTimeBeforeNewIOV_( iConfig.getUntrackedParameter< int >("maxTimeBeforeNewIOV", 86400) ){
  m_connectionPool.setParameters( iConfig.getParameter<edm::ParameterSet>("DBParameters")  );
  m_connectionPool.configure();
  // get last IOV from local sqlite file if "conditionDatabase" is empty
  if (m_condDb.empty()) m_condDb = m_localCondDbFile;
}

SiStripDetVOffHandler::~SiStripDetVOffHandler() {
}

void SiStripDetVOffHandler::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  // get last payload from condDb
  cond::Time_t lastIov = 0;
  boost::shared_ptr<SiStripDetVOff> lastPayload;

  if (getLastIOVFromDB_){
    edm::LogInfo("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::" << __func__ << "] "
        << "Retrieve last IOV from " << m_condDb;
    cond::persistency::Session condDbSession = m_connectionPool.createSession( m_condDb );
    condDbSession.transaction().start( true );
    cond::persistency::IOVProxy iovProxy = condDbSession.readIov( m_targetTag );
    lastIov = iovProxy.getLast().since; // move to LastValidatedTime?
    cond::Hash lastPayloadHash = iovProxy.getLast().payloadId;
    lastPayload = condDbSession.fetchPayload<SiStripDetVOff>( lastPayloadHash );
    condDbSession.transaction().commit();
    edm::LogInfo("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::" << __func__ << "] "
        << " ... last IOV: " << lastIov << " , " << "last Payload: " << lastPayloadHash;
  }

  // build the object!
  newPayloads.clear();
  modHVBuilder->setLastSiStripDetVOff( lastPayload.get(), lastIov );
  modHVBuilder->BuildDetVOffObj();
  newPayloads = modHVBuilder->getModulesVOff();
  edm::LogInfo("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::" << __func__ << "] "
      << "Finished building " << newPayloads.size() << " new payloads.";

  // write the payloads to sqlite file
  edm::LogInfo("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::" << __func__ << "] "
      << "Write new payloads to sqlite file " << m_localCondDbFile;
  cond::persistency::Session localFileSession = m_connectionPool.createSession( m_localCondDbFile, true );
  localFileSession.transaction().start( false );
  if ( newPayloads.size() == 1 && *newPayloads[0].first == *lastPayload ){
    // if no HV/LV transition was found in this period
    edm::LogInfo("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::" << __func__ << "] "
        << "No HV/LV transition was found from PVSS query.";
    bool forceNewIOV = true;
    if (maxTimeBeforeNewIOV_ < 0)
      forceNewIOV = false;
    else {
      auto deltaT = cond::time::to_boost(newPayloads[0].second) - cond::time::to_boost(lastIov);
      forceNewIOV = deltaT.total_seconds() > maxTimeBeforeNewIOV_;
    }
    if ( !forceNewIOV ){
      newPayloads.erase(newPayloads.begin());
      edm::LogInfo("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::" << __func__ << "] "
          << " ... No payload transfered.";
    }else {
      edm::LogInfo("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::" << __func__ << "] "
          << " ... The last IOV is too old. Will start a new IOV from " << newPayloads[0].second
          << "(" << boost::posix_time::to_simple_string(cond::time::to_boost(newPayloads[0].second)) << ") with the same payload.";
    }
  }

  cond::persistency::IOVEditor iovEditor;
  if (localFileSession.existsDatabase() && localFileSession.existsIov(m_targetTag)){
    edm::LogWarning("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::" << __func__ << "] "
        << "IOV of tag " << m_targetTag << " already exists in sqlite file " << m_localCondDbFile;
    iovEditor = localFileSession.editIov(m_targetTag);
  }else {
    iovEditor = localFileSession.createIov<SiStripDetVOff>( m_targetTag, cond::timestamp );
    iovEditor.setDescription( "New IOV" );
  }
  for (const auto &payload : newPayloads){
    cond::Hash thePayloadHash = localFileSession.storePayload<SiStripDetVOff>( *payload.first );
    iovEditor.insert( payload.second, thePayloadHash );
  }
  iovEditor.flush();
  localFileSession.transaction().commit();

  edm::LogInfo("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::" << __func__ << "] "
      << newPayloads.size() << " payloads written to sqlite file.";

}

void SiStripDetVOffHandler::endJob() {
}

// -------------------------------------------------------
DEFINE_FWK_MODULE(SiStripDetVOffHandler);
