///////////////////////////////////////////////////////////////////////
///
/// Module to write trigger bit mappings (AlCaRecoTriggerBits) to DB.
///
///////////////////////////////////////////////////////////////////////

#include <string>
#include <map>
#include <vector>

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Database
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// What I want to write:
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
// <- ??? Or this one:
// #include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"


class  AlCaRecoTriggerBitsRcdWrite : public edm::EDAnalyzer {
public:
  explicit  AlCaRecoTriggerBitsRcdWrite(const edm::ParameterSet& cfg);
  ~AlCaRecoTriggerBitsRcdWrite() {}
  
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);
  
private:
  unsigned int nEventCalls_;
  const unsigned int firstRunIOV_;
  const int lastRunIOV_;
  const std::vector<edm::ParameterSet> triggerLists_;
};

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

AlCaRecoTriggerBitsRcdWrite::AlCaRecoTriggerBitsRcdWrite(const edm::ParameterSet& cfg)
  : nEventCalls_(0), 
    firstRunIOV_(cfg.getParameter<unsigned int>("firstRunIOV")),
    lastRunIOV_(cfg.getParameter<int>("lastRunIOV")),
    triggerLists_(cfg.getParameter<std::vector<edm::ParameterSet> >("triggerLists"))
{
}
  
///////////////////////////////////////////////////////////////////////
void AlCaRecoTriggerBitsRcdWrite::analyze(const edm::Event& evt, const edm::EventSetup& iSetup)
{
  if (nEventCalls_++ > 0) { // postfix increment!
    edm::LogWarning("BadConfig")
      << "@SUB=analyze" << "Writing to DB to be done only once, set\n"
      << "'process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))'\n"
      << " next time. But your writing is fine.)";
    return;
  }
  
  AlCaRecoTriggerBits *bitsToWrite = new AlCaRecoTriggerBits; // auto_ptr?
  // loop on PSets, each containing the key (filter name) and a vstring with triggers
  for (std::vector<edm::ParameterSet>::const_iterator iSet = triggerLists_.begin();
       iSet != triggerLists_.end(); ++iSet) {
    
    const std::vector<std::string> paths
      = iSet->getParameter<std::vector<std::string> >("hltPaths");
    // We must avoid a map<string,vector<string> > in DB for performance reason,
    // so we have to merge the paths into one string that will be decoded when needed:
    const std::string mergedPaths = bitsToWrite->compose(paths);
    
    const std::string filter(iSet->getParameter<std::string>("listName"));
    if (bitsToWrite->m_alcarecoToTrig.find(filter) != bitsToWrite->m_alcarecoToTrig.end()) {
      throw cms::Exception("BadConfig") << "List name '" << filter << "' appears twice in "
                                        << "input configuration, would overwrite first entry\n";
    }
    bitsToWrite->m_alcarecoToTrig[filter] = mergedPaths;
  }
  
  edm::LogInfo("") << "Uploading to the database...";
  
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  
  if (!poolDbService.isAvailable()) {
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available.";
  }

  // ownership of bitsToWrite transferred
  // FIXME: Have to check that timetype is run number! How?
  const std::string recordName("AlCaRecoTriggerBitsRcd");
  if (poolDbService->isNewTagRequest(recordName)) { // tag not yet existing
    // lastRunIOV_ = -1 means infinity:
    const cond::Time_t lastRun = (lastRunIOV_ < 0 ? poolDbService->endOfTime() : lastRunIOV_);
    poolDbService->createNewIOV(bitsToWrite, firstRunIOV_, lastRun, recordName);
  } else { // tag exists, can only append
    if (lastRunIOV_ >= 0) {
      throw cms::Exception("BadConfig") << "Tag already exists, can only append until infinity,"
                                        << " but lastRunIOV = " << lastRunIOV_ << ".\n";
    }
    poolDbService->appendSinceTime(bitsToWrite, firstRunIOV_, recordName);
  }  
  
  edm::LogInfo("") << "...done for runs " << firstRunIOV_ << " to " << lastRunIOV_ 
                   << " (< 0 meaning infinity)!";
}


//define this as a plug-in
DEFINE_FWK_MODULE(AlCaRecoTriggerBitsRcdWrite);
