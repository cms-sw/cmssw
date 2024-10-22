/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/05/14 11:43:08 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigListRcd.h"
#include "CondCore/CondDB/interface/KeyList.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//-----------------------
// This Class' Header --
//-----------------------
class DTKeyedConfigDBDump : public edm::one::EDAnalyzer<> {
public:
  /** Constructor                                                                                                
   */
  explicit DTKeyedConfigDBDump(const edm::ParameterSet& ps);

  /** Destructor                                                                                                 
   */
  ~DTKeyedConfigDBDump() override;

  /** Operations 
   */
  ///
  void beginJob() override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::ESGetToken<cond::persistency::KeyList, DTKeyedConfigListRcd> perskeylistToken_;
};
//-------------------
// Initializations --
//-------------------

//----------------
// Constructors --
//----------------
DTKeyedConfigDBDump::DTKeyedConfigDBDump(const edm::ParameterSet& ps) : perskeylistToken_(esConsumes()) {}

//--------------
// Destructor --
//--------------
DTKeyedConfigDBDump::~DTKeyedConfigDBDump() {}

//--------------
// Operations --
//--------------
void DTKeyedConfigDBDump::beginJob() { return; }

void DTKeyedConfigDBDump::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::eventsetup::EventSetupRecordKey recordKey(
      edm::eventsetup::EventSetupRecordKey::TypeTag::findType("DTKeyedConfigListRcd"));
  if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    //record not found
    edm::LogWarning("DTKeyedConfigDBDump") << "Record \"DTKeyedConfigListRcd "
                                           << "\" does not exist " << std::endl;
  }
  cond::persistency::KeyList const* kp = &iSetup.getData(perskeylistToken_);
  edm::LogInfo("DTKeyedConfigDBDump") << "now load and get" << std::endl;
  auto pkc = kp->getUsingKey<DTKeyedConfig>(999999999);
  edm::LogInfo("DTKeyedConfigDBDump") << "now check" << std::endl;
  if (pkc.get())
    edm::LogInfo("DTKeyedConfigDBDump") << pkc->getId() << " " << *(pkc->dataBegin()) << std::endl;
  else
    edm::LogInfo("DTKeyedConfigDBDump") << "not found" << std::endl;
  edm::LogInfo("DTKeyedConfigDBDump") << std::endl;
  return;
}

DEFINE_FWK_MODULE(DTKeyedConfigDBDump);
