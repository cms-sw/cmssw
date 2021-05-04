/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/05/14 11:43:08 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTKeyedConfigDBDump.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigListRcd.h"
#include "CondCore/CondDB/interface/KeyList.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <memory>

//-------------------
// Initializations --
//-------------------

//----------------
// Constructors --
//----------------
DTKeyedConfigDBDump::DTKeyedConfigDBDump(const edm::ParameterSet& ps) {}

//--------------
// Destructor --
//--------------
DTKeyedConfigDBDump::~DTKeyedConfigDBDump() {}

//--------------
// Operations --
//--------------
void DTKeyedConfigDBDump::beginJob() { return; }

void DTKeyedConfigDBDump::analyze(const edm::Event& e, const edm::EventSetup& c) {
  edm::eventsetup::EventSetupRecordKey recordKey(
      edm::eventsetup::EventSetupRecordKey::TypeTag::findType("DTKeyedConfigListRcd"));
  if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    //record not found
    std::cout << "Record \"DTKeyedConfigListRcd "
              << "\" does not exist " << std::endl;
  }
  edm::ESHandle<cond::persistency::KeyList> klh;
  std::cout << "got eshandle" << std::endl;
  c.get<DTKeyedConfigListRcd>().get(klh);
  std::cout << "got context" << std::endl;
  cond::persistency::KeyList const& kl = *klh.product();
  cond::persistency::KeyList const* kp = &kl;
  std::cout << "now load and get" << std::endl;
  auto pkc = kp->getUsingKey<DTKeyedConfig>(999999999);
  std::cout << "now check" << std::endl;
  if (pkc.get())
    std::cout << pkc->getId() << " " << *(pkc->dataBegin()) << std::endl;
  else
    std::cout << "not found" << std::endl;
  std::cout << std::endl;
  return;
}

DEFINE_FWK_MODULE(DTKeyedConfigDBDump);
