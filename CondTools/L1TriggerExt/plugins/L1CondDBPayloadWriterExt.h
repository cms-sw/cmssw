#ifndef CondTools_L1TriggerExt_L1CondDBPayloadWriterExt_h
#define CondTools_L1TriggerExt_L1CondDBPayloadWriterExt_h
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondTools/L1TriggerExt/interface/DataWriterExt.h"

class L1CondDBPayloadWriterExt : public edm::EDAnalyzer {
public:
  explicit L1CondDBPayloadWriterExt(const edm::ParameterSet&);
  ~L1CondDBPayloadWriterExt() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  l1t::DataWriterExt m_writer;
  // std::string m_tag ; // tag is known by PoolDBOutputService

  // set to false to write config data without valid TSC key
  bool m_writeL1TriggerKeyExt;

  // set to false to write config data only
  bool m_writeConfigData;

  // substitute new payload tokens for existing keys in L1TriggerKeyListExt
  bool m_overwriteKeys;

  bool m_logTransactions;

  // if true, do not retrieve L1TriggerKeyListExt from EventSetup
  bool m_newL1TriggerKeyListExt;
};

#endif
