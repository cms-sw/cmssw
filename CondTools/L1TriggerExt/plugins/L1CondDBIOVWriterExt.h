#ifndef CondTools_L1TriggerExt_L1CondDBIOVWriterExt_h
#define CondTools_L1TriggerExt_L1CondDBIOVWriterExt_h
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondTools/L1TriggerExt/interface/DataWriterExt.h"

class L1CondDBIOVWriterExt : public edm::EDAnalyzer {
public:
  explicit L1CondDBIOVWriterExt(const edm::ParameterSet&);
  ~L1CondDBIOVWriterExt() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  l1t::DataWriterExt m_writer;
  ///      l1t::DataWriter m_writer ;

  std::string m_tscKey, m_rsKey;

  // List of record@type, used only for objects not tied to TSC key.
  // Otherwise, list of records comes from L1TriggerKeyExt.
  std::vector<std::string> m_recordTypes;

  // When true, set IOVs for objects not tied to the TSC key.  The records
  // and objects to be updated are given in the toPut parameter, and
  // m_tscKey is taken to be a common key for all the toPut objects, not
  // the TSC key.  The IOV for L1TriggerKeyExt is not updated when
  // m_ignoreTriggerKey = true.
  bool m_ignoreTriggerKey;

  bool m_logKeys;

  bool m_logTransactions;

  bool m_forceUpdate;
};

#endif
