#ifndef GlobalTriggerAnalyzer_L1GtAnalyzer_h
#define GlobalTriggerAnalyzer_L1GtAnalyzer_h

/**
 * \class L1GtAnalyzer
 * 
 * 
 * Description: test analyzer to illustrate various methods for L1 GT trigger.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// system include files
#include <memory>
#include <string>

// user include files

#include "DataFormats/Common/interface/ConditionsInEdm.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMaps.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtTriggerMenuLite.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1RetrieveL1Extra.h"

// class declaration

class L1GtAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  explicit L1GtAnalyzer(const edm::ParameterSet&);
  ~L1GtAnalyzer() override;

private:
  void beginJob() override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;

  /// analyze: decision and decision word
  ///   bunch cross in event BxInEvent = 0 - L1Accept event
  virtual void analyzeDecisionReadoutRecord(const edm::Event&, const edm::EventSetup&);

  /// analyze: usage of L1GtUtils
  void analyzeL1GtUtilsCore(const edm::Event&, const edm::EventSetup&);
  ///   for tests, use only one of the following methods
  void analyzeL1GtUtilsMenuLite(const edm::Event&, const edm::EventSetup&);
  void analyzeL1GtUtilsEventSetup(const edm::Event&, const edm::EventSetup&);
  void analyzeL1GtUtils(const edm::Event&, const edm::EventSetup&);

  /// full analysis of an algorithm or technical trigger
  void analyzeTrigger(const edm::Event&, const edm::EventSetup&);

  /// analyze: object map product
  virtual void analyzeObjectMap(const edm::Event&, const edm::EventSetup&);

  /// analyze: usage of L1GtTriggerMenuLite
  void analyzeL1GtTriggerMenuLite(const edm::Event&, const edm::EventSetup&);

  /// analyze: usage of ConditionsInEdm
  ///
  /// to be used in beginRun
  void analyzeConditionsInRunBlock(const edm::Run&, const edm::EventSetup&);
  /// to be used in beginLuminosityBlock
  void analyzeConditionsInLumiBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  /// to be used in analyze/produce/filter
  void analyzeConditionsInEventBlock(const edm::Event&, const edm::EventSetup&);

  /// print the output stream to the required output, given by m_printOutput
  void printOutput(std::ostringstream&);

  /// analyze each event: event loop over various code snippets
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  /// end section
  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;

  /// end of job
  void endJob() override;

private:
  // L1Extra collections
  L1RetrieveL1Extra m_retrieveL1Extra;

  /// print output
  int m_printOutput;

  /// enable / disable various analysis methods
  bool m_analyzeDecisionReadoutRecordEnable;
  //
  bool m_analyzeL1GtUtilsMenuLiteEnable;
  bool m_analyzeL1GtUtilsEventSetupEnable;
  bool m_analyzeL1GtUtilsEnable;
  bool m_analyzeTriggerEnable;
  //
  bool m_analyzeObjectMapEnable;
  //
  bool m_analyzeL1GtTriggerMenuLiteEnable;
  //
  bool m_analyzeConditionsInRunBlockEnable;
  bool m_analyzeConditionsInLumiBlockEnable;
  bool m_analyzeConditionsInEventBlockEnable;

private:
  /// input tags for GT DAQ product
  edm::InputTag m_l1GtDaqReadoutRecordInputTag;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_l1GtDaqReadoutRecordToken;

  /// input tags for GT lite product
  edm::InputTag m_l1GtRecordInputTag;

  /// input tags for GT object map collection L1GlobalTriggerObjectMapRecord
  edm::InputTag m_l1GtObjectMapTag;
  edm::EDGetTokenT<L1GlobalTriggerObjectMapRecord> m_l1GtObjectMapToken;

  /// input tags for GT object map collection L1GlobalTriggerObjectMaps
  edm::InputTag m_l1GtObjectMapsInputTag;
  edm::EDGetTokenT<L1GlobalTriggerObjectMaps> m_l1GtObjectMapsToken;

  /// input tag for muon collection from GMT
  edm::InputTag m_l1GmtInputTag;

  /// input tag for L1GtTriggerMenuLite
  edm::InputTag m_l1GtTmLInputTag;
  edm::EDGetTokenT<L1GtTriggerMenuLite> m_l1GtTmLToken;

  /// input tag for ConditionInEdm products
  edm::InputTag m_condInEdmInputTag;
  edm::EDGetTokenT<edm::ConditionsInRunBlock> m_condInRunToken;
  edm::EDGetTokenT<edm::ConditionsInLumiBlock> m_condInLumiToken;
  edm::EDGetTokenT<edm::ConditionsInEventBlock> m_condInEventToken;

  /// an algorithm trigger (name or alias) or a technical trigger name
  std::string m_nameAlgTechTrig;

  /// a condition in the algorithm trigger to test the object maps
  std::string m_condName;

  /// a bit number to retrieve the name and the alias
  unsigned int m_bitNumber;

  /// L1 configuration code for L1GtUtils
  unsigned int m_l1GtUtilsConfiguration;

  /// if true, use methods in L1GtUtils with the input tag for L1GtTriggerMenuLite
  /// from provenance
  bool m_l1GtTmLInputTagProv;

  /// if true, use methods in L1GtUtils with the given input tags
  /// for L1GlobalTriggerReadoutRecord and / or L1GlobalTriggerRecord from provenance
  bool m_l1GtRecordsInputTagProv;

  /// if true, configure (partially) L1GtUtils in beginRun using getL1GtRunCache
  bool m_l1GtUtilsConfigureBeginRun;

  /// expression to test the L1GtUtils methods to retrieve L1 trigger decisions,
  ///   prescale factors and masks for logical expressions
  std::string m_l1GtUtilsLogicalExpression;

private:
  L1GtUtils m_l1GtUtilsProv;
  L1GtUtils m_l1GtUtils;
  L1GtUtils::LogicalExpressionL1Results m_logicalExpressionL1ResultsProv;
  L1GtUtils::LogicalExpressionL1Results m_logicalExpressionL1Results;
};

#endif /*GlobalTriggerAnalyzer_L1GtAnalyzer_h*/
