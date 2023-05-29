#ifndef GlobalTriggerAnalyzer_L1GtPatternGenerator_h
#define GlobalTriggerAnalyzer_L1GtPatternGenerator_h

/**
 * \class L1GtPatternGenerator
 * 
 * 
 * Description: A generator of pattern files for L1 GT hardware testing.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Thomas Themel - HEPHY Vienna
 * 
 *
 */

// system include files
#include <memory>
#include <string>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// class declaration
class L1GtPatternWriter;
class L1GtPatternMap;

class L1GtPatternGenerator : public edm::one::EDAnalyzer<> {
public:
  explicit L1GtPatternGenerator(const edm::ParameterSet&);
  ~L1GtPatternGenerator() override;

protected:
  void extractGlobalTriggerData(const edm::Event& iEvent, L1GtPatternMap& patterns);

private:
  /// analyze
  void beginJob() override;

  /// analyze each event
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  /// end of job
  void endJob() override;

  /** Post-processing for complex mapping of HF records to PSB values */
  void packHfRecords(const std::string& resultName, L1GtPatternMap& allPatterns);

  /** Post-processing for regional muon trigger inputs */
  static uint32_t packRegionalMuons(uint32_t rawValue);

  /** Post-processing for etMissing */
  static uint32_t packEtMiss(uint32_t rawValue);

private:
  /// input tag for GCT data
  edm::InputTag m_gctTag;

  /// input tag for GMT data
  edm::InputTag m_gmtTag;

  /// input tag for GT data
  edm::InputTag m_gtTag;

  /// input tags for regional muon data
  edm::InputTag m_dtTag;
  edm::InputTag m_cscTag;
  edm::InputTag m_rpcbTag;
  edm::InputTag m_rpcfTag;

  /// an algorithm and a condition in that algorithm to test the object maps
  std::string m_destPath;

  /// output file name
  std::string m_fileName;
  std::ofstream m_fileStream;

  /// formatting instructions

  std::string m_header;
  std::string m_footer;
  std::vector<std::string> m_columnNames;
  std::vector<uint32_t> m_columnLengths;
  std::vector<int> m_bx;
  std::vector<uint32_t> m_columnDefaults;
  bool m_debug;

  std::unique_ptr<L1GtPatternWriter> m_writer;
};

#endif /*GlobalTriggerAnalyzer_L1GtPatternGenerator_h*/
