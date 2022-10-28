#ifndef GlobalTriggerAnalyzer_L1GtPackUnpackAnalyzer_h
#define GlobalTriggerAnalyzer_L1GtPackUnpackAnalyzer_h

/**
 * \class L1GtPackUnpackAnalyzer
 * 
 * 
 * Description: pack - unpack validation for L1 GT DAQ record.  
 *
 * Implementation:
 *    Pack (DigiToRaw) and unpack (RawToDigi) a L1 GT DAQ record.
 *    Compare the initial DAQ record with the final one and print them if 
 *    they are different.
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// class declaration

class L1GtPackUnpackAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit L1GtPackUnpackAnalyzer(const edm::ParameterSet&);
  ~L1GtPackUnpackAnalyzer() override;

private:
  void beginJob() override;

  /// GT comparison
  virtual void analyzeGT(const edm::Event&, const edm::EventSetup&);

  /// GMT comparison
  virtual void analyzeGMT(const edm::Event&, const edm::EventSetup&);

  /// analyze each event
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  /// end of job
  void endJob() override;

private:
  /// input tag for the initial GT DAQ record:
  edm::InputTag m_initialDaqGtInputTag;

  /// input tag for the initial GMT readout collection:
  edm::InputTag m_initialMuGmtInputTag;

  /// input tag for the final GT DAQ and GMT records:
  edm::InputTag m_finalGtGmtInputTag;
};

#endif /*GlobalTriggerAnalyzer_L1GtPackUnpackAnalyzer_h*/
