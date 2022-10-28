#ifndef L1GtConfigProducers_L1GtPrescaleFactorsAndMasksTester_h
#define L1GtConfigProducers_L1GtPrescaleFactorsAndMasksTester_h

/**
 * \class L1GtPrescaleFactorsAndMasksTester
 * 
 * 
 * Description: test analyzer for L1 GT prescale factors and masks.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// system include files

// user include files
//   base class
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
class L1GtPrescaleFactors;
class L1GtTriggerMask;

class L1GtPrescaleFactorsAlgoTrigRcd;
class L1GtPrescaleFactorsTechTrigRcd;
class L1GtTriggerMaskAlgoTrigRcd;
class L1GtTriggerMaskTechTrigRcd;
class L1GtTriggerMaskVetoAlgoTrigRcd;
class L1GtTriggerMaskVetoTechTrigRcd;

// class declaration
class L1GtPrescaleFactorsAndMasksTester
    : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  // constructor
  explicit L1GtPrescaleFactorsAndMasksTester(const edm::ParameterSet&);

  struct Tokens {
    edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsAlgoTrigRcd> m_l1GtPfAlgo;
    edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsTechTrigRcd> m_l1GtPfTech;
    edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskAlgoTrigRcd> m_l1GtTmAlgo;
    edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskTechTrigRcd> m_l1GtTmTech;
    edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskVetoAlgoTrigRcd> m_l1GtTmVetoAlgo;
    edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskVetoTechTrigRcd> m_l1GtTmVetoTech;
  };

private:
  /// begin run
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  /// begin luminosity block
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;

  /// analyze
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  /// end luminosity block
  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;

  /// end run
  void endRun(const edm::Run&, const edm::EventSetup&) override;

private:
  /// retrieve all the relevant L1 trigger event setup records
  void retrieveL1EventSetup(const edm::EventSetup&, const Tokens&);

  /// print the requred records
  void printL1EventSetup();

private:
  /// input parameters

  /// analyze prescale factors, trigger masks and trigger veto masks, respectively
  bool m_testerPrescaleFactors;
  bool m_testerTriggerMask;
  bool m_testerTriggerVetoMask;

  /// retrieve the records in beginRun, beginLuminosityBlock, analyze, respectively
  bool m_retrieveInBeginRun;
  bool m_retrieveInBeginLuminosityBlock;
  bool m_retrieveInAnalyze;

  /// print the records in beginRun, beginLuminosityBlock, analyze, respectively
  bool m_printInBeginRun;
  bool m_printInBeginLuminosityBlock;
  bool m_printInAnalyze;

  /// print output
  int m_printOutput;

private:
  /// prescale factors
  const L1GtPrescaleFactors* m_l1GtPfAlgo;
  const L1GtPrescaleFactors* m_l1GtPfTech;

  /// trigger masks & veto masks
  const L1GtTriggerMask* m_l1GtTmAlgo;
  const L1GtTriggerMask* m_l1GtTmTech;

  const L1GtTriggerMask* m_l1GtTmVetoAlgo;
  const L1GtTriggerMask* m_l1GtTmVetoTech;

  Tokens m_run;
  Tokens m_lumi;
  Tokens m_event;
};

#endif /*L1GtConfigProducers_L1GtPrescaleFactorsAndMasksTester_h*/
