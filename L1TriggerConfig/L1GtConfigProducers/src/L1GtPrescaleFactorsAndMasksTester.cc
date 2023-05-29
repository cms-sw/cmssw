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

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtPrescaleFactorsAndMasksTester.h"

// system include files
#include <iomanip>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"

#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoTechTrigRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// forward declarations

namespace {
  template <edm::Transition iTrans>
  L1GtPrescaleFactorsAndMasksTester::Tokens tokens(edm::ConsumesCollector iCC,
                                                   bool prescales,
                                                   bool masks,
                                                   bool vetoMasks) {
    L1GtPrescaleFactorsAndMasksTester::Tokens tokens;
    if (prescales) {
      tokens.m_l1GtPfAlgo = iCC.esConsumes<iTrans>();
      tokens.m_l1GtPfTech = iCC.esConsumes<iTrans>();
    }
    if (masks) {
      tokens.m_l1GtTmAlgo = iCC.esConsumes<iTrans>();
      tokens.m_l1GtTmTech = iCC.esConsumes<iTrans>();
    }
    if (vetoMasks) {
      tokens.m_l1GtTmVetoAlgo = iCC.esConsumes<iTrans>();
      tokens.m_l1GtTmVetoTech = iCC.esConsumes<iTrans>();
    }
    return tokens;
  }
}  // namespace
// constructor(s)
L1GtPrescaleFactorsAndMasksTester::L1GtPrescaleFactorsAndMasksTester(const edm::ParameterSet& parSet)
    : m_testerPrescaleFactors(parSet.getParameter<bool>("TesterPrescaleFactors")),
      m_testerTriggerMask(parSet.getParameter<bool>("TesterTriggerMask")),
      m_testerTriggerVetoMask(parSet.getParameter<bool>("TesterTriggerVetoMask")),
      m_retrieveInBeginRun(parSet.getParameter<bool>("RetrieveInBeginRun")),
      m_retrieveInBeginLuminosityBlock(parSet.getParameter<bool>("RetrieveInBeginLuminosityBlock")),
      m_retrieveInAnalyze(parSet.getParameter<bool>("RetrieveInAnalyze")),
      m_printInBeginRun(parSet.getParameter<bool>("PrintInBeginRun")),
      m_printInBeginLuminosityBlock(parSet.getParameter<bool>("PrintInBeginLuminosityBlock")),
      m_printInAnalyze(parSet.getParameter<bool>("PrintInAnalyze")),
      m_printOutput(parSet.getUntrackedParameter<int>("PrintOutput", 3)),
      m_run(tokens<edm::Transition::BeginRun>(
          consumesCollector(), m_testerPrescaleFactors, m_testerTriggerMask, m_testerTriggerVetoMask)),
      m_lumi(tokens<edm::Transition::BeginLuminosityBlock>(
          consumesCollector(), m_testerPrescaleFactors, m_testerTriggerMask, m_testerTriggerVetoMask)),
      m_event(tokens<edm::Transition::Event>(
          consumesCollector(), m_testerPrescaleFactors, m_testerTriggerMask, m_testerTriggerVetoMask)) {
  // empty
}

// begin run
void L1GtPrescaleFactorsAndMasksTester::beginRun(const edm::Run& iRun, const edm::EventSetup& evSetup) {
  if (m_retrieveInBeginRun) {
    retrieveL1EventSetup(evSetup, m_run);
  }

  if (m_printInBeginRun) {
    printL1EventSetup();
  }
}

// begin luminosity block
void L1GtPrescaleFactorsAndMasksTester::beginLuminosityBlock(const edm::LuminosityBlock& iLumiBlock,
                                                             const edm::EventSetup& evSetup) {
  if (m_retrieveInBeginLuminosityBlock) {
    retrieveL1EventSetup(evSetup, m_lumi);
  }

  if (m_printInBeginLuminosityBlock) {
    printL1EventSetup();
  }
}

// loop over events
void L1GtPrescaleFactorsAndMasksTester::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  if (m_retrieveInAnalyze) {
    retrieveL1EventSetup(evSetup, m_event);
  }

  if (m_printInAnalyze) {
    printL1EventSetup();
  }
}

// end luminosity block
void L1GtPrescaleFactorsAndMasksTester::endLuminosityBlock(const edm::LuminosityBlock& iLumiBlock,
                                                           const edm::EventSetup& evSetup) {}

// end run
void L1GtPrescaleFactorsAndMasksTester::endRun(const edm::Run& iRun, const edm::EventSetup& evSetup) {}

void L1GtPrescaleFactorsAndMasksTester::retrieveL1EventSetup(const edm::EventSetup& evSetup, const Tokens& tokens) {
  if (m_testerPrescaleFactors) {
    // get / update the prescale factors from the EventSetup

    m_l1GtPfAlgo = &evSetup.getData(tokens.m_l1GtPfAlgo);
    m_l1GtPfTech = &evSetup.getData(tokens.m_l1GtPfTech);
  }

  if (m_testerTriggerMask) {
    // get / update the trigger mask from the EventSetup

    m_l1GtTmAlgo = &evSetup.getData(tokens.m_l1GtTmAlgo);
    m_l1GtTmTech = &evSetup.getData(tokens.m_l1GtTmTech);
  }

  if (m_testerTriggerVetoMask) {
    m_l1GtTmVetoAlgo = &evSetup.getData(tokens.m_l1GtTmVetoAlgo);
    m_l1GtTmVetoTech = &evSetup.getData(tokens.m_l1GtTmVetoTech);
  }
}

void L1GtPrescaleFactorsAndMasksTester::printL1EventSetup() {
  // define an output stream to print into
  // it can then be directed to whatever log level is desired
  std::ostringstream myCout;

  if (m_testerPrescaleFactors) {
    myCout << "\nL1 GT prescale factors for algorithm triggers" << std::endl;
    m_l1GtPfAlgo->print(myCout);

    myCout << "\nL1 GT prescale factors for technical triggers" << std::endl;
    m_l1GtPfTech->print(myCout);
  }

  //
  if (m_testerTriggerMask) {
    myCout << "\nL1 GT trigger masks for algorithm triggers" << std::endl;
    m_l1GtTmAlgo->print(myCout);

    myCout << "\nL1 GT trigger masks for technical triggers" << std::endl;
    m_l1GtTmTech->print(myCout);
  }

  //
  if (m_testerTriggerVetoMask) {
    myCout << "\nL1 GT trigger veto masks for algorithm triggers" << std::endl;
    m_l1GtTmVetoAlgo->print(myCout);

    myCout << "\nL1 GT trigger veto masks for technical triggers" << std::endl;
    m_l1GtTmVetoTech->print(myCout);
  }

  switch (m_printOutput) {
    case 0: {
      std::cout << myCout.str() << std::endl;

    }

    break;
    case 1: {
      LogTrace("L1GtPrescaleFactorsAndMasksTester") << myCout.str() << std::endl;

    } break;

    case 2: {
      edm::LogVerbatim("L1GtPrescaleFactorsAndMasksTester") << myCout.str() << std::endl;

    }

    break;
    case 3: {
      edm::LogInfo("L1GtPrescaleFactorsAndMasksTester") << myCout.str();

    }

    break;
    default: {
      myCout << "\n\n  L1GtPrescaleFactorsAndMasksTester: Error - no print output = " << m_printOutput
             << " defined! \n  Check available values in the cfi file."
             << "\n"
             << std::endl;

    } break;
  }
}
