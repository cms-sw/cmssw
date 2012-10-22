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
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtPrescaleFactorsAndMasksTester.h"

// system include files
#include <iomanip>

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

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

// constructor(s)
L1GtPrescaleFactorsAndMasksTester::L1GtPrescaleFactorsAndMasksTester(
        const edm::ParameterSet& parSet) :
            m_testerPrescaleFactors(
                    parSet.getParameter<bool> ("TesterPrescaleFactors")),
            m_testerTriggerMask(parSet.getParameter<bool> ("TesterTriggerMask")),
            m_testerTriggerVetoMask(
                    parSet.getParameter<bool> ("TesterTriggerVetoMask")),
            m_retrieveInBeginRun(
                    parSet.getParameter<bool> ("RetrieveInBeginRun")),
            m_retrieveInBeginLuminosityBlock(
                    parSet.getParameter<bool> ("RetrieveInBeginLuminosityBlock")),
            m_retrieveInAnalyze(parSet.getParameter<bool> ("RetrieveInAnalyze")),
            m_printInBeginRun(parSet.getParameter<bool> ("PrintInBeginRun")),
            m_printInBeginLuminosityBlock(
                    parSet.getParameter<bool> ("PrintInBeginLuminosityBlock")),
            m_printInAnalyze(parSet.getParameter<bool> ("PrintInAnalyze")),
            m_printOutput(parSet.getUntrackedParameter<int> ("PrintOutput", 3)) {
    // empty
}

// destructor
L1GtPrescaleFactorsAndMasksTester::~L1GtPrescaleFactorsAndMasksTester() {
    // empty
}

// begin job
void L1GtPrescaleFactorsAndMasksTester::beginJob() {

}

// begin run
void L1GtPrescaleFactorsAndMasksTester::beginRun(const edm::Run& iRun,
        const edm::EventSetup& evSetup) {

    if (m_retrieveInBeginRun) {
        retrieveL1EventSetup(evSetup);
    }

    if (m_printInBeginRun) {
        printL1EventSetup(evSetup);
    }

}

// begin luminosity block
void L1GtPrescaleFactorsAndMasksTester::beginLuminosityBlock(
        const edm::LuminosityBlock& iLumiBlock, const edm::EventSetup& evSetup) {

    if (m_retrieveInBeginLuminosityBlock) {
        retrieveL1EventSetup(evSetup);
    }

    if (m_printInBeginLuminosityBlock) {
        printL1EventSetup(evSetup);
    }

}

// loop over events
void L1GtPrescaleFactorsAndMasksTester::analyze(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    if (m_retrieveInAnalyze) {
        retrieveL1EventSetup(evSetup);
    }

    if (m_printInAnalyze) {
        printL1EventSetup(evSetup);
    }

}

// end luminosity block
void L1GtPrescaleFactorsAndMasksTester::endLuminosityBlock(
        const edm::LuminosityBlock& iLumiBlock, const edm::EventSetup& evSetup) {

}

// end run
void L1GtPrescaleFactorsAndMasksTester::endRun(const edm::Run& iRun,
        const edm::EventSetup& evSetup) {

}

// end job
void L1GtPrescaleFactorsAndMasksTester::endJob() {

}

void L1GtPrescaleFactorsAndMasksTester::retrieveL1EventSetup(
        const edm::EventSetup& evSetup) {

    if (m_testerPrescaleFactors) {

        // get / update the prescale factors from the EventSetup

        edm::ESHandle<L1GtPrescaleFactors> l1GtPfAlgo;
        evSetup.get<L1GtPrescaleFactorsAlgoTrigRcd> ().get(l1GtPfAlgo);
        m_l1GtPfAlgo = l1GtPfAlgo.product();

        edm::ESHandle<L1GtPrescaleFactors> l1GtPfTech;
        evSetup.get<L1GtPrescaleFactorsTechTrigRcd> ().get(l1GtPfTech);
        m_l1GtPfTech = l1GtPfTech.product();
    }

    if (m_testerTriggerMask) {
        // get / update the trigger mask from the EventSetup

        edm::ESHandle<L1GtTriggerMask> l1GtTmAlgo;
        evSetup.get<L1GtTriggerMaskAlgoTrigRcd> ().get(l1GtTmAlgo);
        m_l1GtTmAlgo = l1GtTmAlgo.product();

        edm::ESHandle<L1GtTriggerMask> l1GtTmTech;
        evSetup.get<L1GtTriggerMaskTechTrigRcd> ().get(l1GtTmTech);
        m_l1GtTmTech = l1GtTmTech.product();
    }

    if (m_testerTriggerVetoMask) {
        edm::ESHandle<L1GtTriggerMask> l1GtTmVetoAlgo;
        evSetup.get<L1GtTriggerMaskVetoAlgoTrigRcd> ().get(l1GtTmVetoAlgo);
        m_l1GtTmVetoAlgo = l1GtTmVetoAlgo.product();

        edm::ESHandle<L1GtTriggerMask> l1GtTmVetoTech;
        evSetup.get<L1GtTriggerMaskVetoTechTrigRcd> ().get(l1GtTmVetoTech);
        m_l1GtTmVetoTech = l1GtTmVetoTech.product();
    }

}

void L1GtPrescaleFactorsAndMasksTester::printL1EventSetup(
        const edm::EventSetup& evSetup) {

    // define an output stream to print into
    // it can then be directed to whatever log level is desired
    std::ostringstream myCout;

    if (m_testerPrescaleFactors) {

        myCout << "\nL1 GT prescale factors for algorithm triggers"
                << std::endl;
        m_l1GtPfAlgo->print(myCout);

        myCout << "\nL1 GT prescale factors for technical triggers"
                << std::endl;
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
        myCout << "\nL1 GT trigger veto masks for algorithm triggers"
                << std::endl;
        m_l1GtTmVetoAlgo->print(myCout);

        myCout << "\nL1 GT trigger veto masks for technical triggers"
                << std::endl;
        m_l1GtTmVetoTech->print(myCout);

    }

    switch (m_printOutput) {
        case 0: {

            std::cout << myCout.str() << std::endl;

        }

            break;
        case 1: {

            LogTrace("L1GtPrescaleFactorsAndMasksTester") << myCout.str()
                    << std::endl;

        }
            break;

        case 2: {

            edm::LogVerbatim("L1GtPrescaleFactorsAndMasksTester")
                    << myCout.str() << std::endl;

        }

            break;
        case 3: {

            edm::LogInfo("L1GtPrescaleFactorsAndMasksTester") << myCout.str();

        }

            break;
        default: {
            myCout
                    << "\n\n  L1GtPrescaleFactorsAndMasksTester: Error - no print output = "
                    << m_printOutput
                    << " defined! \n  Check available values in the cfi file."
                    << "\n" << std::endl;

        }
            break;
    }

}
