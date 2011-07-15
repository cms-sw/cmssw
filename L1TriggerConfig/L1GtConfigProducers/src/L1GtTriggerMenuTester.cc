/**
 * \class L1GtTriggerMenuTester
 *
 *
 * Description: test analyzer for L1 GT trigger menu.
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuTester.h"

// system include files
#include <iomanip>

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"

#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoTechTrigRcd.h"

// forward declarations

// constructor(s)
L1GtTriggerMenuTester::L1GtTriggerMenuTester(const edm::ParameterSet& parSet) {
    // empty
}

// destructor
L1GtTriggerMenuTester::~L1GtTriggerMenuTester() {
    // empty
}

// loop over events
void L1GtTriggerMenuTester::analyze(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    edm::ESHandle< L1GtTriggerMenu> l1GtMenu;
    evSetup.get< L1GtTriggerMenuRcd>().get(l1GtMenu);
    (const_cast<L1GtTriggerMenu*>(l1GtMenu.product()))->buildGtConditionMap();

    // print with various level of verbosities

    int printVerbosity = 0;
    l1GtMenu->print(std::cout, printVerbosity);
    std::cout << std::flush << std::endl;

    printVerbosity = 1;
    l1GtMenu->print(std::cout, printVerbosity);
    std::cout << std::flush << std::endl;

    printVerbosity = 2;
    l1GtMenu->print(std::cout, printVerbosity);
    std::cout << std::flush << std::endl;

    //
    // print menu, prescale factors and trigger mask in wiki format
    //

    // L1 GT prescale factors for algorithm triggers
    edm::ESHandle< L1GtPrescaleFactors> l1GtPfAlgo;
    evSetup.get< L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo);

    int indexPfSet = 0; // FIXME

    std::vector<int> prescaleFactorsAlgoTrig =
        (l1GtPfAlgo->gtPrescaleFactors()).at(indexPfSet);


    // L1 GT prescale factors for technical triggers
    edm::ESHandle< L1GtPrescaleFactors> l1GtPfTech;
    evSetup.get< L1GtPrescaleFactorsTechTrigRcd>().get(l1GtPfTech);

    std::vector<int> prescaleFactorsTechTrig =
        (l1GtPfTech->gtPrescaleFactors()).at(indexPfSet);


    // L1 GT trigger masks for algorithm triggers
    edm::ESHandle< L1GtTriggerMask> l1GtTmAlgo;
    evSetup.get< L1GtTriggerMaskAlgoTrigRcd>().get(l1GtTmAlgo);

    std::vector<unsigned int> triggerMaskAlgoTrig = l1GtTmAlgo->gtTriggerMask();


    // L1 GT trigger masks for technical triggers
    edm::ESHandle< L1GtTriggerMask> l1GtTmTech;
    evSetup.get< L1GtTriggerMaskTechTrigRcd>().get(l1GtTmTech);

    std::vector<unsigned int> triggerMaskTechTrig = l1GtTmTech->gtTriggerMask();


    // L1 GT trigger veto masks for algorithm triggers
    edm::ESHandle< L1GtTriggerMask> l1GtTmVetoAlgo;
    evSetup.get< L1GtTriggerMaskVetoAlgoTrigRcd>().get(l1GtTmVetoAlgo);

    std::vector<unsigned int> triggerMaskVetoAlgoTrig = l1GtTmVetoAlgo->gtTriggerMask();


    // L1 GT trigger veto masks for technical triggers
    edm::ESHandle< L1GtTriggerMask> l1GtTmVetoTech;
    evSetup.get< L1GtTriggerMaskVetoTechTrigRcd>().get(l1GtTmVetoTech);

    std::vector<unsigned int> triggerMaskVetoTechTrig = l1GtTmVetoTech->gtTriggerMask();

    // set the index of physics DAQ partition TODO EventSetup?
    int physicsDaqPartition = 0;

    // use another map <int, L1GtAlgorithm> to get the menu sorted after bit number
    // both algorithm and bit numbers are unique
    typedef std::map<int, const L1GtAlgorithm*>::const_iterator CItBit;

    //    physics algorithms
    std::map<int, const L1GtAlgorithm*> algoBitToAlgo;
    const AlgorithmMap& algorithmMap = l1GtMenu->gtAlgorithmMap();

    for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {

        int bitNumber = (itAlgo->second).algoBitNumber();
        algoBitToAlgo[bitNumber] = &(itAlgo->second);
    }

    //    technical triggers
    std::map<int, const L1GtAlgorithm*> techBitToAlgo;
    const AlgorithmMap& technicalTriggerMap = l1GtMenu->gtTechnicalTriggerMap();

    for (CItAlgo itAlgo = technicalTriggerMap.begin(); itAlgo != technicalTriggerMap.end(); itAlgo++) {

        int bitNumber = (itAlgo->second).algoBitNumber();
        techBitToAlgo[bitNumber] = &(itAlgo->second);
    }

    // header for printing algorithms

    std::cout << "\n   ********** L1 Trigger Menu - printing   ********** \n\n"
            << "\n---+++ Summary\n"
            << "\n|L1 Trigger Menu Interface: |!" << l1GtMenu->gtTriggerMenuInterface() << " |"
            << "\n|L1 Trigger Menu Name: |!" << l1GtMenu->gtTriggerMenuName() << " |"
            << "\n|L1 Trigger Menu Implementation: |!" << l1GtMenu->gtTriggerMenuImplementation() << " |"
            << "\n|Associated L1 scale DB key: |!" << l1GtMenu->gtScaleDbKey() << " |" << "\n\n"
            << std::flush << std::endl;

    std::cout << "\n---+++ List of physics algorithms\n" << std::endl;

    std::cout
    << "| *Algorithm* | *Alias* | *Bit number* | *Prescale factor* | *Mask* |"
    << std::endl;

    for (CItBit itBit = algoBitToAlgo.begin(); itBit != algoBitToAlgo.end(); itBit++) {

        int bitNumber = itBit->first;
        std::string aName = (itBit->second)->algoName();
        std::string aAlias = (itBit->second)->algoAlias();

        unsigned int triggerMaskAlgo =
            (triggerMaskAlgoTrig.at(bitNumber)) & (1 << physicsDaqPartition);

        std::cout
        << "|" << std::left << aName << "  |" << aAlias << "  |  " << std::right << bitNumber
        << "|  " << prescaleFactorsAlgoTrig.at(bitNumber)
        << "|  " << triggerMaskAlgo
        << " |"
        << std::endl;
    }

    std::cout << "\n---+++ List of technical triggers\n" << std::endl;

    std::cout
    << "| *Technical trigger* | *Bit number* | *Prescale factor* | *Mask* | *Veto mask* |"
    << std::endl;

    for (CItBit itBit = techBitToAlgo.begin(); itBit != techBitToAlgo.end(); itBit++) {

        int bitNumber = itBit->first;
        std::string aName = (itBit->second)->algoName();
        std::string aAlias = (itBit->second)->algoAlias();

        unsigned int triggerMaskTech =
            (triggerMaskTechTrig.at(bitNumber)) & (1 << physicsDaqPartition);
        unsigned int triggerMaskVetoTech =
            (triggerMaskVetoTechTrig.at(bitNumber)) & (1 << physicsDaqPartition);

        std::cout
        << "|!" << std::left << aName << "  |  " << std::right << bitNumber
        << "|  " << prescaleFactorsTechTrig.at(bitNumber)
        << "|  " << triggerMaskTech
        << "|  " << triggerMaskVetoTech
        << " |"
        << std::endl;
    }

    std::cout
    << "\nNOTE: only the prescale factors from set index zero are printed!"
    << std::endl;

}
