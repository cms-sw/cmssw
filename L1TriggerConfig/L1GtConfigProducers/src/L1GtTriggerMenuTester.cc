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
#include <boost/algorithm/string/erase.hpp>

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

#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

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
    //int physicsDaqPartition = 0;

    // use another map <int, L1GtAlgorithm> to get the menu sorted after bit number
    // both algorithm and bit numbers are unique
    typedef std::map<int, const L1GtAlgorithm*>::const_iterator CItBit;

    //    algorithm triggers
    const AlgorithmMap& algorithmMap = l1GtMenu->gtAlgorithmMap();


    std::map<int, const L1GtAlgorithm*> algoBitToAlgo;


    std::map<std::string, const L1GtAlgorithm*> jetAlgoTrig;
    std::map<std::string, const L1GtAlgorithm*> egammaAlgoTrig;
    std::map<std::string, const L1GtAlgorithm*> esumAlgoTrig;
    std::map<std::string, const L1GtAlgorithm*> muonAlgoTrig;
    std::map<std::string, const L1GtAlgorithm*> crossAlgoTrig;
    std::map<std::string, const L1GtAlgorithm*> bkgdAlgoTrig;

    int algoTrigNumberTotal = 128; // FIXME take it from stable parameters

    int algoTrigNumber = 0;
    int freeAlgoTrigNumber = 0;

    int jetAlgoTrigNumber = 0;
    int egammaAlgoTrigNumber = 0;
    int esumAlgoTrigNumber = 0;
    int muonAlgoTrigNumber = 0;
    int crossAlgoTrigNumber = 0;
    int bkgdAlgoTrigNumber = 0;

    for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {

        const int bitNumber = (itAlgo->second).algoBitNumber();
        const std::string& algName = (itAlgo->second).algoName();

        algoBitToAlgo[bitNumber] = &(itAlgo->second);

        algoTrigNumber++;

        // per category

        const ConditionMap& conditionMap =
                (l1GtMenu->gtConditionMap()).at((itAlgo->second).algoChipNumber());

        const std::vector<L1GtLogicParser::TokenRPN>& rpnVector = (itAlgo->second).algoRpnVector();
        const L1GtLogicParser::OperationType condOperand = L1GtLogicParser::OP_OPERAND;

        std::list<L1GtObject> listObjects;

        for (size_t i = 0; i < rpnVector.size(); ++i) {

            if ((rpnVector[i]).operation == condOperand) {

                const std::string& cndName = (rpnVector[i]).operand;

                // search the condition in the condition list

                bool foundCond = false;

                CItCond itCond = conditionMap.find(cndName);
                if (itCond != conditionMap.end()) {
                    foundCond = true;

                    // loop through object types and add them to the list

                    const std::vector<L1GtObject>& objType =
                            (itCond->second)->objectType();

                    for (std::vector<L1GtObject>::const_iterator itObject =
                            objType.begin(); itObject != objType.end(); itObject++) {
                        listObjects.push_back(*itObject);

                        std::cout << (*itObject) << std::endl;
                    }

                    // FIXME
                    if ((itCond->second)->condCategory() == CondExternal) {
                        listObjects.push_back(GtExternal);
                    }

                }

                if (!foundCond) {

                    // it should never be happen, all conditions are in the maps
                    throw cms::Exception("FailModule") << "\nCondition "
                            << cndName << " not found in the condition map"
                            << " for chip number "
                            << ((itAlgo->second).algoChipNumber()) << std::endl;
                }

            }

        }

        // eliminate duplicates
        listObjects.sort();
        listObjects.unique();

        // add the algorithm to the corresponding group

        bool jetGroup = false;
        bool egammaGroup = false;
        bool esumGroup = false;
        bool muonGroup = false;
        bool crossGroup = false;
        bool bkgdGroup = false;

        for (std::list<L1GtObject>::const_iterator itObj = listObjects.begin(); itObj != listObjects.end(); ++itObj) {


            switch (*itObj) {
                case Mu: {
                    muonGroup = true;
                }

                break;
                case NoIsoEG: {
                    egammaGroup = true;
                }

                break;
                case IsoEG: {
                    egammaGroup = true;
                }

                break;
                case CenJet: {
                    jetGroup = true;
                }

                break;
                case ForJet: {
                    jetGroup = true;
                }

                break;
                case TauJet: {
                    jetGroup = true;
                }

                break;
                case ETM: {
                    esumGroup = true;

                }

                break;
                case ETT: {
                    esumGroup = true;

                }

                break;
                case HTT: {
                    esumGroup = true;

                }

                break;
                case HTM: {
                    esumGroup = true;

                }

                break;
                case JetCounts: {
                    // do nothing - not available
                }

                break;
                case HfBitCounts: {
                    bkgdGroup = true;
                }

                break;
                case HfRingEtSums: {
                    bkgdGroup = true;
                }

                break;
                case GtExternal: {
                    bkgdGroup = true;
                }

                break;
                case TechTrig:
                case Castor:
                case BPTX:
                default: {
                    // should not arrive here

                    std::cout << "\n      Unknown object of type " << *itObj
                    << std::endl;
                }
                break;
            }

        }


        int sumGroup = jetGroup + egammaGroup + esumGroup + muonGroup
                + crossGroup + bkgdGroup;

        if (sumGroup > 1) {
            crossAlgoTrig[algName] = &(itAlgo->second);
        } else {

            if (jetGroup) {
                jetAlgoTrig[algName] = &(itAlgo->second);

            } else if (egammaGroup) {
                egammaAlgoTrig[algName] = &(itAlgo->second);

            } else if (esumGroup && (listObjects.size() > 1)) {
                crossAlgoTrig[algName] = &(itAlgo->second);

            } else if (esumGroup) {
                esumAlgoTrig[algName] = &(itAlgo->second);

            } else if (muonGroup) {
                muonAlgoTrig[algName] = &(itAlgo->second);

            } else if (bkgdGroup) {
                bkgdAlgoTrig[algName] = &(itAlgo->second);

            } else {
                // do nothing
            }

        }

        std::cout << algName << " sum: " << sumGroup << " size: " << listObjects.size() << std::endl;

    }

    freeAlgoTrigNumber = algoTrigNumberTotal - algoTrigNumber;

    jetAlgoTrigNumber = jetAlgoTrig.size();
    egammaAlgoTrigNumber = egammaAlgoTrig.size();
    esumAlgoTrigNumber = esumAlgoTrig.size();
    muonAlgoTrigNumber = muonAlgoTrig.size();
    crossAlgoTrigNumber = crossAlgoTrig.size();
    bkgdAlgoTrigNumber = bkgdAlgoTrig.size();



    //    technical triggers
    std::map<int, const L1GtAlgorithm*> techBitToAlgo;
    const AlgorithmMap& technicalTriggerMap = l1GtMenu->gtTechnicalTriggerMap();

    int techTrigNumberTotal = 64; // FIXME take it from stable parameters

    int techTrigNumber = 0;
    int freeTechTrigNumber = 0;

    for (CItAlgo itAlgo = technicalTriggerMap.begin(); itAlgo != technicalTriggerMap.end(); itAlgo++) {

        int bitNumber = (itAlgo->second).algoBitNumber();
        techBitToAlgo[bitNumber] = &(itAlgo->second);

        techTrigNumber++;
    }

    freeTechTrigNumber = techTrigNumberTotal - techTrigNumber;

    // name of the attached HTML file FIXME get it automatically from L1 Trigger Menu Implementation
    //m_htmlFile =  "%ATTACHURL%/L1Menu_Collisions2011_v6_L1T_Scales_20101224_Imp0_0x1024.html";
    m_htmlFile =  "%ATTACHURL%/L1Menu_Collisions2012_v0_L1T_Scales_20101224_Imp0_0x1027.html";


    // header for printing algorithms

    std::cout << "\n   ********** L1 Trigger Menu - printing   ********** \n\n"
            << "\n---+++ L1 menu identification\n"
            << "\n|L1 Trigger Menu Interface: |!" << l1GtMenu->gtTriggerMenuInterface() << " |"
            << "\n|L1 Trigger Menu Name: |!" << l1GtMenu->gtTriggerMenuName() << " |"
            << "\n|L1 Trigger Menu Implementation: |!" << l1GtMenu->gtTriggerMenuImplementation() << " |"
            << "\n|Associated L1 scale DB key: |!" << l1GtMenu->gtScaleDbKey() << " |" << "\n\n"
            << std::flush << std::endl;

    // Overview page
    std::cout << "\n---+++ Summary\n" << std::endl;
    std::cout << "   * Number of algorithm triggers: " << algoTrigNumber << " defined, 128 possible." << std::endl;
    std::cout << "   * Number of technical triggers: " << techTrigNumber << " defined,  64 possible.<BR><BR>" << std::endl;

    std::cout << "   * Number of free bits for algorithm triggers: " << freeAlgoTrigNumber << std::endl;
    std::cout << "   * Number of free bits for technical triggers: " << freeTechTrigNumber << "<BR>" << std::endl;

    std::cout << "\nNumber of algorithm triggers per trigger group\n" << std::endl;
    std::cout << "    | *Trigger group* | *Number of bits used*|"<< std::endl;
    std::cout << "    | Jet algorithm triggers: |  " << jetAlgoTrigNumber << "|"<< std::endl;
    std::cout << "    | EGamma algorithm triggers: |  " << egammaAlgoTrigNumber << "|"<< std::endl;
    std::cout << "    | Energy sum algorithm triggers: |  " << esumAlgoTrigNumber << "|"<< std::endl;
    std::cout << "    | Muon algorithm triggers: |  " << muonAlgoTrigNumber << "|"<< std::endl;
    std::cout << "    | Cross algorithm triggers: |  " << crossAlgoTrigNumber << "|"<< std::endl;
    std::cout << "    | Background algorithm triggers: |  " << bkgdAlgoTrigNumber << "|"<< std::endl;

    // force a page break
    std::cout << "<p style=\"page-break-before: always\">&nbsp;</p>";

    std::cout << "\n---+++ List of algorithm triggers sorted by trigger groups\n" << std::endl;

    // Jet algorithm triggers
    printTriggerGroup("Jet algorithm triggers", jetAlgoTrig);

    // EGamma algorithm triggers
    printTriggerGroup("EGamma algorithm triggers", egammaAlgoTrig);

    // Energy sum algorithm triggers
    printTriggerGroup("Energy sum algorithm triggers", esumAlgoTrig);

    // Muon algorithm triggers
    printTriggerGroup("Muon algorithm triggers", muonAlgoTrig);

    // Cross algorithm triggers
    printTriggerGroup("Cross algorithm triggers", crossAlgoTrig);

    // Background algorithm triggers
    printTriggerGroup("Background algorithm triggers", bkgdAlgoTrig);


    // force a page break
    std::cout << "<p style=\"page-break-before: always\">&nbsp;</p>";

    std::cout << "\n---+++ List of algorithm triggers sorted by bits\n" << std::endl;

    std::cout
    << "| *Algorithm* | *Alias* | *Bit number* |"
    << std::endl;

    for (CItBit itBit = algoBitToAlgo.begin(); itBit != algoBitToAlgo.end(); itBit++) {

        int bitNumber = itBit->first;
        std::string aName = (itBit->second)->algoName();
        std::string aAlias = (itBit->second)->algoAlias();

        std::cout
        << "|" << std::left << aName << "  |" << aAlias << "  |  " << std::right << bitNumber
        << " |"
        << std::endl;
    }


    // force a page break
    std::cout << "<p style=\"page-break-before: always\">&nbsp;</p>";
    std::cout << "\n---+++ List of technical triggers\n" << std::endl;

    std::cout << "| *Technical trigger* | *Bit number* |" << std::endl;

    for (CItBit itBit = techBitToAlgo.begin(); itBit != techBitToAlgo.end(); itBit++) {

        int bitNumber = itBit->first;
        std::string aName = (itBit->second)->algoName();
        std::string aAlias = (itBit->second)->algoAlias();

        std::cout << "|!" << std::left << aName << "  |  " << std::right
                << bitNumber << " |" << std::endl;
    }

}


// printing template for a trigger group
void L1GtTriggerMenuTester::printTriggerGroup(const std::string& trigGroupName,
        const std::map<std::string, const L1GtAlgorithm*>& trigGroup) {


    // force a page break before each group
    std::cout << "<p style=\"page-break-before: always\">&nbsp;</p>";

    // FIXME get values - either read from a specific L1 menu file, or from
    std::string lumiVal1 = "5.0E33";
    std::string lumiVal2 = "7.0E33";
    std::string seedsHlt;
    std::string trigComment;

    int trigPfVal1 = 0;
    int trigPfVal2 = 0;

    int trigRateVal1 = 0;
    int trigRateVal2 = 0;

    std::cout << "\n---++++ " << trigGroupName << "\n" << std::endl;

    std::cout
            << "|  *Trigger Name*  |  *Trigger Alias*  |  *Bit*  |  *Luminosity*  ||||  *Seed for !HLT path(s)*  |  *Comments*  |"
            << std::endl;

    std::cout << "|^|^|^|  *" << lumiVal1 << "*  ||  *" << lumiVal2
            << "*  ||  **  |  **  |" << std::endl;

    std::cout << "|^|^|^|  *PF*  |  *Rate*  |  *PF*  |  *Rate*  |  ** |  **  |"
            << std::endl;

    for (CItAlgoP itAlgo = trigGroup.begin(); itAlgo != trigGroup.end(); itAlgo++) {

        const std::string& aName = (itAlgo->second)->algoName();
        const std::string& aAlias = (itAlgo->second)->algoAlias();
        const int& bitNumber = (itAlgo->second)->algoBitNumber();

        std::cout
                << "|" << std::left << "[[" << (m_htmlFile + "#" + aName) << "][ " << aName << "]] " << "  |"
                << aAlias << "  |  "
                << bitNumber << "|  "
                << ( (trigPfVal1 != 0) ? trigPfVal1 : 0) << "  |  "
                << ( (trigRateVal1 != 0) ? trigRateVal1 : 0) << "  |  "
                << ( (trigPfVal2 != 0) ? trigPfVal2 : 0) << "  |  "
                << ( (trigRateVal2 != 0) ? trigRateVal2 : 0) << "  |  "
                << seedsHlt << "  |  " << trigComment << "  |"<< std::endl;
    }

    std::cout << "\n" << trigGroupName << ": " << (trigGroup.size())
            << " bits defined." << std::endl;

}


