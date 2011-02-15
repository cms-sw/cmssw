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
    int physicsDaqPartition = 0;

    // use another map <int, L1GtAlgorithm> to get the menu sorted after bit number
    // both algorithm and bit numbers are unique
    typedef std::map<int, const L1GtAlgorithm*>::const_iterator CItBit;

    //    algorithm triggers
    const AlgorithmMap& algorithmMap = l1GtMenu->gtAlgorithmMap();


    std::map<int, const L1GtAlgorithm*> algoBitToAlgo;

    typedef std::map<std::string, const L1GtAlgorithm*>::const_iterator CItAlgoP;

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


    // header for printing algorithms

    std::cout << "\n   ********** L1 Trigger Menu - printing   ********** \n\n"
            << "\n---++ L1 menu identification\n"
            << "\n|L1 Trigger Menu Interface: |!" << l1GtMenu->gtTriggerMenuInterface() << " |"
            << "\n|L1 Trigger Menu Name: |!" << l1GtMenu->gtTriggerMenuName() << " |"
            << "\n|L1 Trigger Menu Implementation: |!" << l1GtMenu->gtTriggerMenuImplementation() << " |"
            << "\n|Associated L1 scale DB key: |!" << l1GtMenu->gtScaleDbKey() << " |" << "\n\n"
            << std::flush << std::endl;

    // Overview page
    std::cout << "\n---++ Overview\n" << std::endl;
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

    std::cout << "\n---++ List of algorithm triggers sorted by trigger groups\n" << std::endl;

    // Jet algorithm triggers
    std::cout << "\n---+++ Jet algorithm triggers\n" << std::endl;
    std::cout << "| *AlgoTrig Name* | *AlgoTrig Alias* | *Prescale status* | *Seed for HLT path(s)* | *Expected rate* | *Comments* | " << std::endl;

    for (CItAlgoP itAlgo = jetAlgoTrig.begin(); itAlgo != jetAlgoTrig.end(); itAlgo++) {

        const std::string& aName = (itAlgo->second)->algoName();
        const std::string& aAlias = (itAlgo->second)->algoAlias();

        std::string aNameNoU = aName; // ...ugly
        boost::erase_all(aNameNoU, "_");

        std::cout
        << "|" << std::left << aName << "  |" << aAlias << "  |  " << "  |  " << "  |  " << ( "  | [[#"+aNameNoU ) << "][link]] " << "  |  "
        << std::endl;
    }

    std::cout << "\nJet algorithm triggers: " << jetAlgoTrigNumber << " bits defined."<< std::endl;


    // EGamma algorithm triggers
    std::cout << "\n---+++ EGamma algorithm triggers\n" << std::endl;
    std::cout << "| *AlgoTrig Name* | *AlgoTrig Alias* | *Prescale status* | *Seed for HLT path(s)* | *Expected rate* | *Comments* | " << std::endl;

    for (CItAlgoP itAlgo = egammaAlgoTrig.begin(); itAlgo != egammaAlgoTrig.end(); itAlgo++) {

        const std::string& aName = (itAlgo->second)->algoName();
        const std::string& aAlias = (itAlgo->second)->algoAlias();

        std::string aNameNoU = aName; // ...ugly
        boost::erase_all(aNameNoU, "_");

        std::cout
        << "|" << std::left << aName << "  |" << aAlias << "  |  " << "  |  " << "  |  " << ( "  | [[#"+aNameNoU ) << "][link]] " << "  |  "
        << std::endl;
    }

    std::cout << "\nEGamma algorithm triggers: " << egammaAlgoTrigNumber << " bits defined."<< std::endl;



    // Energy sum algorithm triggers
    std::cout << "\n---+++ Energy sum algorithm triggers\n" << std::endl;
    std::cout << "| *AlgoTrig Name* | *AlgoTrig Alias* | *Prescale status* | *Seed for HLT path(s)* | *Expected rate* | *Comments* | " << std::endl;

    for (CItAlgoP itAlgo = esumAlgoTrig.begin(); itAlgo != esumAlgoTrig.end(); itAlgo++) {

        const std::string& aName = (itAlgo->second)->algoName();
        const std::string& aAlias = (itAlgo->second)->algoAlias();

        std::string aNameNoU = aName; // ...ugly
        boost::erase_all(aNameNoU, "_");

        std::cout
        << "|" << std::left << aName << "  |" << aAlias << "  |  " << "  |  " << "  |  " << ( "  | [[#"+aNameNoU ) << "][link]] " << "  |  "
        << std::endl;
    }

    std::cout << "\nEnergy sum algorithm triggers: " << esumAlgoTrigNumber << " bits defined."<< std::endl;


    // Muon algorithm triggers
    std::cout << "\n---+++ Muon algorithm triggers\n" << std::endl;
    std::cout << "| *AlgoTrig Name* | *AlgoTrig Alias* | *Prescale status* | *Seed for HLT path(s)* | *Expected rate* | *Comments* | " << std::endl;

    for (CItAlgoP itAlgo = muonAlgoTrig.begin(); itAlgo != muonAlgoTrig.end(); itAlgo++) {

        const std::string& aName = (itAlgo->second)->algoName();
        const std::string& aAlias = (itAlgo->second)->algoAlias();

        std::string aNameNoU = aName; // ...ugly
        boost::erase_all(aNameNoU, "_");

        std::cout
        << "|" << std::left << aName << "  |" << aAlias << "  |  " << "  |  " << "  |  " << ( "  | [[#"+aNameNoU ) << "][link]] " << "  |  "
        << std::endl;
    }

    std::cout << "\nMuon algorithm triggers: " << muonAlgoTrigNumber << " bits defined."<< std::endl;


    // Cross algorithm triggers
    std::cout << "\n---+++ Cross algorithm triggers\n" << std::endl;
    std::cout << "| *AlgoTrig Name* | *AlgoTrig Alias* | *Prescale status* | *Seed for HLT path(s)* | *Expected rate* | *Comments* | " << std::endl;

    for (CItAlgoP itAlgo = crossAlgoTrig.begin(); itAlgo != crossAlgoTrig.end(); itAlgo++) {

        const std::string& aName = (itAlgo->second)->algoName();
        const std::string& aAlias = (itAlgo->second)->algoAlias();

        std::string aNameNoU = aName; // ...ugly
        boost::erase_all(aNameNoU, "_");

        std::cout
        << "|" << std::left << aName << "  |" << aAlias << "  |  " << "  |  " << "  |  " << ( "  | [[#"+aNameNoU ) << "][link]] " << "  |  "
        << std::endl;
    }

    std::cout << "\nCross algorithm triggers: " << crossAlgoTrigNumber << " bits defined."<< std::endl;


    // Background algorithm triggers
    std::cout << "\n---+++ Background algorithm triggers\n" << std::endl;
    std::cout << "| *AlgoTrig Name* | *AlgoTrig Alias* | *Prescale status* | *Seed for HLT path(s)* | *Expected rate* | *Comments* | " << std::endl;

    for (CItAlgoP itAlgo = bkgdAlgoTrig.begin(); itAlgo != bkgdAlgoTrig.end(); itAlgo++) {

        const std::string& aName = (itAlgo->second)->algoName();
        const std::string& aAlias = (itAlgo->second)->algoAlias();

        std::string aNameNoU = aName; // ...ugly
        boost::erase_all(aNameNoU, "_");

        std::cout
        << "|" << std::left << aName << "  |" << aAlias << "  |  " << "  |  " << "  |  " << ( "  | [[#"+aNameNoU ) << "][link]] " << "  |  "
        << std::endl;
    }

    std::cout << "\nBackground algorithm triggers: " << bkgdAlgoTrigNumber << " bits defined."<< std::endl;



    std::cout << "\n---+++ Comments\n" << std::endl;


    // Jet algorithm triggers - comments
    std::cout << "\n---++++ Jet algorithm triggers - comments\n" << std::endl;
    std::cout
            << "| *Algorithm trigger name*  | *Uncorrected energy* | *Typical prescale factor* | *HLT path* | *Requested by* | *POG/PAG group* | *Comments* | "
            << std::endl;

    for (CItAlgoP itAlgo = jetAlgoTrig.begin(); itAlgo != jetAlgoTrig.end(); itAlgo++) {

        const std::string& aName = (itAlgo->second)->algoName();

        std::string aNameNoU = aName; // ...ugly
        boost::erase_all(aNameNoU, "_");

        std::cout << ("#" + aNameNoU) << std::endl;
        std::cout << "| " << std::left << aName << " |  |  |  |  |  |  | "
                << std::endl;

    }


    // EGamma algorithm triggers - comments
    std::cout << "\n---++++ EGamma algorithm triggers - comments\n" << std::endl;
    std::cout
            << "| *Algorithm trigger name*  | *Typical prescale factor* | *HLT path* | *Requested by* | *POG/PAG group* | *Comments* | "
            << std::endl;

    for (CItAlgoP itAlgo = egammaAlgoTrig.begin(); itAlgo != egammaAlgoTrig.end(); itAlgo++) {

        const std::string& aName = (itAlgo->second)->algoName();

        std::string aNameNoU = aName; // ...ugly
        boost::erase_all(aNameNoU, "_");

        std::cout << ("#" + aNameNoU) << std::endl;
        std::cout << "| " << std::left << aName << " |  |  |  |  |  | "
                << std::endl;

    }


    // Energy sum algorithm triggers - comments
    std::cout << "\n---++++ Energy sum algorithm triggers - comments\n" << std::endl;
    std::cout
            << "| *Algorithm trigger name*  | *Typical prescale factor* | *HLT path* | *Requested by* | *POG/PAG group* | *Comments* | "
            << std::endl;

    for (CItAlgoP itAlgo = esumAlgoTrig.begin(); itAlgo != esumAlgoTrig.end(); itAlgo++) {

        const std::string& aName = (itAlgo->second)->algoName();

        std::string aNameNoU = aName; // ...ugly
        boost::erase_all(aNameNoU, "_");

        std::cout << ("#" + aNameNoU) << std::endl;
        std::cout << "| " << std::left << aName << " |  |  |  |  |  | "
                << std::endl;

    }


    // Muon algorithm triggers - comments
    std::cout << "\n---++++ Muon algorithm triggers - comments\n" << std::endl;
    std::cout
            << "| *Algorithm trigger name*  | *Typical prescale factor* | *HLT path* | *Requested by* | *POG/PAG group* | *Comments* | "
            << std::endl;

    for (CItAlgoP itAlgo = muonAlgoTrig.begin(); itAlgo != muonAlgoTrig.end(); itAlgo++) {

        const std::string& aName = (itAlgo->second)->algoName();

        std::string aNameNoU = aName; // ...ugly
        boost::erase_all(aNameNoU, "_");

        std::cout << ("#" + aNameNoU) << std::endl;
        std::cout << "| " << std::left << aName << " |  |  |  |  |  | "
                << std::endl;

    }


    // Cross algorithm triggers - comments
    std::cout << "\n---++++ Cross algorithm triggers - comments\n" << std::endl;
    std::cout
            << "| *Algorithm trigger name*  | *Typical prescale factor* | *HLT path* | *Requested by* | *POG/PAG group* | *Comments* | "
            << std::endl;

    for (CItAlgoP itAlgo = crossAlgoTrig.begin(); itAlgo != crossAlgoTrig.end(); itAlgo++) {

        const std::string& aName = (itAlgo->second)->algoName();

        std::string aNameNoU = aName; // ...ugly
        boost::erase_all(aNameNoU, "_");

        std::cout << ("#" + aNameNoU) << std::endl;
        std::cout << "| " << std::left << aName << " |  |  |  |  |  | "
                << std::endl;

    }

    // Background algorithm triggers - comments
    std::cout << "\n---++++ Background algorithm triggers - comments\n" << std::endl;
    std::cout
            << "| *Algorithm trigger name*  | *Typical prescale factor* | *HLT path* | *Requested by* | *POG/PAG group* | *Comments* | "
            << std::endl;

    for (CItAlgoP itAlgo = bkgdAlgoTrig.begin(); itAlgo != bkgdAlgoTrig.end(); itAlgo++) {

        const std::string& aName = (itAlgo->second)->algoName();

        std::string aNameNoU = aName; // ...ugly
        boost::erase_all(aNameNoU, "_");

        std::cout << ("#" + aNameNoU) << std::endl;
        std::cout << "| " << std::left << aName << " |  |  |  |  |  | "
                << std::endl;

    }





    std::cout << "\n---++ List of algorithm triggers sorted by bits\n" << std::endl;

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




    std::cout << "\n---++ List of technical triggers\n" << std::endl;

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
