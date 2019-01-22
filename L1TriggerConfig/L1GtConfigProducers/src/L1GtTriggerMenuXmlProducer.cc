/**
 * \class L1GtTriggerMenuXmlProducer
 *
 *
 * Description: ESProducer for the L1 Trigger Menu from an XML file .
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuXmlProducer.h"

// system include files
#include <memory>


// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuXmlParser.h"



// forward declarations

// constructor(s)
L1GtTriggerMenuXmlProducer::L1GtTriggerMenuXmlProducer(
    const edm::ParameterSet& parSet)
{
    // tell the framework what data is being produced
    setWhatProduced(this, &L1GtTriggerMenuXmlProducer::produceGtTriggerMenu);


    // now do what ever other initialization is needed


    // directory in /data/Luminosity for the trigger menu
    std::string menuDir = parSet.getParameter<std::string>("TriggerMenuLuminosity");


    // def.xml file
    std::string defXmlFileName = parSet.getParameter<std::string>("DefXmlFile");

    edm::FileInPath f1("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/" +
                       menuDir + "/" + defXmlFileName);

    m_defXmlFile = f1.fullPath();


    // vme.xml file
    std::string vmeXmlFileName = parSet.getParameter<std::string>("VmeXmlFile");

    if (!vmeXmlFileName.empty()) {
        edm::FileInPath f2("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/" +
                           menuDir + "/" + vmeXmlFileName);

        m_vmeXmlFile = f2.fullPath();

    }

    edm::LogInfo("L1GtConfigProducers")
    << "\n\nL1 Trigger Menu: "
    << "\n\n  def.xml file: \n    " << m_defXmlFile
    << "\n\n  vme.xml file: \n    " << m_vmeXmlFile
    << "\n\n"
    << std::endl;

}

// destructor
L1GtTriggerMenuXmlProducer::~L1GtTriggerMenuXmlProducer()
{

    // empty

}


// member functions

// method called to produce the data
std::unique_ptr<L1GtTriggerMenu> L1GtTriggerMenuXmlProducer::produceGtTriggerMenu(
    const L1GtTriggerMenuRcd& l1MenuRecord)
{

    // get the parameters needed from other records
    const L1GtStableParametersRcd& stableParametersRcd =
        l1MenuRecord.getRecord<L1GtStableParametersRcd>();

    edm::ESHandle<L1GtStableParameters> stableParameters;
    stableParametersRcd.get(stableParameters);

    unsigned int numberConditionChips = stableParameters->gtNumberConditionChips();
    unsigned int pinsOnConditionChip = stableParameters->gtPinsOnConditionChip();
    std::vector<int> orderConditionChip = stableParameters->gtOrderConditionChip();
    unsigned int numberPhysTriggers = stableParameters->gtNumberPhysTriggers();
    unsigned int numberTechTriggers = stableParameters->gtNumberTechnicalTriggers();
    unsigned int numberL1JetCounts = stableParameters->gtNumberL1JetCounts();

    L1GtTriggerMenuXmlParser gtXmlParser = L1GtTriggerMenuXmlParser();

    gtXmlParser.setGtNumberConditionChips(numberConditionChips);
    gtXmlParser.setGtPinsOnConditionChip(pinsOnConditionChip);
    gtXmlParser.setGtOrderConditionChip(orderConditionChip);
    gtXmlParser.setGtNumberPhysTriggers(numberPhysTriggers);
    gtXmlParser.setGtNumberTechTriggers(numberTechTriggers);
    gtXmlParser.setGtNumberL1JetCounts(numberL1JetCounts);

    gtXmlParser.parseXmlFile(m_defXmlFile, m_vmeXmlFile);

    // transfer the condition map and algorithm map from parser to L1GtTriggerMenu

    auto pL1GtTriggerMenu = std::make_unique<L1GtTriggerMenu>(
                        gtXmlParser.gtTriggerMenuName(), numberConditionChips,
                        gtXmlParser.vecMuonTemplate(),
                        gtXmlParser.vecCaloTemplate(),
                        gtXmlParser.vecEnergySumTemplate(),
                        gtXmlParser.vecJetCountsTemplate(),
                        gtXmlParser.vecCastorTemplate(),
                        gtXmlParser.vecHfBitCountsTemplate(),
                        gtXmlParser.vecHfRingEtSumsTemplate(),
                        gtXmlParser.vecBptxTemplate(),
                        gtXmlParser.vecExternalTemplate(),
                        gtXmlParser.vecCorrelationTemplate(),
                        gtXmlParser.corMuonTemplate(),
                        gtXmlParser.corCaloTemplate(),
                        gtXmlParser.corEnergySumTemplate());


    pL1GtTriggerMenu->setGtTriggerMenuInterface(gtXmlParser.gtTriggerMenuInterface());
    pL1GtTriggerMenu->setGtTriggerMenuImplementation(gtXmlParser.gtTriggerMenuImplementation());
    pL1GtTriggerMenu->setGtScaleDbKey(gtXmlParser.gtScaleDbKey());

    pL1GtTriggerMenu->setGtAlgorithmMap(gtXmlParser.gtAlgorithmMap());
    pL1GtTriggerMenu->setGtAlgorithmAliasMap(gtXmlParser.gtAlgorithmAliasMap());
    pL1GtTriggerMenu->setGtTechnicalTriggerMap(gtXmlParser.gtTechnicalTriggerMap());

    //LogDebug("L1GtConfigProducers")
    //<< "\n\nReturning L1 Trigger Menu!"
    //<< "\n\n"
    //<< std::endl;

    return pL1GtTriggerMenu ;
}

