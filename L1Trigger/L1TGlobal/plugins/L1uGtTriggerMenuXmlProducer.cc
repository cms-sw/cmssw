/**
 * \class L1uGtTriggerMenuXmlProducer
 *
 *
 * Description: ESProducer for the L1 Trigger Menu from an XML file .
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
#include "L1Trigger/L1TGlobal/interface/L1uGtTriggerMenuXmlProducer.h"

// system include files
#include <memory>

#include "boost/shared_ptr.hpp"


// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/L1TObjects/interface/L1uGtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

#include "L1Trigger/L1TGlobal/interface/L1uGtTriggerMenuXmlParser.h"



// forward declarations

// constructor(s)
l1t::L1uGtTriggerMenuXmlProducer::L1uGtTriggerMenuXmlProducer(
    const edm::ParameterSet& parSet)
{
    // tell the framework what data is being produced
    setWhatProduced(this, &l1t::L1uGtTriggerMenuXmlProducer::produceGtTriggerMenu);


    // now do what ever other initialization is needed


    // directory in /data/Luminosity for the trigger menu
    std::string menuDir = parSet.getParameter<std::string>("TriggerMenuLuminosity");


    // def.xml file
    std::string defXmlFileName = parSet.getParameter<std::string>("DefXmlFile");

    edm::FileInPath f1("L1Trigger/L1TGlobal/data/Luminosity/" +
                       menuDir + "/" + defXmlFileName);

    m_defXmlFile = f1.fullPath();


    // vme.xml file
    std::string vmeXmlFileName = parSet.getParameter<std::string>("VmeXmlFile");

    if (vmeXmlFileName != "") {
        edm::FileInPath f2("L1Trigger/L1TGlobal/data/Luminosity/" +
                           menuDir + "/" + vmeXmlFileName);

        m_vmeXmlFile = f2.fullPath();

    }

    edm::LogInfo("L1TGlobal")
    << "\n\nL1 Trigger Menu: "
    << "\n\n  def.xml file: \n    " << m_defXmlFile
    << "\n\n  vme.xml file: \n    " << m_vmeXmlFile
    << "\n\n"
    << std::endl;

}

// destructor
l1t::L1uGtTriggerMenuXmlProducer::~L1uGtTriggerMenuXmlProducer()
{

    // empty

}


// member functions

// method called to produce the data
boost::shared_ptr<L1uGtTriggerMenu> l1t::L1uGtTriggerMenuXmlProducer::produceGtTriggerMenu(
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

    LogDebug("l1t|Global")
      << "\n\t numberConditionChips = " << numberConditionChips 
      << "\n\t pinsOnConditionChip = " << pinsOnConditionChip 
      << "\n\t orderConditionChip.size() = " << orderConditionChip.size()
      << std::endl;
    for( int i=0; i<int(orderConditionChip.size()); i++ ){
      LogDebug("l1t|Global") << "\t\t " << i << "\t" << orderConditionChip[i] << std::endl;
    }
    LogDebug("l1t|Global")
      << "\n\t numberPhysTriggers = " << numberPhysTriggers 
      << "\n\t numberTechTriggers = " << numberTechTriggers 
      << "\n\t numberL1JetCounts = " << numberL1JetCounts 
      << std::endl;

    l1t::L1uGtTriggerMenuXmlParser gtXmlParser = l1t::L1uGtTriggerMenuXmlParser();

    gtXmlParser.setGtNumberConditionChips(numberConditionChips);
    gtXmlParser.setGtPinsOnConditionChip(pinsOnConditionChip);
    gtXmlParser.setGtOrderConditionChip(orderConditionChip);
    gtXmlParser.setGtNumberPhysTriggers(numberPhysTriggers);
    gtXmlParser.setGtNumberTechTriggers(numberTechTriggers);
    gtXmlParser.setGtNumberL1JetCounts(numberL1JetCounts);

    gtXmlParser.parseXmlFile(m_defXmlFile, m_vmeXmlFile);

    // transfer the condition map and algorithm map from parser to L1uGtTriggerMenu

    boost::shared_ptr<L1uGtTriggerMenu> pL1uGtTriggerMenu = boost::shared_ptr<L1uGtTriggerMenu>(
                new L1uGtTriggerMenu(gtXmlParser.gtTriggerMenuName(), numberConditionChips,
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
                        gtXmlParser.corEnergySumTemplate()) );


    pL1uGtTriggerMenu->setGtTriggerMenuInterface(gtXmlParser.gtTriggerMenuInterface());
    pL1uGtTriggerMenu->setGtTriggerMenuImplementation(gtXmlParser.gtTriggerMenuImplementation());
    pL1uGtTriggerMenu->setGtScaleDbKey(gtXmlParser.gtScaleDbKey());

    pL1uGtTriggerMenu->setGtAlgorithmMap(gtXmlParser.gtAlgorithmMap());
    pL1uGtTriggerMenu->setGtAlgorithmAliasMap(gtXmlParser.gtAlgorithmAliasMap());
    pL1uGtTriggerMenu->setGtTechnicalTriggerMap(gtXmlParser.gtTechnicalTriggerMap());

    //LogDebug("L1TGlobalConfig")
    //<< "\n\nReturning L1 Trigger Menu!"
    //<< "\n\n"
    //<< std::endl;

    return pL1uGtTriggerMenu ;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(l1t::L1uGtTriggerMenuXmlProducer);
