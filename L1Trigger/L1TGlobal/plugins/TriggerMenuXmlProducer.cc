/**
 * \class TriggerMenuXmlProducer
 *
 *
 * Description: ESProducer for the L1 Trigger Menu from an XML file .
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 */

// this class header
#include "TriggerMenuXmlProducer.h"

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

#include "L1Trigger/L1TGlobal/interface/TriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TGlobalTriggerMenuRcd.h"

#include "CondFormats/L1TObjects/interface/GlobalStableParameters.h"
#include "CondFormats/DataRecord/interface/L1TGlobalStableParametersRcd.h"

#include "TriggerMenuXmlParser.h"



// forward declarations

// constructor(s)
l1t::TriggerMenuXmlProducer::TriggerMenuXmlProducer(
    const edm::ParameterSet& parSet)
{
    // tell the framework what data is being produced
    setWhatProduced(this, &l1t::TriggerMenuXmlProducer::produceGtTriggerMenu);


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
l1t::TriggerMenuXmlProducer::~TriggerMenuXmlProducer()
{

    // empty

}


// method called to produce the data
boost::shared_ptr<TriggerMenu> l1t::TriggerMenuXmlProducer::produceGtTriggerMenu(
    const L1TGlobalTriggerMenuRcd& l1MenuRecord)
{

    // get the parameters needed from other records
    const L1TGlobalStableParametersRcd& stableParametersRcd =
    l1MenuRecord.getRecord<L1TGlobalStableParametersRcd>();

    edm::ESHandle<GlobalStableParameters> stableParameters;
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



    // Added by DP 4/25/2014

    /// get / set the number of physics trigger algorithms
    unsigned int gtNumberPhysTriggers = stableParameters->gtNumberPhysTriggers();
    /// get / set the additional number of physics trigger algorithms
    unsigned int gtNumberPhysTriggersExtended = stableParameters->gtNumberPhysTriggersExtended();
    /// get / set the number of technical triggers
    unsigned int gtNumberTechnicalTriggers = stableParameters->gtNumberTechnicalTriggers();
    ///  get / set the number of L1 muons received by GT
    unsigned int gtNumberL1Mu = stableParameters->gtNumberL1Mu();
    ///  get / set the number of L1 e/gamma objects received by GT
    unsigned int gtNumberL1NoIsoEG = stableParameters->gtNumberL1NoIsoEG();
    ///  get / set the number of L1 isolated e/gamma objects received by GT
    unsigned int gtNumberL1IsoEG = stableParameters->gtNumberL1IsoEG();
    ///  get / set the number of L1 central jets received by GT
    unsigned int gtNumberL1CenJet = stableParameters->gtNumberL1CenJet();
    ///  get / set the number of L1 forward jets received by GT
    unsigned int gtNumberL1ForJet = stableParameters->gtNumberL1ForJet();
    ///  get / set the number of L1 tau jets received by GT
    unsigned int gtNumberL1TauJet = stableParameters->gtNumberL1TauJet();
    ///  get / set the number of L1 jet counts received by GT
    unsigned int gtNumberL1JetCounts = stableParameters->gtNumberL1JetCounts();

    /// hardware stuff

    ///   get / set the number of condition chips in GTL
    unsigned int gtNumberConditionChips = stableParameters->gtNumberConditionChips();
    ///   get / set the number of pins on the GTL condition chips
    unsigned int gtPinsOnConditionChip = stableParameters->gtPinsOnConditionChip();
    ///   get / set the correspondence "condition chip - GTL algorithm word"
    ///   in the hardware
    std::vector<int> gtOrderConditionChip = stableParameters->gtOrderConditionChip();
    ///   get / set the number of PSB boards in GT
    int gtNumberPsbBoards = stableParameters->gtNumberPsbBoards();
    ///   get / set the number of bits for eta of calorimeter objects
    unsigned int gtIfCaloEtaNumberBits = stableParameters->gtIfCaloEtaNumberBits();
    ///   get / set the number of bits for eta of muon objects
    unsigned int gtIfMuEtaNumberBits = stableParameters->gtIfMuEtaNumberBits();
    ///    get / set WordLength
    int gtWordLength = stableParameters->gtWordLength();
    ///    get / set one UnitLength
    int gtUnitLength = stableParameters->gtUnitLength();

    std::cout
      << "\n\t gtNumberPhysTriggers = " << gtNumberPhysTriggers
      << "\n\t gtNumberPhysTriggersExtended = " << gtNumberPhysTriggersExtended
      << "\n\t gtNumberTechnicalTriggers = " << gtNumberTechnicalTriggers
      << "\n\t gtNumberL1Mu = " << gtNumberL1Mu
      << "\n\t gtNumberL1NoIsoEG = " << gtNumberL1NoIsoEG
      << "\n\t gtNumberL1IsoEG = " << gtNumberL1IsoEG
      << "\n\t gtNumberL1CenJet = " << gtNumberL1CenJet
      << "\n\t gtNumberL1ForJet = " << gtNumberL1ForJet
      << "\n\t gtNumberL1TauJet = " << gtNumberL1TauJet
      << "\n\t gtNumberL1JetCounts = " << gtNumberL1JetCounts
      << "\n\t gtNumberConditionChips = " << gtNumberConditionChips
      << "\n\t gtPinsOnConditionChip = " << gtPinsOnConditionChip
      << "\n\t gtNumberPsbBoards = " << gtNumberPsbBoards
      << "\n\t gtIfCaloEtaNumberBits = " << gtIfCaloEtaNumberBits
      << "\n\t gtIfMuEtaNumberBits = " << gtIfMuEtaNumberBits
      << "\n\t gtWordLength = " << gtWordLength
      << "\n\t gtUnitLength = " << gtUnitLength
      << std::endl;

    for( int i=0; i<int(gtOrderConditionChip.size()); i++ ){
      std::cout << "\t\t " << i << "\t" << gtOrderConditionChip[i] << std::endl;
    }

    // End add


    l1t::TriggerMenuXmlParser gtXmlParser = l1t::TriggerMenuXmlParser();

    gtXmlParser.setGtNumberConditionChips(numberConditionChips);
    gtXmlParser.setGtPinsOnConditionChip(pinsOnConditionChip);
    gtXmlParser.setGtOrderConditionChip(orderConditionChip);
    gtXmlParser.setGtNumberPhysTriggers(numberPhysTriggers);
    gtXmlParser.setGtNumberTechTriggers(numberTechTriggers);
    gtXmlParser.setGtNumberL1JetCounts(numberL1JetCounts);

    gtXmlParser.parseXmlFile(m_defXmlFile, m_vmeXmlFile);

    // transfer the condition map and algorithm map from parser to L1uGtTriggerMenu

    boost::shared_ptr<TriggerMenu> pL1uGtTriggerMenu = boost::shared_ptr<TriggerMenu>(
                new TriggerMenu(gtXmlParser.gtTriggerMenuName(), numberConditionChips,
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
DEFINE_FWK_EVENTSETUP_MODULE(l1t::TriggerMenuXmlProducer);
