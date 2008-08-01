/**
 * \class L1GlobalTrigger
 * 
 * 
 * Description: see header file.  
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
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"

// system include files
#include <memory>
#include <iostream>
#include <iomanip>
#include <bitset>

#include <boost/cstdint.hpp>

// user include files
//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"
#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoTechTrigRcd.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFDL.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeExtWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1TcsWord.h"

#include "DataFormats/Common/interface/RefProd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"



// constructors

L1GlobalTrigger::L1GlobalTrigger(const edm::ParameterSet& parSet)
{

    // input tag for muon collection from GMT
    m_muGmtInputTag = parSet.getParameter<edm::InputTag>("GmtInputTag");

    // input tag for calorimeter collection from GCT
    m_caloGctInputTag = parSet.getParameter<edm::InputTag>("GctInputTag");

    /// input tag for technical triggers
    m_technicalTriggersInputTag = parSet.getParameter<edm::InputTag>("TechnicalTriggersInputTag");

    // logical flag to produce the L1 GT DAQ readout record
    //     if true, produce the record
    m_produceL1GtDaqRecord = parSet.getParameter<bool>("ProduceL1GtDaqRecord");


    // logical flag to produce the L1 GT EVM readout record
    //     if true, produce the record
    m_produceL1GtEvmRecord = parSet.getParameter<bool>("ProduceL1GtEvmRecord");


    // logical flag to produce the L1 GT object map record
    //     if true, produce the record
    m_produceL1GtObjectMapRecord = parSet.getParameter<bool>("ProduceL1GtObjectMapRecord");


    // logical flag to write the PSB content in the  L1 GT DAQ record
    //     if true, write the PSB content in the record
    m_writePsbL1GtDaqRecord = parSet.getParameter<bool>("WritePsbL1GtDaqRecord");
 
    // logical flag to read the technical trigger records
    //     if true, it will read via getMany the available records
    m_readTechnicalTriggerRecords = parSet.getParameter<bool>("ReadTechnicalTriggerRecords");
    
    // number of "bunch crossing in the event" (BxInEvent) to be emulated
    // symmetric around L1Accept (BxInEvent = 0):
    //    1 (BxInEvent = 0); 3 (F 0 1) (standard record); 5 (E F 0 1 2) (debug record)
    // even numbers (except 0) "rounded" to the nearest lower odd number
    m_emulateBxInEvent = parSet.getParameter<int>("EmulateBxInEvent");

    LogTrace("L1GlobalTrigger")
    << "\nInput tag for muon collection from GMT:         "
    << m_muGmtInputTag
    << "\nInput tag for calorimeter collections from GCT: "
    << m_caloGctInputTag
    << "\nInput tag for technical triggers                "
    << m_technicalTriggersInputTag
    << "\nProduce the L1 GT DAQ readout record:           "
    << m_produceL1GtDaqRecord
    << "\nProduce the L1 GT EVM readout record:           "
    << m_produceL1GtEvmRecord
    << "\nProduce the L1 GT Object Map record:            "
    << m_produceL1GtObjectMapRecord << " \n"
    << "\nWrite Psb content to L1 GT DAQ Record:          "
    << m_writePsbL1GtDaqRecord << " \n"
    << "\nRead technical trigger records:                 "
    << m_readTechnicalTriggerRecords << " \n"
    << "\nNumber of BxInEvent to be emulated:             "
    << m_emulateBxInEvent << " \n"
    << std::endl;

    if ((m_emulateBxInEvent > 0)  && ( (m_emulateBxInEvent%2) == 0) ) {
        m_emulateBxInEvent = m_emulateBxInEvent - 1;

        edm::LogInfo("L1GlobalTrigger")
        << "\nWARNING: Number of bunch crossing to be emulated rounded to: "
        << m_emulateBxInEvent 
        << "\n         The number must be an odd number!\n"
        << std::endl;
    }

    // register products
    if (m_produceL1GtDaqRecord) {
        produces<L1GlobalTriggerReadoutRecord>();
    }

    if (m_produceL1GtEvmRecord) {
        produces<L1GlobalTriggerEvmReadoutRecord>();
    }

    if (m_produceL1GtObjectMapRecord) {
        produces<L1GlobalTriggerObjectMapRecord>();
    }


    // create new PSBs
    m_gtPSB = new L1GlobalTriggerPSB(m_technicalTriggersInputTag.label());

    // create new GTL
    m_gtGTL = new L1GlobalTriggerGTL();

    // create new FDL
    m_gtFDL = new L1GlobalTriggerFDL();


    // initialize cached IDs
    
    //
    m_l1GtStableParCacheID = 0ULL;

    m_numberPhysTriggers = 0;
    m_numberTechnicalTriggers = 0;
    m_numberDaqPartitions = 0;
    
    m_nrL1Mu = 0;

    m_nrL1NoIsoEG = 0;
    m_nrL1IsoEG = 0;

    m_nrL1CenJet = 0;
    m_nrL1ForJet = 0;
    m_nrL1TauJet = 0;
    
    m_nrL1JetCounts = 0;


    m_ifMuEtaNumberBits = 0;
    m_ifCaloEtaNumberBits = 0;
    
    //
    m_l1GtParCacheID = 0ULL;
    
    m_totalBxInEvent = 0;
    
    m_activeBoardsGtDaq = 0;
    m_activeBoardsGtEvm = 0;

    //
    m_l1GtBMCacheID = 0ULL;
    
    //
    m_l1GtPfAlgoCacheID = 0ULL;
    m_l1GtPfTechCacheID = 0ULL;
    
    m_l1GtTmAlgoCacheID = 0ULL;
    m_l1GtTmTechCacheID = 0ULL;
    
    m_l1GtTmVetoAlgoCacheID = 0ULL;
    m_l1GtTmVetoTechCacheID = 0ULL;
    
 
}

// destructor
L1GlobalTrigger::~L1GlobalTrigger()
{

    delete m_gtPSB;
    delete m_gtGTL;
    delete m_gtFDL;
}

// member functions

// method called to produce the data
void L1GlobalTrigger::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    // process event iEvent

	// get / update the stable parameters from the EventSetup 
    // local cache & check on cacheIdentifier

    unsigned long long l1GtStableParCacheID = evSetup.get<L1GtStableParametersRcd>().cacheIdentifier();

    if (m_l1GtStableParCacheID != l1GtStableParCacheID) {
        
        edm::ESHandle< L1GtStableParameters > l1GtStablePar;
        evSetup.get< L1GtStableParametersRcd >().get( l1GtStablePar );        
        m_l1GtStablePar = l1GtStablePar.product();
        
        // number of physics triggers
        m_numberPhysTriggers = m_l1GtStablePar->gtNumberPhysTriggers();

        // number of technical triggers
        m_numberTechnicalTriggers = m_l1GtStablePar->gtNumberTechnicalTriggers();
        
        // number of DAQ partitions
        m_numberDaqPartitions = 8; // FIXME add it to stable parameters

        // number of objects of each type
        //    { Mu, NoIsoEG, IsoEG, CenJet, ForJet, TauJet, ETM, ETT, HTT, JetCounts };
        m_nrL1Mu = static_cast<int> (m_l1GtStablePar->gtNumberL1Mu());

        m_nrL1NoIsoEG = static_cast<int> (m_l1GtStablePar->gtNumberL1NoIsoEG());
        m_nrL1IsoEG = static_cast<int> (m_l1GtStablePar->gtNumberL1IsoEG());

        m_nrL1CenJet = static_cast<int> (m_l1GtStablePar->gtNumberL1CenJet());
        m_nrL1ForJet = static_cast<int> (m_l1GtStablePar->gtNumberL1ForJet());
        m_nrL1TauJet = static_cast<int> (m_l1GtStablePar->gtNumberL1TauJet());

        m_nrL1JetCounts = static_cast<int> (m_l1GtStablePar->gtNumberL1JetCounts());

        // ... the rest of the objects are global

        m_ifMuEtaNumberBits = static_cast<int> (m_l1GtStablePar->gtIfMuEtaNumberBits());
        m_ifCaloEtaNumberBits = static_cast<int> (m_l1GtStablePar->gtIfCaloEtaNumberBits());
        
        // (re)initialize L1GlobalTriggerGTL
        m_gtGTL->init(m_nrL1Mu, m_numberPhysTriggers);

        // (re)initialize L1GlobalTriggerPSB
        m_gtPSB->init(m_nrL1NoIsoEG, m_nrL1IsoEG, 
                m_nrL1CenJet, m_nrL1ForJet, m_nrL1TauJet,
                m_numberTechnicalTriggers);
        
        //
        m_l1GtStableParCacheID = l1GtStableParCacheID;

    }

    // get / update the parameters from the EventSetup 
    // local cache & check on cacheIdentifier

    unsigned long long l1GtParCacheID = evSetup.get<L1GtParametersRcd>().cacheIdentifier();

    if (m_l1GtParCacheID != l1GtParCacheID) {
        
        edm::ESHandle< L1GtParameters > l1GtPar;
        evSetup.get< L1GtParametersRcd >().get( l1GtPar );        
        m_l1GtPar = l1GtPar.product();
       
        //    total number of Bx's in the event coming from EventSetup
        m_totalBxInEvent = m_l1GtPar->gtTotalBxInEvent();
 
        //    active boards in L1 GT DAQ record and in L1 GT EVM record
        m_activeBoardsGtDaq = m_l1GtPar->gtDaqActiveBoards();
        m_activeBoardsGtEvm = m_l1GtPar->gtEvmActiveBoards();


        m_l1GtParCacheID = l1GtParCacheID;

    }
    
    // negative value: emulate TotalBxInEvent as given in EventSetup 
    if (m_emulateBxInEvent < 0) {
        m_emulateBxInEvent = m_totalBxInEvent;
    }

    int minBxInEvent = (m_emulateBxInEvent + 1)/2 - m_emulateBxInEvent;
    int maxBxInEvent = (m_emulateBxInEvent + 1)/2 - 1;

    LogDebug("L1GlobalTrigger")
    << "\nTotal number of bunch crosses to put in the GT readout record: "
    << m_emulateBxInEvent << " = " << "["
    << minBxInEvent << ", " << maxBxInEvent << "] BX\n"
    << "\n  Active boards in L1 GT DAQ record (hex format) = "
    << std::hex << std::setw(sizeof(m_activeBoardsGtDaq)*2) << std::setfill('0')
    << m_activeBoardsGtDaq
    << std::dec << std::setfill(' ')
    << std::endl;

    // get / update the board maps from the EventSetup 
    // local cache & check on cacheIdentifier

    typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;

    unsigned long long l1GtBMCacheID = evSetup.get<L1GtBoardMapsRcd>().cacheIdentifier();

    if (m_l1GtBMCacheID != l1GtBMCacheID) {
        
        edm::ESHandle< L1GtBoardMaps > l1GtBM;
        evSetup.get< L1GtBoardMapsRcd >().get( l1GtBM );        
        m_l1GtBM = l1GtBM.product();
        
        m_l1GtBMCacheID = l1GtBMCacheID;

    }

    // TODO need changes in CondFormats to cache the maps
    const std::vector<L1GtBoard>& boardMaps = m_l1GtBM->gtBoardMaps();

    // get / update the prescale factors from the EventSetup 
    // local cache & check on cacheIdentifier

    unsigned long long l1GtPfAlgoCacheID = 
        evSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtPfAlgoCacheID != l1GtPfAlgoCacheID) {
        
        edm::ESHandle< L1GtPrescaleFactors > l1GtPfAlgo;
        evSetup.get< L1GtPrescaleFactorsAlgoTrigRcd >().get( l1GtPfAlgo );        
        m_l1GtPfAlgo = l1GtPfAlgo.product();
        
        m_prescaleFactorsAlgoTrig = m_l1GtPfAlgo->gtPrescaleFactors();
        
        m_l1GtPfAlgoCacheID = l1GtPfAlgoCacheID;

    }

    unsigned long long l1GtPfTechCacheID = 
        evSetup.get<L1GtPrescaleFactorsTechTrigRcd>().cacheIdentifier();

    if (m_l1GtPfTechCacheID != l1GtPfTechCacheID) {
        
        edm::ESHandle< L1GtPrescaleFactors > l1GtPfTech;
        evSetup.get< L1GtPrescaleFactorsTechTrigRcd >().get( l1GtPfTech );        
        m_l1GtPfTech = l1GtPfTech.product();
        
        m_prescaleFactorsTechTrig = m_l1GtPfTech->gtPrescaleFactors();
        
        m_l1GtPfTechCacheID = l1GtPfTechCacheID;

    }

    
    // get / update the trigger mask from the EventSetup 
    // local cache & check on cacheIdentifier

    unsigned long long l1GtTmAlgoCacheID = 
        evSetup.get<L1GtTriggerMaskAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtTmAlgoCacheID != l1GtTmAlgoCacheID) {
        
        edm::ESHandle< L1GtTriggerMask > l1GtTmAlgo;
        evSetup.get< L1GtTriggerMaskAlgoTrigRcd >().get( l1GtTmAlgo );        
        m_l1GtTmAlgo = l1GtTmAlgo.product();
        
        m_triggerMaskAlgoTrig = m_l1GtTmAlgo->gtTriggerMask();
        
        m_l1GtTmAlgoCacheID = l1GtTmAlgoCacheID;

    }
    

    unsigned long long l1GtTmTechCacheID = 
        evSetup.get<L1GtTriggerMaskTechTrigRcd>().cacheIdentifier();

    if (m_l1GtTmTechCacheID != l1GtTmTechCacheID) {
        
        edm::ESHandle< L1GtTriggerMask > l1GtTmTech;
        evSetup.get< L1GtTriggerMaskTechTrigRcd >().get( l1GtTmTech );        
        m_l1GtTmTech = l1GtTmTech.product();
        
        m_triggerMaskTechTrig = m_l1GtTmTech->gtTriggerMask();
        
        m_l1GtTmTechCacheID = l1GtTmTechCacheID;

    }
    
    unsigned long long l1GtTmVetoAlgoCacheID = 
        evSetup.get<L1GtTriggerMaskVetoAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtTmVetoAlgoCacheID != l1GtTmVetoAlgoCacheID) {
        
        edm::ESHandle< L1GtTriggerMask > l1GtTmVetoAlgo;
        evSetup.get< L1GtTriggerMaskVetoAlgoTrigRcd >().get( l1GtTmVetoAlgo );        
        m_l1GtTmVetoAlgo = l1GtTmVetoAlgo.product();
        
        m_triggerMaskVetoAlgoTrig = m_l1GtTmVetoAlgo->gtTriggerMask();
        
        m_l1GtTmVetoAlgoCacheID = l1GtTmVetoAlgoCacheID;

    }
    

    unsigned long long l1GtTmVetoTechCacheID = 
        evSetup.get<L1GtTriggerMaskVetoTechTrigRcd>().cacheIdentifier();

    if (m_l1GtTmVetoTechCacheID != l1GtTmVetoTechCacheID) {
        
        edm::ESHandle< L1GtTriggerMask > l1GtTmVetoTech;
        evSetup.get< L1GtTriggerMaskVetoTechTrigRcd >().get( l1GtTmVetoTech );        
        m_l1GtTmVetoTech = l1GtTmVetoTech.product();
        
        m_triggerMaskVetoTechTrig = m_l1GtTmVetoTech->gtTriggerMask();
        
        m_l1GtTmVetoTechCacheID = l1GtTmVetoTechCacheID;

    }
    
    // loop over blocks in the GT DAQ record receiving data, count them if they are active
    // all board type are defined in CondFormats/L1TObjects/L1GtFwd
    // enum L1GtBoardType { GTFE, FDL, PSB, GMT, TCS, TIM };
    // &
    // set the active flag for each object type received from GMT and GCT
    // all objects in the GT system are defined in enum L1GtObject from
    // DataFormats/L1Trigger/L1GlobalTriggerReadoutSetupFwd
    // enum L1GtObject
    //    { Mu, NoIsoEG, IsoEG, CenJet, ForJet, TauJet, ETM, ETT, HTT, JetCounts };

    int daqNrGtfeBoards = 0;

    int daqNrFdlBoards = 0;
    int daqNrPsbBoards = 0;
    int daqNrGmtBoards = 0;
    int daqNrTcsBoards = 0;
    int daqNrTimBoards = 0;

    //
    bool receiveMu = false;
    bool receiveNoIsoEG = false;
    bool receiveIsoEG = false;
    bool receiveCenJet = false;
    bool receiveForJet = false;
    bool receiveTauJet = false;
    bool receiveETM = false;
    bool receiveETT = false;
    bool receiveHTT = false;
    bool receiveJetCounts = false;

    bool receiveTechTr = false;

    for (CItBoardMaps
            itBoard = boardMaps.begin();
            itBoard != boardMaps.end(); ++itBoard) {

        int iPosition = itBoard->gtPositionDaqRecord();
        if (iPosition > 0) {

            int iActiveBit = itBoard->gtBitDaqActiveBoards();
            bool activeBoard = false;

            if (iActiveBit >= 0) {
                activeBoard = m_activeBoardsGtDaq & (1 << iActiveBit);
            }

            // use board if: in the record, but not in ActiveBoardsMap (iActiveBit < 0)
            //               in the record and ActiveBoardsMap, and active
            if ((iActiveBit < 0) || activeBoard) {

                switch (itBoard->gtBoardType()) {

                    case GTFE: {
                            daqNrGtfeBoards++;
                        }

                        break;
                    case FDL: {
                            daqNrFdlBoards++;
                        }

                        break;
                    case PSB: {
                            daqNrPsbBoards++;

                            // get the objects coming to this PSB
                            std::vector<L1GtPsbQuad> quadInPsb = itBoard->gtQuadInPsb();
                            for (std::vector<L1GtPsbQuad>::const_iterator
                                    itQuad = quadInPsb.begin();
                                    itQuad != quadInPsb.end(); ++itQuad) {

                                switch (*itQuad) {

                                    case TechTr: {
                                            receiveTechTr = true;
                                        }

                                        break;
                                    case NoIsoEGQ: {
                                            receiveNoIsoEG = true;
                                        }

                                        break;
                                    case IsoEGQ: {
                                            receiveIsoEG = true;
                                        }

                                        break;
                                    case CenJetQ: {
                                            receiveCenJet = true;
                                        }

                                        break;
                                    case ForJetQ: {
                                            receiveForJet = true;
                                        }

                                        break;
                                    case TauJetQ: {
                                            receiveTauJet = true;
                                        }

                                        break;
                                    case ESumsQ: {
                                            receiveETM = true;
                                            receiveETT = true;
                                            receiveHTT = true;
                                        }

                                        break;
                                    case JetCountsQ: {
                                            receiveJetCounts = true;
                                        }

                                        break;
                                        // FIXME add MIP/Iso bits
                                    default: {
                                            // do nothing
                                        }

                                        break;
                                }

                            }

                        }

                        break;
                    case GMT: {
                            daqNrGmtBoards++;
                            receiveMu = true;
                        }

                        break;
                    case TCS: {
                            daqNrTcsBoards++;
                        }

                        break;
                    case TIM: {
                            daqNrTimBoards++;
                        }

                        break;
                    default: {
                            // do nothing, all blocks are given in GtBoardType enum
                        }

                        break;
                }
            }
        }

    }

    // produce the L1GlobalTriggerReadoutRecord now, after we found how many
    // BxInEvent the record has and how many boards are active
    std::auto_ptr<L1GlobalTriggerReadoutRecord> gtDaqReadoutRecord(
        new L1GlobalTriggerReadoutRecord(
            m_emulateBxInEvent, daqNrFdlBoards, daqNrPsbBoards) );


    // * produce the L1GlobalTriggerEvmReadoutRecord
    std::auto_ptr<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord(
        new L1GlobalTriggerEvmReadoutRecord(m_emulateBxInEvent, daqNrFdlBoards) );
    // daqNrFdlBoards OK, just reserve memory at this point

    // * produce the L1GlobalTriggerObjectMapRecord
    std::auto_ptr<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord(
        new L1GlobalTriggerObjectMapRecord() );


    // fill the boards not depending on the BxInEvent in the L1 GT DAQ record
    // GMT, PSB and FDL depend on BxInEvent

    if (m_produceL1GtDaqRecord) {

        for (CItBoardMaps
                itBoard = boardMaps.begin();
                itBoard != boardMaps.end(); ++itBoard) {

            int iPosition = itBoard->gtPositionDaqRecord();
            if (iPosition > 0) {

                int iActiveBit = itBoard->gtBitDaqActiveBoards();
                bool activeBoard = false;

                if (iActiveBit >= 0) {
                    activeBoard = m_activeBoardsGtDaq & (1 << iActiveBit);
                }

                // use board if: in the record, but not in ActiveBoardsMap (iActiveBit < 0)
                //               in the record and ActiveBoardsMap, and active
                if ((iActiveBit < 0) || activeBoard) {

                    switch (itBoard->gtBoardType()) {

                        case GTFE: {
                                L1GtfeWord gtfeWordValue;

                                gtfeWordValue.setBoardId( itBoard->gtBoardId() );

                                // cast int to boost::uint16_t
                                // there are normally 3 or 5 BxInEvent
                                gtfeWordValue.setRecordLength(
                                    static_cast<boost::uint16_t>(m_emulateBxInEvent));

                                // set the list of active boards
                                gtfeWordValue.setActiveBoards(m_activeBoardsGtDaq);

                                // set the TOTAL_TRIGNR as read from iEvent
                                // TODO check again - PTC stuff

                                gtfeWordValue.setTotalTriggerNr(
                                    static_cast<boost::uint32_t>(iEvent.id().event()));

                                // ** fill L1GtfeWord in GT DAQ record

                                gtDaqReadoutRecord->setGtfeWord(gtfeWordValue);
                            }

                            break;
                        case TCS: {
                                // nothing
                            }

                            break;
                        case TIM: {
                                // nothing
                            }

                            break;
                        default: {
                                // do nothing, all blocks are given in GtBoardType enum
                            }

                            break;
                    }
                }
            }

        }

    }

    // fill the boards not depending on the BxInEvent in the L1 GT EVM record

    int evmNrFdlBoards = 0;

    if (m_produceL1GtEvmRecord) {

        for (CItBoardMaps
                itBoard = boardMaps.begin();
                itBoard != boardMaps.end(); ++itBoard) {

            int iPosition = itBoard->gtPositionEvmRecord();
            if (iPosition > 0) {

                int iActiveBit = itBoard->gtBitEvmActiveBoards();
                bool activeBoard = false;

                if (iActiveBit >= 0) {
                    activeBoard = m_activeBoardsGtEvm & (1 << iActiveBit);
                }

                // use board if: in the record, but not in ActiveBoardsMap (iActiveBit < 0)
                //               in the record and ActiveBoardsMap, and active
                if ((iActiveBit < 0) || activeBoard) {

                    switch (itBoard->gtBoardType()) {

                        case GTFE: {
                                L1GtfeExtWord gtfeWordValue;

                                gtfeWordValue.setBoardId( itBoard->gtBoardId() );

                                // cast int to boost::uint16_t
                                // there are normally 3 or 5 BxInEvent
                                gtfeWordValue.setRecordLength(
                                    static_cast<boost::uint16_t>(m_emulateBxInEvent));

                                // set the list of active boards
                                gtfeWordValue.setActiveBoards(m_activeBoardsGtEvm);

                                // set the TOTAL_TRIGNR as read from iEvent
                                // TODO check again - PTC stuff

                                gtfeWordValue.setTotalTriggerNr(
                                    static_cast<boost::uint32_t>(iEvent.id().event()));

                                // ** fill L1GtfeWord in GT EVM record

                                gtEvmReadoutRecord->setGtfeWord(gtfeWordValue);
                            }

                            break;
                        case FDL: {
                                evmNrFdlBoards++;
                            }

                            break;
                        case TCS: {

                                L1TcsWord tcsWordValue;

                                tcsWordValue.setBoardId( itBoard->gtBoardId() );

                                boost::uint16_t trigType = 0x5; // 0101 simulated event
                                tcsWordValue.setTriggerType(trigType);

                                // set the Event_Nr as read from iEvent
                                tcsWordValue.setEventNr(
                                    static_cast<boost::uint32_t>(iEvent.id().event()));

                                // ** fill L1TcsWord in the EVM record

                                gtEvmReadoutRecord->setTcsWord(tcsWordValue);

                            }

                            break;
                        case TIM: {
                                // nothing
                            }

                            break;
                        default: {
                                // do nothing, all blocks are given in GtBoardType enum
                            }

                            break;
                    }
                }
            }

        }

    }

    // loop over BxInEvent
    for (int iBxInEvent = minBxInEvent; iBxInEvent <= maxBxInEvent;
            ++iBxInEvent) {

        // * receive GCT object data via PSBs
        //LogDebug("L1GlobalTrigger")
        //<< "\nL1GlobalTrigger : receiving PSB data for bx = " << iBxInEvent << "\n"
        //<< std::endl;

        m_gtPSB->receiveGctObjectData(
            iEvent,
            m_caloGctInputTag, iBxInEvent,
            receiveNoIsoEG, m_nrL1NoIsoEG,
            receiveIsoEG, m_nrL1IsoEG,
            receiveCenJet, m_nrL1CenJet,
            receiveForJet, m_nrL1ForJet,
            receiveTauJet, m_nrL1TauJet,
            receiveETM, receiveETT, receiveHTT,
            receiveJetCounts);

        /// receive technical trigger
        if (m_readTechnicalTriggerRecords) {
            m_gtPSB->receiveTechnicalTriggers(iEvent,
                    m_technicalTriggersInputTag, iBxInEvent, receiveTechTr,
                    m_numberTechnicalTriggers);
        }

        if (m_produceL1GtDaqRecord && m_writePsbL1GtDaqRecord) {
            m_gtPSB->fillPsbBlock(iEvent, m_activeBoardsGtDaq, boardMaps,
                    iBxInEvent, gtDaqReadoutRecord);
        }

        // * receive GMT object data via GTL
        //LogDebug("L1GlobalTrigger")
        //<< "\nL1GlobalTrigger : receiving GMT data for bx = " << iBxInEvent << "\n"
        //<< std::endl;

        m_gtGTL->receiveGmtObjectData(iEvent, m_muGmtInputTag, iBxInEvent,
                receiveMu, m_nrL1Mu);

        // * run GTL
        //LogDebug("L1GlobalTrigger")
        //<< "\nL1GlobalTrigger : running GTL for bx = " << iBxInEvent << "\n"
        //<< std::endl;

        m_gtGTL->run(iEvent, evSetup, m_gtPSB, 
            m_produceL1GtObjectMapRecord, iBxInEvent, gtObjectMapRecord, 
            m_numberPhysTriggers,
            m_nrL1Mu,
            m_nrL1NoIsoEG,
            m_nrL1IsoEG,
            m_nrL1CenJet,
            m_nrL1ForJet,
            m_nrL1TauJet,
            m_nrL1JetCounts,
            m_ifMuEtaNumberBits,
            m_ifCaloEtaNumberBits);

        //LogDebug("L1GlobalTrigger")
        //<< "\n AlgorithmOR\n" << m_gtGTL->getAlgorithmOR() << "\n"
        //<< std::endl;

        // * run FDL
        //LogDebug("L1GlobalTrigger")
        //<< "\nL1GlobalTrigger : running FDL for bx = " << iBxInEvent << "\n"
        //<< std::endl;

        m_gtFDL->run(iEvent, 
                m_prescaleFactorsAlgoTrig, m_prescaleFactorsTechTrig, 
                m_triggerMaskAlgoTrig, m_triggerMaskTechTrig, 
                m_triggerMaskVetoAlgoTrig, m_triggerMaskVetoTechTrig, 
                boardMaps, m_emulateBxInEvent, iBxInEvent,
                m_numberPhysTriggers, m_numberTechnicalTriggers,
                m_numberDaqPartitions,
                m_gtGTL, m_gtPSB);

        if (m_produceL1GtDaqRecord && (daqNrFdlBoards > 0)) {
            m_gtFDL->fillDaqFdlBlock(
                m_activeBoardsGtDaq, boardMaps,
                gtDaqReadoutRecord);
        }


        if (m_produceL1GtEvmRecord && (evmNrFdlBoards > 0)) {
            m_gtFDL->fillEvmFdlBlock(
                m_activeBoardsGtEvm, boardMaps,
                gtEvmReadoutRecord);
        }

        // reset
        m_gtPSB->reset();
        m_gtGTL->reset();
        m_gtFDL->reset();

        //LogDebug("L1GlobalTrigger") << "\n Reset PSB, GTL, FDL\n" << std::endl;

    }


    if ( daqNrGmtBoards > 0 ) {


        //LogDebug("L1GlobalTrigger")
        //<< "\n**** "
        //<< "\n  Persistent reference for L1MuGMTReadoutCollection with input tag: "
        //<< m_muGmtInputTag
        //<< "\n**** \n"
        //<< std::endl;

        // get L1MuGMTReadoutCollection reference and set it in GT record

        edm::Handle<L1MuGMTReadoutCollection> gmtRcHandle;
        iEvent.getByLabel(m_muGmtInputTag, gmtRcHandle);

        gtDaqReadoutRecord->setMuCollectionRefProd(gmtRcHandle);

    }

    if ( edm::isDebugEnabled() ) {

    	std::ostringstream myCoutStream;
        gtDaqReadoutRecord->print(myCoutStream);
        LogTrace("L1GlobalTrigger")
        << "\n The following L1 GT DAQ readout record was produced:\n"
        << myCoutStream.str() << "\n"
        << std::endl;

        myCoutStream.str("");
        myCoutStream.clear();

        gtEvmReadoutRecord->print(myCoutStream);
        LogTrace("L1GlobalTrigger")
        << "\n The following L1 GT EVM readout record was produced:\n"
        << myCoutStream.str() << "\n"
        << std::endl;

        myCoutStream.str("");
        myCoutStream.clear();

        const std::vector<L1GlobalTriggerObjectMap> objMapVec =
            gtObjectMapRecord->gtObjectMap();

        for (std::vector<L1GlobalTriggerObjectMap>::const_iterator
                it = objMapVec.begin(); it != objMapVec.end(); ++it) {

            (*it).print(myCoutStream);

        }


        LogDebug("L1GlobalTrigger")
        << "Test gtObjectMapRecord in L1GlobalTrigger \n\n" << myCoutStream.str() << "\n\n"
        << std::endl;

        myCoutStream.str("");
        myCoutStream.clear();

    }

    // **
    // register products
    if (m_produceL1GtDaqRecord) {
        iEvent.put( gtDaqReadoutRecord );
    }

    if (m_produceL1GtEvmRecord) {
        iEvent.put( gtEvmReadoutRecord );
    }

    if (m_produceL1GtObjectMapRecord) {
        iEvent.put( gtObjectMapRecord );
    }

}

// static data members

