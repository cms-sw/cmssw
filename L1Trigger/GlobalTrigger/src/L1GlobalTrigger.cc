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

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConfig.h"

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
    m_muGmtInputTag = parSet.getUntrackedParameter<edm::InputTag>(
                          "GmtInputTag", edm::InputTag("L1GmtEmulDigis"));

    // input tag for calorimeter collection from GCT
    m_caloGctInputTag = parSet.getUntrackedParameter<edm::InputTag>(
                            "GctInputTag", edm::InputTag("L1GctEmulDigis"));


    // logical flag to produce the L1 GT DAQ readout record
    //     if true, produce the record
    m_produceL1GtDaqRecord = parSet.getUntrackedParameter<bool>(
                                 "ProduceL1GtDaqRecord", true);


    // logical flag to produce the L1 GT EVM readout record
    //     if true, produce the record
    m_produceL1GtEvmRecord = parSet.getUntrackedParameter<bool>(
                                 "ProduceL1GtEvmRecord", true);


    // logical flag to produce the L1 GT object map record
    //     if true, produce the record
    m_produceL1GtObjectMapRecord = parSet.getUntrackedParameter<bool>(
                                       "ProduceL1GtObjectMapRecord", true);


    // logical flag to write the PSB content in the  L1 GT DAQ record
    //     if true, write the PSB content in the record
    m_writePsbL1GtDaqRecord = parSet.getUntrackedParameter<bool>(
                                  "WritePsbL1GtDaqRecord", true);

    LogTrace("L1GlobalTrigger")
    << "\nInput tag for muon collection from GMT:         "
    << m_muGmtInputTag.label()
    << "\nInput tag for calorimeter collections from GCT: "
    << m_caloGctInputTag.label()
    << "\nProduce the L1 GT DAQ readout record:           "
    << m_produceL1GtDaqRecord
    << "\nProduce the L1 GT EVM readout record:           "
    << m_produceL1GtEvmRecord
    << "\nProduce the L1 GT Object Map record:            "
    << m_produceL1GtObjectMapRecord << " \n"
    << "\nWrite Psb content to L1 GT DAQ Record:          "
    << m_writePsbL1GtDaqRecord << " \n"
    << std::endl;


    // set L1 GT configuration parameters
    if(!m_gtSetup) {
        m_gtSetup = new L1GlobalTriggerSetup(*this, parSet);
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
    m_gtPSB = new L1GlobalTriggerPSB(*this);

    // create new GTL
    m_gtGTL = new L1GlobalTriggerGTL(*this);

    // create new FDL
    m_gtFDL = new L1GlobalTriggerFDL(*this);



}

// destructor
L1GlobalTrigger::~L1GlobalTrigger()
{

    if(m_gtSetup)
        delete m_gtSetup;
    m_gtSetup = 0;

    delete m_gtPSB;
    delete m_gtGTL;
    delete m_gtFDL;
}

// member functions

// method called to produce the data
void L1GlobalTrigger::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    // process event iEvent

    // get parameters

    edm::ESHandle< L1GtStableParameters > l1GtStablePar;
    evSetup.get< L1GtStableParametersRcd >().get( l1GtStablePar );

    // number of physics triggers
    unsigned int numberPhysTriggers = l1GtStablePar->gtNumberPhysTriggers();

    edm::ESHandle< L1GtParameters > l1GtPar;
    evSetup.get< L1GtParametersRcd >().get( l1GtPar );

    //    total number of Bx's in the event
    int totalBxInEvent = l1GtPar->gtTotalBxInEvent();

    int minBxInEvent = (totalBxInEvent + 1)/2 - totalBxInEvent;
    int maxBxInEvent = (totalBxInEvent + 1)/2 - 1;

    //    active boards in L1 GT DAQ record and in L1 GT EVM record
    boost::uint16_t activeBoardsGtDaq = l1GtPar->gtDaqActiveBoards();
    boost::uint16_t activeBoardsGtEvm = l1GtPar->gtEvmActiveBoards();

    LogDebug("L1GlobalTrigger")
    << "\nTotal number of bunch crosses to put in the GT readout record: "
    << totalBxInEvent << " = " << "["
    << minBxInEvent << ", " << maxBxInEvent << "] BX\n"
    << "\n  Active boards in L1 GT DAQ record (hex format) = "
    << std::hex << std::setw(sizeof(activeBoardsGtDaq)*2) << std::setfill('0')
    << activeBoardsGtDaq
    << std::dec << std::setfill(' ')
    << std::endl;

    // get board maps
    edm::ESHandle< L1GtBoardMaps > l1GtBM;
    evSetup.get< L1GtBoardMapsRcd >().get( l1GtBM );

    const std::vector<L1GtBoard> boardMaps = l1GtBM->gtBoardMaps();
    typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;

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
                activeBoard = activeBoardsGtDaq & (1 << iActiveBit);
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

    // number of objects of each type
    //    { Mu, NoIsoEG, IsoEG, CenJet, ForJet, TauJet, ETM, ETT, HTT, JetCounts };
    unsigned int nrL1Mu = l1GtStablePar->gtNumberL1Mu();

    unsigned int nrL1NoIsoEG = l1GtStablePar->gtNumberL1NoIsoEG();
    unsigned int nrL1IsoEG = l1GtStablePar->gtNumberL1IsoEG();

    unsigned int nrL1CenJet = l1GtStablePar->gtNumberL1CenJet();
    unsigned int nrL1ForJet = l1GtStablePar->gtNumberL1ForJet();
    unsigned int nrL1TauJet = l1GtStablePar->gtNumberL1TauJet();

    // ... the rest of the objects are global

    // produce the L1GlobalTriggerReadoutRecord now, after we found how many
    // BxInEvent the record has and how many boards are active
    std::auto_ptr<L1GlobalTriggerReadoutRecord> gtDaqReadoutRecord(
        new L1GlobalTriggerReadoutRecord(
            totalBxInEvent, daqNrFdlBoards, daqNrPsbBoards) );


    // * produce the L1GlobalTriggerEvmReadoutRecord
    std::auto_ptr<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord(
        new L1GlobalTriggerEvmReadoutRecord(totalBxInEvent, daqNrFdlBoards) );
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
                    activeBoard = activeBoardsGtDaq & (1 << iActiveBit);
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
                                    static_cast<boost::uint16_t>(totalBxInEvent));

                                // set the list of active boards
                                gtfeWordValue.setActiveBoards(activeBoardsGtDaq);

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
                    activeBoard = activeBoardsGtEvm & (1 << iActiveBit);
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
                                    static_cast<boost::uint16_t>(totalBxInEvent));

                                // set the list of active boards
                                gtfeWordValue.setActiveBoards(activeBoardsGtEvm);

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
            receiveNoIsoEG, nrL1NoIsoEG,
            receiveIsoEG, nrL1IsoEG,
            receiveCenJet, nrL1CenJet,
            receiveForJet, nrL1ForJet,
            receiveTauJet, nrL1TauJet,
            receiveETM, receiveETT, receiveHTT,
            receiveJetCounts);

        if (m_produceL1GtDaqRecord && m_writePsbL1GtDaqRecord) {
            m_gtPSB->fillPsbBlock(iEvent, evSetup, iBxInEvent, gtDaqReadoutRecord);
        }

        // * receive GMT object data via GTL
        //LogDebug("L1GlobalTrigger")
        //<< "\nL1GlobalTrigger : receiving GMT data for bx = " << iBxInEvent << "\n"
        //<< std::endl;

        m_gtGTL->receiveGmtObjectData(
            iEvent,
            m_muGmtInputTag, iBxInEvent,
            receiveMu, nrL1Mu);

        // * run GTL
        //LogDebug("L1GlobalTrigger")
        //<< "\nL1GlobalTrigger : running GTL for bx = " << iBxInEvent << "\n"
        //<< std::endl;

        m_gtGTL->run(iEvent, evSetup, m_gtPSB, 
            m_produceL1GtObjectMapRecord, iBxInEvent, gtObjectMapRecord, 
            numberPhysTriggers);

        //LogDebug("L1GlobalTrigger")
        //<< "\n AlgorithmOR\n" << m_gtGTL->getAlgorithmOR() << "\n"
        //<< std::endl;

        ////maps only for BX = 0
        //if (m_produceL1GtObjectMapRecord && (iBxInEvent == 0)) {
        //
        //    const std::vector<L1GlobalTriggerObjectMap>* objMapVec = m_gtGTL->objectMap();
        //
        //    gtObjectMapRecord->setGtObjectMap(*objMapVec);
        //    delete objMapVec;
        //
        //}


        // * run FDL
        //LogDebug("L1GlobalTrigger")
        //<< "\nL1GlobalTrigger : running FDL for bx = " << iBxInEvent << "\n"
        //<< std::endl;

        m_gtFDL->run(iEvent, evSetup,
                     boardMaps,
                     totalBxInEvent,
                     iBxInEvent);

        if (m_produceL1GtDaqRecord && (daqNrFdlBoards > 0)) {
            m_gtFDL->fillDaqFdlBlock(
                activeBoardsGtDaq, boardMaps,
                gtDaqReadoutRecord);
        }


        if (m_produceL1GtEvmRecord && (evmNrFdlBoards > 0)) {
            m_gtFDL->fillEvmFdlBlock(
                activeBoardsGtEvm, boardMaps,
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
        //<< m_muGmtInputTag.label()
        //<< "\n**** \n"
        //<< std::endl;

        // get L1MuGMTReadoutCollection reference and set it in GT record

        edm::Handle<L1MuGMTReadoutCollection> gmtRcHandle;
        iEvent.getByLabel(m_muGmtInputTag.label(), gmtRcHandle);

        gtDaqReadoutRecord->setMuCollectionRefProd(gmtRcHandle);

    }

    if ( edm::isDebugEnabled() ) {
        std::ostringstream myCoutStream;
        gtDaqReadoutRecord->print(myCoutStream);
        LogTrace("L1GlobalTrigger")
        << "\n The following L1 GT DAQ readout record was produced:\n"
        << myCoutStream.str() << "\n"
        << std::endl;
    }



    if ( edm::isDebugEnabled() ) {
        std::ostringstream myCoutStream;
        gtEvmReadoutRecord->print(myCoutStream);
        LogTrace("L1GlobalTrigger")
        << "\n The following L1 GT EVM readout record was produced:\n"
        << myCoutStream.str() << "\n"
        << std::endl;
    }


    if ( edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;

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

L1GlobalTriggerSetup* L1GlobalTrigger::m_gtSetup = 0;
