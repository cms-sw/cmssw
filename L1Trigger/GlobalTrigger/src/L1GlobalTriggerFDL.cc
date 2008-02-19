/**
 * \class L1GlobalTriggerFDL
 * 
 * 
 * Description: Final Decision Logic board.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M. Fierro            - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFDL.h"

// system include files
#include <iostream>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/Framework/interface/Event.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"


// forward declarations


// constructor
L1GlobalTriggerFDL::L1GlobalTriggerFDL()
{

    // create empty FDL word
    m_gtFdlWord = new L1GtFdlWord();

    // logical switches
    m_firstEv = true;
    m_firstEvLumiSegment = true;
    m_firstEvRun = true;

    // can not reserve memory here for m_prescaleCounter - no access to EventSetup

    // prescale counters: NumberPhysTriggers counters per bunch cross
    //m_prescaleCounter.reserve(numberPhysTriggers*totalBxInEvent);

}

// destructor
L1GlobalTriggerFDL::~L1GlobalTriggerFDL()
{

    reset();
    delete m_gtFdlWord;

}

// Operations

// run FDL
void L1GlobalTriggerFDL::run(
    edm::Event& iEvent,
    const std::vector<int>& prescaleFactors,
    const std::vector<unsigned int>& triggerMaskV,   
    const std::vector<L1GtBoard>& boardMaps,
    const int totalBxInEvent,
    const int iBxInEvent,
    const L1GlobalTriggerGTL* ptrGTL)
{

    const unsigned int numberPhysTriggers =
        L1GlobalTriggerReadoutSetup::NumberPhysTriggers;

    // get gtlDecisionWord from GTL
    std::bitset<numberPhysTriggers> gtlDecisionWord = ptrGTL->getAlgorithmOR();
    std::bitset<numberPhysTriggers> fdlDecisionWord = gtlDecisionWord;

    // prescale counters are reset at the beginning of the luminosity segment

    if (m_firstEv) {

        // prescale counters: NumberPhysTriggers counters per bunch cross
        m_prescaleCounter.reserve(numberPhysTriggers*totalBxInEvent);

        for (int iBxInEvent = 0; iBxInEvent <= totalBxInEvent; ++iBxInEvent) {

            m_prescaleCounter.push_back(prescaleFactors);
        }


        m_firstEv = false;
    }

    // TODO FIXME find the beginning of the luminosity segment
    if (m_firstEvLumiSegment) {

        m_prescaleCounter.clear();

        for (int iBxInEvent = 0; iBxInEvent <= totalBxInEvent; ++iBxInEvent) {

            m_prescaleCounter.push_back(prescaleFactors);
        }

        m_firstEvLumiSegment = false;

    }


    // prescale it, if necessary

    // iBxInEvent is ... -2 -1 0 1 2 ... while counters are 0 1 2 3 4 ...
    int inBxInEvent =  totalBxInEvent/2 + iBxInEvent;

    for (unsigned int iBit = 0; iBit < numberPhysTriggers; ++iBit) {

        if (prescaleFactors.at(iBit) != 1) {

            bool bitValue = gtlDecisionWord.test( iBit );
            if (bitValue) {

                (m_prescaleCounter.at(inBxInEvent).at(iBit))--;
                if (m_prescaleCounter.at(inBxInEvent).at(iBit) == 0) {

                    // bit already true in fdlDecisionWord, just reset counter
                    m_prescaleCounter.at(inBxInEvent).at(iBit) = prescaleFactors.at(iBit);

                    //LogTrace("L1GlobalTriggerFDL")
                    //<< "\nPrescaled algorithm: " << iBit << ". Reset counter to "
                    //<< prescaleFactors.at(iBit) << "\n"
                    //<< std::endl;

                } else {

                    // change bit to false
                    fdlDecisionWord.set( iBit, false );

                    //LogTrace("L1GlobalTriggerFDL")
                    //<< "\nPrescaled algorithm: " << iBit << ". Result set to false"
                    //<< std::endl;

                }
            }
        }
    }

    // set the trigger mask: block the corresponding algorithm if bit value is 1
    // one converts from vector in EventSetup as there there is no "bitset" parameter type

    std::bitset<numberPhysTriggers> triggerMask;
    for (unsigned int i = 0; i < numberPhysTriggers; ++i) {
        if ( triggerMaskV[i] ) {
            triggerMask.set(i);
        }
    }

    fdlDecisionWord = fdlDecisionWord & ~(triggerMask);


    // add trigger veto mask TODO FIXME


    // [ ... convert decision word from std::bitset to std::vector<bool>
    //       TODO remove this block when changing DecisionWord to std::bitset


    DecisionWord fdlDecisionWordVec(numberPhysTriggers);

    for (unsigned int iBit = 0; iBit < numberPhysTriggers; ++iBit) {

        bool bitValue = fdlDecisionWord.test( iBit );
        fdlDecisionWordVec[ iBit ] = bitValue;
    }

    // ... ]

    // fill everything we know in the L1GtFdlWord

    typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;
    for (CItBoardMaps
            itBoard = boardMaps.begin();
            itBoard != boardMaps.end(); ++itBoard) {


        if ((itBoard->gtBoardType() == FDL)) {

            m_gtFdlWord->setBoardId( itBoard->gtBoardId() );

            // BxInEvent
            m_gtFdlWord->setBxInEvent(iBxInEvent);

            // decision word
            m_gtFdlWord->setGtDecisionWord(fdlDecisionWordVec);

            // finalOR
            // TODO FIXME set DAQ partition where L1A is sent; now: hardwired, first partition
            // add technical trigger to the final OR
            int daqPartitionL1A = 0;
            uint16_t finalOrValue = 0;

            if ( fdlDecisionWord.any() ) {
                finalOrValue = 1 << daqPartitionL1A;
            }

            m_gtFdlWord->setFinalOR(finalOrValue);

            //


        }

    }




}

// fill the FDL block in the L1 GT DAQ record for iBxInEvent
void L1GlobalTriggerFDL::fillDaqFdlBlock(
    const boost::uint16_t& activeBoardsGtDaq,
    const std::vector<L1GtBoard>& boardMaps,
    std::auto_ptr<L1GlobalTriggerReadoutRecord>& gtDaqReadoutRecord)
{

    typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;
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

            if (activeBoard && (itBoard->gtBoardType() == FDL)) {

                gtDaqReadoutRecord->setGtFdlWord(*m_gtFdlWord);


            }

        }

    }


}

// fill the FDL block in the L1 GT EVM record for iBxInEvent
void L1GlobalTriggerFDL::fillEvmFdlBlock(
    const boost::uint16_t& activeBoardsGtEvm,
    const std::vector<L1GtBoard>& boardMaps,
    std::auto_ptr<L1GlobalTriggerEvmReadoutRecord>& gtEvmReadoutRecord)
{

    typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;
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

            if (activeBoard && (itBoard->gtBoardType() == FDL)) {

                gtEvmReadoutRecord->setGtFdlWord(*m_gtFdlWord);


            }

        }

    }

}


// clear FDL
void L1GlobalTriggerFDL::reset()
{

    m_gtFdlWord->reset();

    // do NOT reset the prescale counters

}
