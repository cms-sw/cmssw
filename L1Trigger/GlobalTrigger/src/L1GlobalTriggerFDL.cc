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

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"
#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"


// forward declarations


// constructor
L1GlobalTriggerFDL::L1GlobalTriggerFDL(
    L1GlobalTrigger& gt)
        : m_GT(gt)
{

    // create empty FDL word
    m_gtFdlWord = new L1GtFdlWord();

    // logical switches
    m_firstEv = true;
    m_firstEvLumiSegment = true;
    m_firstEvRun = true;

    // can not reserve memory here for m_prescaleCounter - no access to EventSetup

    // prescale counters: NumberPhysTriggers counters per bunch cross
    //m_prescaleCounter.reserve(numberTriggerBits*totalBxInEvent);

}

// destructor
L1GlobalTriggerFDL::~L1GlobalTriggerFDL()
{

    reset();
    delete m_gtFdlWord;

}

// Operations

// run FDL
void L1GlobalTriggerFDL::run(int iBxInEvent, const edm::EventSetup& evSetup)
{

    // TODO take it from EventSetup
    const unsigned int numberTriggerBits =
        L1GlobalTriggerReadoutSetup::NumberPhysTriggers;

    // get gtlDecisionWord from GTL
    std::bitset<numberTriggerBits> gtlDecisionWord = m_GT.gtGTL()->getAlgorithmOR();
    std::bitset<numberTriggerBits> fdlDecisionWord = gtlDecisionWord;

    // get from EventSetup: total number of Bx's in the event

    edm::ESHandle< L1GtParameters > l1GtPar ;
    evSetup.get< L1GtParametersRcd >().get( l1GtPar ) ;

    int totalBxInEvent = l1GtPar->gtTotalBxInEvent();

    // get from EventSetup: prescale factors
    edm::ESHandle< L1GtPrescaleFactors > l1GtPF ;
    evSetup.get< L1GtPrescaleFactorsRcd >().get( l1GtPF ) ;

    std::vector<int> prescaleFactors = l1GtPF->gtPrescaleFactors();

    // TODO FIXME check with firmware to see where and if the prescale counters are reset

    // prescale counters are reset at the beginning of the luminosity segment

    if (m_firstEv) {

        // prescale counters: NumberPhysTriggers counters per bunch cross
        m_prescaleCounter.reserve(numberTriggerBits*totalBxInEvent);

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

    for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {

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

    edm::ESHandle< L1GtTriggerMask > l1GtTM ;
    evSetup.get< L1GtTriggerMaskRcd >().get( l1GtTM ) ;

    std::vector<unsigned int> triggerMaskV = l1GtTM->gtTriggerMask();

    std::bitset<numberTriggerBits> triggerMask;
    for (unsigned int i = 0; i < numberTriggerBits; ++i) {
        if ( triggerMaskV[i] ) {
            triggerMask.set(i);
        }
    }

    fdlDecisionWord = fdlDecisionWord & ~(triggerMask);


    // add trigger veto mask TODO FIXME


    // [ ... convert decision word from std::bitset to std::vector<bool>
    //       TODO remove this block when changing DecisionWord to std::bitset


    DecisionWord fdlDecisionWordVec(numberTriggerBits);

    for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {

        bool bitValue = fdlDecisionWord.test( iBit );
        fdlDecisionWordVec[ iBit ] = bitValue;
    }

    // ... ]

    // fill everything we know in the L1GtFdlWord

    // get from EventSetup the board maps
    edm::ESHandle< L1GtBoardMaps > l1GtBM;
    evSetup.get< L1GtBoardMapsRcd >().get( l1GtBM );

    // set boardId 
    int iBoard = 0; // just one FDL board, one could use the record map however    
    L1GtBoard fdlBoard = L1GtBoard(FDL, iBoard);    
    m_gtFdlWord->setBoardId( l1GtBM->boardId(fdlBoard) );

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

// clear FDL
void L1GlobalTriggerFDL::reset()
{

    m_gtFdlWord->reset();

    // do NOT reset the prescale counters

}
