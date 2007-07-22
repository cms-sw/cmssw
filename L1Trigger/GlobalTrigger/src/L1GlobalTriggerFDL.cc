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
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConfig.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// forward declarations


// constructor
L1GlobalTriggerFDL::L1GlobalTriggerFDL(
    L1GlobalTrigger& gt)
        : m_GT(gt)
{

    // create empty FDL word
    m_gtFdlWord = new L1GtFdlWord();

    //
    const unsigned int numberTriggerBits =
        L1GlobalTriggerReadoutSetup::NumberPhysTriggers;

    // get number of bunch crosses in event
    m_totalBxInEvent =
        m_GT.gtSetup()->getParameterSet()->getParameter<int>("totalBxInEvent");

    if (m_totalBxInEvent > 0) {
        if ( (m_totalBxInEvent%2) == 0 ) {
            m_totalBxInEvent = m_totalBxInEvent - 1;

            edm::LogInfo("L1GlobalTriggerFDL")
            << "\nWARNING: Number of bunch crossing in event rounded to: "
            << m_totalBxInEvent << "\n         The number must be an odd number!\n"
            << std::endl;
        }
    } else {

        edm::LogInfo("L1GlobalTriggerFDL")
        << "\nWARNING: Number of bunch crossing in event must be a positive number!"
        << "\n  Requested value was: " << m_totalBxInEvent
        << "\n  Reset to 1 (L1Accept bunch only).\n"
        << std::endl;

        m_totalBxInEvent = 1;

    }

    // algorithm prescale factors (default to 1)
    m_prescaleFactor.reserve(numberTriggerBits);
    m_prescaleFactor.assign(numberTriggerBits, 1);

    // get the non-default scale factors
    // TODO FIXME EventSetup
    m_prescaleFactor =
        m_GT.gtSetup()->getParameterSet()->getParameter<std::vector<int> >("PrescaleFactors");

    if ( edm::isDebugEnabled() ) {
        LogTrace("L1GlobalTriggerFDL")
        << "\nPrescale factors for algorithms\n"
        << std::endl;

        for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {
            LogTrace("L1GlobalTriggerFDL")
            << iBit << "\t" << m_prescaleFactor.at(iBit)
            << std::endl;
        }
    }

    // prescale counters: NumberPhysTriggers counters per bunch cross
    m_prescaleCounter.reserve(numberTriggerBits*m_totalBxInEvent);

    for (int iBxInEvent = 0; iBxInEvent <= m_totalBxInEvent;
            ++iBxInEvent) {

        m_prescaleCounter.push_back(m_prescaleFactor);
    }

}

// destructor
L1GlobalTriggerFDL::~L1GlobalTriggerFDL()
{

    reset();
    delete m_gtFdlWord;

}

// Operations

// run FDL
void L1GlobalTriggerFDL::run(int iBxInEvent)
{

    const L1GlobalTriggerConfig* gtConf = m_GT.gtSetup()->gtConfig();


    const unsigned int numberTriggerBits =
        L1GlobalTriggerReadoutSetup::NumberPhysTriggers;

    // get gtlDecisionWord from GTL
    std::bitset<numberTriggerBits> gtlDecisionWord = m_GT.gtGTL()->getAlgorithmOR();
    std::bitset<numberTriggerBits> fdlDecisionWord = gtlDecisionWord;

    // prescale it, if necessary

    // iBxInEvent is ... -2 -1 0 1 2 ... while counters are 0 1 2 3 4 ...
    int inBxInEvent =  m_totalBxInEvent/2 + iBxInEvent;

    for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {

        if (m_prescaleFactor.at(iBit) != 1) {

            bool bitValue = gtlDecisionWord.test( iBit );
            if (bitValue) {

                (m_prescaleCounter.at(inBxInEvent).at(iBit))--;
                if (m_prescaleCounter.at(inBxInEvent).at(iBit) == 0) {

                    // bit already true in fdlDecisionWord, just reset counter
                    m_prescaleCounter.at(inBxInEvent).at(iBit) = m_prescaleFactor.at(iBit);

                    //LogTrace("L1GlobalTriggerFDL")
                    //<< "\nPrescaled algorithm: " << iBit << ". Reset counter to "
                    //<< m_prescaleFactor.at(iBit) << "\n"
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

    // add trigger mask

    if (gtConf != 0) {
        fdlDecisionWord = fdlDecisionWord & ~(gtConf->getTriggerMask());
    }


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
