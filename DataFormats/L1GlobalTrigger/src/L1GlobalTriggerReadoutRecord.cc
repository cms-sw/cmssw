/**
 * \class L1GlobalTriggerReadoutRecord
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
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

// system include files
#include <iomanip>


// user include files




#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructors
L1GlobalTriggerReadoutRecord::L1GlobalTriggerReadoutRecord()
{

    // empty GTFE
    m_gtfeWord = L1GtfeWord();

    // no FDL, no PSB
}

L1GlobalTriggerReadoutRecord::L1GlobalTriggerReadoutRecord(int numberBxInEvent)
{

    m_gtfeWord = L1GtfeWord();

    m_gtFdlWord.reserve(numberBxInEvent);
    m_gtFdlWord.assign(numberBxInEvent, L1GtFdlWord());

    // min value of bxInEvent
    int minBxInEvent = (numberBxInEvent + 1)/2 - numberBxInEvent;
    //int maxBxInEvent = (numberBxInEvent + 1)/2 - 1; // not needed

    // matrix index [0, numberBxInEvent) -> bxInEvent [minBxInEvent, maxBxInEvent]
    // warning: matrix index != bxInEvent
    for (int iFdl = 0; iFdl < numberBxInEvent; ++iFdl) {
        int iBxInEvent = minBxInEvent + iFdl;
        m_gtFdlWord[iFdl].setBxInEvent(iBxInEvent);
    }

    // PSBs
    int numberPsb = L1GlobalTriggerReadoutSetup::NumberPsbBoards;
    int totalNumberPsb = numberPsb*numberBxInEvent;

    m_gtPsbWord.reserve(totalNumberPsb);
    m_gtPsbWord.assign(totalNumberPsb, L1GtPsbWord());


}

L1GlobalTriggerReadoutRecord::L1GlobalTriggerReadoutRecord(
    const int numberBxInEvent,
    const int numberFdlBoards,
    const int numberPsbBoards)
{

    // GTFE board
    m_gtfeWord = L1GtfeWord();

    // FDL board
    if (numberFdlBoards > 0) {
        m_gtFdlWord.reserve(numberBxInEvent);
    }

    // PSB boards
    if (numberPsbBoards > 0) {
        m_gtPsbWord.reserve(numberPsbBoards*numberBxInEvent);
    }

}

// copy constructor
L1GlobalTriggerReadoutRecord::L1GlobalTriggerReadoutRecord(
    const L1GlobalTriggerReadoutRecord& result)
{

    m_gtfeWord = result.m_gtfeWord;
    m_gtFdlWord = result.m_gtFdlWord;
    m_gtPsbWord = result.m_gtPsbWord;

    m_muCollRefProd = result.m_muCollRefProd;


}

// destructor
L1GlobalTriggerReadoutRecord::~L1GlobalTriggerReadoutRecord()
{

    // empty now

}

// assignment operator
L1GlobalTriggerReadoutRecord& L1GlobalTriggerReadoutRecord::operator=(
    const L1GlobalTriggerReadoutRecord& result)
{

    if ( this != &result ) {

        m_gtfeWord = result.m_gtfeWord;
        m_gtFdlWord  = result.m_gtFdlWord;
        m_gtPsbWord  = result.m_gtPsbWord;

        m_muCollRefProd = result.m_muCollRefProd;

    }

    return *this;

}

// equal operator
bool L1GlobalTriggerReadoutRecord::operator==(
    const L1GlobalTriggerReadoutRecord& result) const
{

    if (m_gtfeWord != result.m_gtfeWord) {
        return false;
    }

    if (m_gtFdlWord  != result.m_gtFdlWord) {
        return false;
    }

    if (m_gtPsbWord  != result.m_gtPsbWord) {
        return false;
    }

    if (m_muCollRefProd != result.m_muCollRefProd) {
        return false;
    }

    // all members identical
    return true;

}

// unequal operator
bool L1GlobalTriggerReadoutRecord::operator!=(
    const L1GlobalTriggerReadoutRecord& result) const
{

    return !( result == *this);

}
// methods


// get Global Trigger decision
//    general bxInEvent
const bool L1GlobalTriggerReadoutRecord::decision(int bxInEventValue) const
{

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx).finalOR();
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)
    // TODO re-evaluate action

    //    throw cms::Exception("NotFound")
    LogTrace("L1GlobalTriggerReadoutRecord")
    << "\nError: requested GtFdlWord for bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << "Can not return global decision for this bx!\n"
    << std::endl;

    return false;
}

//    bxInEvent = 0
const bool L1GlobalTriggerReadoutRecord::decision() const
{

    int bxInEventL1Accept = 0;
    return decision(bxInEventL1Accept);
}

// get final OR for all DAQ partitions
//    general bxInEvent
const boost::uint16_t L1GlobalTriggerReadoutRecord::finalOR(int bxInEventValue) const
{

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx).finalOR();
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)
    // TODO re-evaluate action

    //    throw cms::Exception("NotFound")
    LogTrace("L1GlobalTriggerReadoutRecord")
    << "\nError: requested GtFdlWord for bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << "Can not return finalOR for this bx!\n"
    << std::endl;

    return 0;
}

//    bxInEvent = 0
const boost::uint16_t L1GlobalTriggerReadoutRecord::finalOR() const
{

    int bxInEventL1Accept = 0;
    return finalOR(bxInEventL1Accept);
}


// get Global Trigger decision word

const DecisionWord &
L1GlobalTriggerReadoutRecord::decisionWord(int bxInEventValue) const
{
    static DecisionWord emptyDecisionWord;

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx).gtDecisionWord();
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)
    // TODO re-evaluate action

    //    throw cms::Exception("NotFound")
    LogTrace("L1GlobalTriggerReadoutRecord")
    << "\nError: requested GtFdlWord for bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << "Can not return decision word for this bx!\n"
    << std::endl;

    return emptyDecisionWord;
}

const DecisionWord &
L1GlobalTriggerReadoutRecord::decisionWord() const
{

    int bxInEventL1Accept = 0;
    return decisionWord(bxInEventL1Accept);
}



const TechnicalTriggerWord & 
L1GlobalTriggerReadoutRecord::technicalTriggerWord(int bxInEventValue) const {
    static TechnicalTriggerWord emptyTechnicalTriggerWord;

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx).gtTechnicalTriggerWord();
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)
    // TODO re-evaluate action

    //    throw cms::Exception("NotFound")
    LogTrace("L1GlobalTriggerReadoutRecord")
    << "\nError: requested GtFdlWord for bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << "Can not return technical trigger word for this bx!\n"
    << std::endl;

    return emptyTechnicalTriggerWord;
}

const TechnicalTriggerWord & 
L1GlobalTriggerReadoutRecord::technicalTriggerWord() const {

    int bxInEventL1Accept = 0;
    return technicalTriggerWord(bxInEventL1Accept);

}


// set global decision
//    general
void L1GlobalTriggerReadoutRecord::setDecision(const bool& t, int bxInEventValue)
{

    for (std::vector<L1GtFdlWord>::iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            // TODO FIXME when manipulating partitions
            (*itBx).setFinalOR(static_cast<uint16_t> (t));
            return;
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)

    //    throw cms::Exception("NotFound")
    LogTrace("L1GlobalTriggerReadoutRecord")
    << "\nError: requested GtFdlWord for bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << "Can not set global decision for this bx!\n"
    << std::endl;

}

//    bxInEvent = 0
void L1GlobalTriggerReadoutRecord::setDecision(const bool& t)
{

    int bxInEventL1Accept = 0;
    setDecision(t, bxInEventL1Accept);
}

// set decision word
void L1GlobalTriggerReadoutRecord::setDecisionWord(
    const DecisionWord& decisionWordValue,
    int bxInEventValue)
{

    for (std::vector<L1GtFdlWord>::iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            (*itBx).setGtDecisionWord (decisionWordValue);
            return;
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)

    //    throw cms::Exception("NotFound")
    LogTrace("L1GlobalTriggerReadoutRecord")
    << "\nError: requested GtFdlWord for bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << "Can not set decision word for this bx!\n"
    << std::endl;

}

void L1GlobalTriggerReadoutRecord::setDecisionWord(
    const DecisionWord& decisionWordValue)
{

    int bxInEventL1Accept = 0;
    setDecisionWord(decisionWordValue, bxInEventL1Accept);

}

void L1GlobalTriggerReadoutRecord::setTechnicalTriggerWord(
        const TechnicalTriggerWord& ttWordValue, int bxInEventValue)
{

    for (std::vector<L1GtFdlWord>::iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ((*itBx).bxInEvent() == bxInEventValue) {

            (*itBx).setGtTechnicalTriggerWord(ttWordValue);
            return;
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)

    //    throw cms::Exception("NotFound")
    LogTrace("L1GlobalTriggerReadoutRecord")
            << "\nError: requested GtFdlWord for bxInEvent = "
            << bxInEventValue << " does not exist.\n"
            << "Can not set technical trigger word for this bx!\n" << std::endl;

}

void L1GlobalTriggerReadoutRecord::setTechnicalTriggerWord(
        const TechnicalTriggerWord& ttWordValue)
{

    int bxInEventL1Accept = 0;
    setTechnicalTriggerWord(ttWordValue, bxInEventL1Accept);

}



// print global decision and algorithm decision word
void L1GlobalTriggerReadoutRecord::printGtDecision(
    std::ostream& myCout, int bxInEventValue) const
{

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            myCout << "\nL1 Global Trigger Record: " << std::endl;

            myCout << "  Bunch cross " << bxInEventValue
            << std::endl
            << "  Global Decision = " << std::setw(5) << (*itBx).globalDecision()
            << std::endl;

            (*itBx).printGtDecisionWord(myCout);

        }
    }

    myCout << std::endl;

}

void L1GlobalTriggerReadoutRecord::printGtDecision(std::ostream& myCout) const
{

    int bxInEventL1Accept = 0;
    printGtDecision(myCout, bxInEventL1Accept);

}

// print technical trigger word (reverse order for vector<bool>)
void L1GlobalTriggerReadoutRecord::printTechnicalTrigger(
    std::ostream& myCout, int bxInEventValue
) const
{

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            myCout << "\nL1 Global Trigger Record: " << std::endl;

            myCout << "  Bunch cross " << bxInEventValue
            << std::endl;

            (*itBx).printGtTechnicalTriggerWord(myCout);
        }
    }

    myCout << std::endl;

}

void L1GlobalTriggerReadoutRecord::printTechnicalTrigger(std::ostream& myCout) const
{

    int bxInEventL1Accept = 0;
    printTechnicalTrigger(myCout, bxInEventL1Accept);

}


// get / set reference to L1MuGMTReadoutCollection
const edm::RefProd<L1MuGMTReadoutCollection>
L1GlobalTriggerReadoutRecord::muCollectionRefProd() const
{

    return m_muCollRefProd;
}

void L1GlobalTriggerReadoutRecord::setMuCollectionRefProd(
    edm::Handle<L1MuGMTReadoutCollection>& muHandle)
{

    m_muCollRefProd = edm::RefProd<L1MuGMTReadoutCollection>(muHandle);

}

void L1GlobalTriggerReadoutRecord::setMuCollectionRefProd(
    const edm::RefProd<L1MuGMTReadoutCollection>& refProdMuGMT)
{

    m_muCollRefProd = refProdMuGMT;

}


// get/set hardware-related words

// get / set GTFE word (record) in the GT readout record
const L1GtfeWord L1GlobalTriggerReadoutRecord::gtfeWord() const
{

    return m_gtfeWord;

}

void L1GlobalTriggerReadoutRecord::setGtfeWord(const L1GtfeWord& gtfeWordValue)
{

    m_gtfeWord = gtfeWordValue;

}

// get / set FDL word (record) in the GT readout record
const L1GtFdlWord L1GlobalTriggerReadoutRecord::gtFdlWord(int bxInEventValue) const
{

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx);
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)

    //    throw cms::Exception("NotFound")
    LogTrace("L1GlobalTriggerReadoutRecord")
    << "\nError: requested L1GtFdlWord for bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << std::endl;

    // return empty record - actually does not arrive here
    return L1GtFdlWord();

}

const L1GtFdlWord L1GlobalTriggerReadoutRecord::gtFdlWord() const
{

    int bxInEventL1Accept = 0;
    return gtFdlWord(bxInEventL1Accept);
}

void L1GlobalTriggerReadoutRecord::setGtFdlWord(
    const L1GtFdlWord& gtFdlWordValue, int bxInEventValue)
{

    // if a L1GtFdlWord exists for bxInEventValue, replace it
    for (std::vector<L1GtFdlWord>::iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            *itBx = gtFdlWordValue;
            LogTrace("L1GlobalTriggerReadoutRecord")
            << "L1GlobalTriggerReadoutRecord: replacing L1GtFdlWord for bxInEvent = "
            << bxInEventValue << "\n"
            << std::endl;
            return;
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)
    // all L1GtFdlWord are created in the record constructor for allowed bunch crosses

    //    throw cms::Exception("NotFound")
    LogTrace("L1GlobalTriggerReadoutRecord")
    << "\nError: Cannot set L1GtFdlWord for bxInEvent = " << bxInEventValue
    << std::endl;

}

void L1GlobalTriggerReadoutRecord::setGtFdlWord(const L1GtFdlWord& gtFdlWordValue)
{

    // just push back the new FDL block
    m_gtFdlWord.push_back(gtFdlWordValue);

}


// get / set PSB word (record) in the GT readout record
const L1GtPsbWord L1GlobalTriggerReadoutRecord::gtPsbWord(
    boost::uint16_t boardIdValue, int bxInEventValue) const
{

    for (std::vector<L1GtPsbWord>::const_iterator itBx = m_gtPsbWord.begin();
            itBx != m_gtPsbWord.end(); ++itBx) {

        if (
            ((*itBx).bxInEvent() == bxInEventValue) &&
            ((*itBx).boardId() == boardIdValue)) {

            return (*itBx);

        }
    }

    // if bunch cross or boardId not found, throw exception (action: SkipEvent)

    //    throw cms::Exception("NotFound")
    LogTrace("L1GlobalTriggerReadoutRecord")
    << "\nError: requested L1GtPsbWord for boardId = "
    << std::hex << boardIdValue << std::dec
    << " and bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << std::endl;

    // return empty record - actually does not arrive here
    return L1GtPsbWord();

}

const L1GtPsbWord L1GlobalTriggerReadoutRecord::gtPsbWord(boost::uint16_t boardIdValue) const
{

    int bxInEventL1Accept = 0;
    return gtPsbWord(boardIdValue, bxInEventL1Accept);
}

void L1GlobalTriggerReadoutRecord::setGtPsbWord(
    const L1GtPsbWord& gtPsbWordValue, boost::uint16_t boardIdValue, int bxInEventValue)
{

    // if a L1GtPsbWord with the same bxInEventValue and boardIdValue exists, replace it
    for (std::vector<L1GtPsbWord>::iterator itBx = m_gtPsbWord.begin();
            itBx != m_gtPsbWord.end(); ++itBx) {

        if (
            ((*itBx).bxInEvent() == bxInEventValue) &&
            ((*itBx).boardId() == boardIdValue)) {

            *itBx = gtPsbWordValue;

            LogTrace("L1GlobalTriggerReadoutRecord")
            << "\nL1GlobalTriggerReadoutRecord: replacing L1GtPsbWord with boardId = "
            << std::hex << boardIdValue << std::dec
            << " and bxInEvent = " << bxInEventValue
            << "\n"
            << std::endl;
            return;
        }
    }

    // otherwise, write in the first empty PSB
    // empty means: PSB with bxInEvent = 0, boardId = 0

    for (std::vector<L1GtPsbWord>::iterator itBx = m_gtPsbWord.begin();
            itBx != m_gtPsbWord.end(); ++itBx) {

        if (
            ((*itBx).bxInEvent() == 0) &&
            ((*itBx).boardId() == 0)) {

            *itBx = gtPsbWordValue;

            LogTrace("L1GlobalTriggerReadoutRecord")
            << "\nL1GlobalTriggerReadoutRecord: filling an empty L1GtPsbWord"
            << " for PSB with boardId = "
            << std::hex << boardIdValue << std::dec
            << " and bxInEvent = " << bxInEventValue
            << "\n"
            << std::endl;
            return;
        }
    }

    // no PSB to replace, no empty PSB: throw exception (action: SkipEvent)
    // all L1GtPsbWord are created in the record constructor

    //    throw cms::Exception("NotFound")
    LogTrace("L1GlobalTriggerReadoutRecord")
    << "\nError: Cannot set L1GtPsbWord for PSB with boardId = "
    << std::hex << boardIdValue << std::dec
    << " and bxInEvent = " << bxInEventValue
    << "\n  No PSB to replace and no empty PSB found!\n"
    << std::endl;


}

void L1GlobalTriggerReadoutRecord::setGtPsbWord(
    const L1GtPsbWord& gtPsbWordValue, boost::uint16_t boardIdValue)
{

    int bxInEventL1Accept = 0;
    setGtPsbWord(gtPsbWordValue, boardIdValue, bxInEventL1Accept);

}

void L1GlobalTriggerReadoutRecord::setGtPsbWord(const L1GtPsbWord& gtPsbWordValue)
{
    // just push back the new PSB block
    m_gtPsbWord.push_back(gtPsbWordValue);

}

// other methods

// clear the record
// it resets the content of the members only!
void L1GlobalTriggerReadoutRecord::reset()
{

    m_gtfeWord.reset();

    for (std::vector<L1GtFdlWord>::iterator itFdl = m_gtFdlWord.begin();
            itFdl != m_gtFdlWord.end(); ++itFdl) {

        itFdl->reset();

    }

    for (std::vector<L1GtPsbWord>::iterator itPsb = m_gtPsbWord.begin();
            itPsb != m_gtPsbWord.end(); ++itPsb) {

        itPsb->reset();

    }

    // TODO FIXME reset m_muCollRefProd

}

/// pretty print the content of a L1GlobalTriggerReadoutRecord
void L1GlobalTriggerReadoutRecord::print(std::ostream& myCout) const
{

    myCout << "\n L1GlobalTriggerReadoutRecord::print \n" << std::endl;

    m_gtfeWord.print(myCout);

    for (std::vector<L1GtFdlWord>::const_iterator itFdl = m_gtFdlWord.begin();
            itFdl != m_gtFdlWord.end(); ++itFdl) {

        itFdl->print(myCout);

    }

    for (std::vector<L1GtPsbWord>::const_iterator itPsb = m_gtPsbWord.begin();
            itPsb != m_gtPsbWord.end(); ++itPsb) {

        itPsb->print(myCout);

    }

    // FIXME add  L1MuGMTReadoutCollection printing
    //    edm::RefProd<L1MuGMTReadoutCollection> m_muCollRefProd;




}

// output stream operator
std::ostream& operator<<(std::ostream& s, const L1GlobalTriggerReadoutRecord& result)
{
    // TODO FIXME put together all prints
    s << "Not available yet - sorry";

    return s;

}
