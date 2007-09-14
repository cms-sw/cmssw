/**
 * \class L1GtBoard
 * 
 * 
 * Description: simple class for L1 GT board.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"

// system include files

// user include files
//   base class

// forward declarations

// constructors
L1GtBoard::L1GtBoard()
{

    // empty

}

L1GtBoard::L1GtBoard(const L1GtBoardType& boardTypeValue)
{

    m_boardType = boardTypeValue;

    m_boardIndex = -1;
}

L1GtBoard::L1GtBoard(const L1GtBoardType& boardTypeValue, const int& boardIndexValue)
{

    m_boardType = boardTypeValue;

    m_boardIndex = boardIndexValue;
}

// destructor
L1GtBoard::~L1GtBoard()
{
    // empty
}

// copy constructor
L1GtBoard::L1GtBoard(const L1GtBoard& gtb)
{

    m_boardType = gtb.boardType();
    m_boardIndex = gtb.boardIndex();

}

// assignment operator
L1GtBoard::L1GtBoard& L1GtBoard::operator=(const L1GtBoard& gtb)
{

    if ( this != &gtb ) {

        m_boardType = gtb.boardType();
        m_boardIndex = gtb.boardIndex();

    }

    return *this;

}

// equal operator
bool L1GtBoard::operator==(const L1GtBoard& gtb) const
{

    if (m_boardType != gtb.boardType()) {
        return false;
    }
    if (m_boardIndex != gtb.boardIndex()) {
        return false;
    }

    // all members identical
    return true;

}



// unequal operator
bool L1GtBoard::operator!=(const L1GtBoard& result) const
{

    return !( result == *this);

}

// less than operator
bool L1GtBoard::operator< (const L1GtBoard& gtb) const
{
    if (m_boardType < gtb.boardType()) {
        return true;
    } else {
        if (m_boardType == gtb.boardType()) {

            if (m_boardIndex < gtb.boardIndex()) {
                return true;
            }
        }
    }

    return false;
}


// return board name - it depends on L1GtBoardType enum!!!
std::string L1GtBoard::boardName() const
{

    std::string boardNameValue;

    // active board, add its size
    switch (m_boardType) {

        case GTFE: {
                boardNameValue = "GTFE";
            }
            break;
        case FDL: {
                boardNameValue = "FDL";
            }
            break;
        case PSB: {
                boardNameValue = "PSB";
            }
            break;
        case GMT: {
                boardNameValue = "GMT";
            }
            break;
        case TCS: {
                boardNameValue = "TCS";
            }
            break;
        case TIM: {
                boardNameValue = "TIM";
            }
            break;
        default: {

                // do nothing here
                // TODO throw exception instead of returning empty string?
            }
            break;
    }


    return boardNameValue;

}

