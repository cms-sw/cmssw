#ifndef CondFormats_L1TObjects_L1GtBoard_h
#define CondFormats_L1TObjects_L1GtBoard_h

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

// system include files
#include <string>
#include <boost/cstdint.hpp>

// user include files
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"

// forward declarations

// class declaration
class L1GtBoard
{

public:

    /// constructors
    L1GtBoard();

    L1GtBoard(const L1GtBoardType&);

    L1GtBoard(const L1GtBoardType&, const int&);

    /// destructor
    virtual ~L1GtBoard();

    /// copy constructor
    L1GtBoard(const L1GtBoard&);

    /// assignment operator
    L1GtBoard& operator=(const L1GtBoard&);

    /// equal operator
    bool operator==(const L1GtBoard&) const;

    /// unequal operator
    bool operator!=(const L1GtBoard&) const;

    /// less than operator
    bool operator< (const L1GtBoard&) const;

public:

    /// return board type
    const L1GtBoardType boardType() const
    {
        return m_boardType;
    }

    /// return board index
    const int boardIndex() const
    {
        return m_boardIndex;
    }

    /// return board name - it depends on L1GtBoardType enum!!!
    std::string boardName() const;

private:

    L1GtBoardType m_boardType;

    int m_boardIndex;

};

#endif /*CondFormats_L1TObjects_L1GtBoard_h*/
