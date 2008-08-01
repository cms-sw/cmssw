#ifndef CondFormats_L1TObjects_L1GtParameters_h
#define CondFormats_L1TObjects_L1GtParameters_h

/**
 * \class L1GtParameters
 * 
 * 
 * Description: L1 GT parameters.  
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

// system include files
#include <ostream>

#include <boost/cstdint.hpp>

// user include files
//   base class

// forward declarations

// class declaration
class L1GtParameters
{

public:

    // constructor
    L1GtParameters();

    // destructor
    virtual ~L1GtParameters();

public:

    /// get / set the total Bx's in the event
    inline int gtTotalBxInEvent() const
    {
        return m_totalBxInEvent;
    }

    void setGtTotalBxInEvent(const int&);


    /// get / set the active boards for L1 GT DAQ record
    inline boost::uint16_t gtDaqActiveBoards() const
    {
        return m_daqActiveBoards;
    }

    void setGtDaqActiveBoards(const boost::uint16_t&);


    /// get / set the active boards for L1 GT EVM record
    inline boost::uint16_t gtEvmActiveBoards() const
    {
        return m_evmActiveBoards;
    }

    void setGtEvmActiveBoards(const boost::uint16_t&);


    /// print all the L1 GT parameters
    void print(std::ostream&) const;

private:

    /// total Bx's in the event
    int m_totalBxInEvent;

    /// active boards in the L1 DAQ record
    boost::uint16_t m_daqActiveBoards;

    /// active boards in the L1 EVM record
    boost::uint16_t m_evmActiveBoards;

};

#endif /*CondFormats_L1TObjects_L1GtParameters_h*/
