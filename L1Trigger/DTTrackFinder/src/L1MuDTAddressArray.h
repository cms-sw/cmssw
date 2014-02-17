//-------------------------------------------------
//
/**  \class L1MuDTAddressArray
 *
 *   Array of relative addresses
 *
 *   Array of 4 relative addresses (one per station);<BR>
 *   the valid range of a relative address is 0 - 11
 *   thus a relative address is a 4 bit word;<BR>
 *   address = 15 indicates a negative extrapolation result
 *
 *   \verbatim
 *         ------------------------
 *         |   4    5  |   6   7  |
 *      P  ------------+-----------
 *      H  |   0    1  |   2   3  |
 *      I  ------------+-----------
 *         |   8    9  |  10  11  |
 *         ------------+-----------
 *            my Wheel  next Wheel
 *   \endverbatim
 *
 *
 *   $Date: 2007/02/27 11:44:00 $
 *   $Revision: 1.2 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_ADDRESS_ARRAY_H
#define L1MUDT_ADDRESS_ARRAY_H

//---------------
// C++ Headers --
//---------------

#include <iosfwd>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTAddressArray {

  public:

    /// default constructor
    L1MuDTAddressArray();

    /// copy constructor
    L1MuDTAddressArray(const L1MuDTAddressArray&);

    /// destructor
    virtual ~L1MuDTAddressArray();

    /// assignment operator
    L1MuDTAddressArray& operator=(const L1MuDTAddressArray&);

    /// equal operator
    bool operator==(const L1MuDTAddressArray&) const;
   
    /// unequal operator
    bool operator!=(const L1MuDTAddressArray&) const;

    /// reset address array
    void reset();

    /// set address of a given station [1-4]
    void setStation(int stat, int adr);
    
    /// set addresses of all four stations
    void setStations(int adr1, int adr2, int adr3, int adr4);

    /// get address of a given station [1-4]
    inline unsigned short station(int stat) const { return m_station[stat-1]; } 

    /// get track address code (for eta track finder)
    int trackAddressCode() const;

    /// get converted Addresses
    L1MuDTAddressArray converted() const;

    /// output stream operator for address array
    friend std::ostream& operator<<(std::ostream&, const L1MuDTAddressArray&);

    /// convert address to corresponding VHDL addresse
    static unsigned short int convert(unsigned short int adr);
    
    /// is it a same wheel address?
    static bool sameWheel(unsigned short int adr);
    
    /// is it a next wheel address?
    static bool nextWheel(unsigned short int adr);

  private:
  
    unsigned short int m_station[4];

};

#endif
