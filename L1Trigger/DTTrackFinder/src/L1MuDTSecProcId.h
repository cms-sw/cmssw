//-------------------------------------------------
//
/**  \class L1MuDTSecProcId
 *
 *   Sector Processor identifier:
 *
 *   There are 6 sector processors along the eta direction 
 *   numbered: -3 -2 -1 +1 +2 +3 (= wheel)<BR>
 *   and 12 sector processors along the phi direction [0,11].<BR>
 *   This is necessary because wheel 0 needs two sector processors
 *
 *
 *   $Date: 2007/02/27 11:44:00 $
 *   $Revision: 1.2 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_SEC_PROC_ID_H
#define L1MUDT_SEC_PROC_ID_H

//---------------
// C++ Headers --
//---------------

#include <iosfwd>
#include <cstdlib>

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTSecProcId {

  public:

    /// default constructor
    L1MuDTSecProcId();

    /// constructor
    L1MuDTSecProcId(int wheel_id, int sector_id);
  
    /// copy constructor
    L1MuDTSecProcId(const L1MuDTSecProcId&);
 
    /// destructor
    virtual ~L1MuDTSecProcId();

    /// return wheel number
    inline int wheel() const { return m_wheel; }
    
    /// return sector number
    inline int sector() const { return m_sector; }
    
    /// is it an overlap region Sector Processor?
    inline bool ovl() const { return (abs(m_wheel) == 3) ? true : false; }

    /// return physical wheel number (-2,-1,0,+1,+2)
    int locwheel() const;

    /// assignment operator
    L1MuDTSecProcId& operator=(const L1MuDTSecProcId&);
    
    /// equal operator
    bool operator==(const L1MuDTSecProcId&) const;
    
    /// unequal operator
    bool operator!=(const L1MuDTSecProcId&) const;
    
    /// less operator
    bool operator<(const L1MuDTSecProcId&) const;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1MuDTSecProcId&);

  private:

    int m_wheel;
    int m_sector;

};

#endif
