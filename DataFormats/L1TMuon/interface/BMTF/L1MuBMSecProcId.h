//-------------------------------------------------
//
/**  \class L1MuBMSecProcId
 *
 *   Sector Processor identifier:
 *
 *   There are 6 sector processors along the eta direction 
 *   numbered: -3 -2 -1 +1 +2 +3 (= wheel)<BR>
 *   and 12 sector processors along the phi direction [0,11].<BR>
 *   This is necessary because wheel 0 needs two sector processors
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_SEC_PROC_ID_H
#define L1MUBM_SEC_PROC_ID_H

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

class L1MuBMSecProcId {

  public:

    /// default constructor
    L1MuBMSecProcId();

    /// constructor
    L1MuBMSecProcId(int wheel_id, int sector_id);
  
    /// copy constructor
    L1MuBMSecProcId(const L1MuBMSecProcId&);
 
    /// destructor
    virtual ~L1MuBMSecProcId();

    /// return wheel number
    inline int wheel() const { return m_wheel; }
    
    /// return sector number
    inline int sector() const { return m_sector; }
    
    /// is it an overlap region Sector Processor?
    inline bool ovl() const { return (abs(m_wheel) == 3) ? true : false; }

    /// return physical wheel number (-2,-1,0,+1,+2)
    int locwheel() const;

    /// assignment operator
    L1MuBMSecProcId& operator=(const L1MuBMSecProcId&);
    
    /// equal operator
    bool operator==(const L1MuBMSecProcId&) const;
    
    /// unequal operator
    bool operator!=(const L1MuBMSecProcId&) const;
    
    /// less operator
    bool operator<(const L1MuBMSecProcId&) const;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1MuBMSecProcId&);

  private:

    int m_wheel;
    int m_sector;

};

#endif
