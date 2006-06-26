//-------------------------------------------------
//
/**  \class L1MuDTAssignmentUnit
 *
 *   Assignment Unit:
 *
 *   assigns pt, charge, phi
 *   and quality to a muon candidate
 *   found by the Track Assembler
 *
 *
 *   $Date: 2006/06/01 00:00:00 $
 *   $Revision: 1.1 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_ASSIGNMENT_UNIT_H
#define L1MUDT_ASSIGNMENT_UNIT_H

//---------------
// C++ Headers --
//---------------

#include <vector>
#include <iosfwd>

//----------------------
// Base Class Headers --
//----------------------

#include "DataFormats/L1DTTrackFinder/interface/L1AbstractProcessor.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTAddressArray.h"
class L1MuDTPhiLut;
class L1MuDTPtaLut;
class L1MuDTTrackSegPhi;
class L1MuDTSectorProcessor;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTAssignmentUnit : public L1AbstractProcessor {

  public:

    /// maximal number of pt assignment methods
    static const int MAX_PTASSMETH = 28;
                       
    /// pt assignment methods
    enum PtAssMethod { PT12L,  PT12H,  PT13L,  PT13H,  PT14L,  PT14H,
                       PT23L,  PT23H,  PT24L,  PT24H,  PT34L,  PT34H, 
                       PT12LO, PT12HO, PT13LO, PT13HO, PT14LO, PT14HO,
                       PT23LO, PT23HO, PT24LO, PT24HO, PT34LO, PT34HO, 
                       PT15LO, PT15HO, PT25LO, PT25HO, UNDEF };

    /// constructor
    L1MuDTAssignmentUnit(L1MuDTSectorProcessor& sp, int id );

    /// destructor
    virtual ~L1MuDTAssignmentUnit();

    /// run Assignment Unit
    virtual void run();
    
    /// reset Assignment Unit
    virtual void reset();
    
    /// assign phi
    void PhiAU();
    
    /// assign pt and charge
    void PtAU();
    
    /// assign quality
    void QuaAU();

    /// set precision of phi and phib 
    static void setPrecision();

  private:

    /// Track Segment Router
    void TSR();
    
    /// get track segment from a given station
    const L1MuDTTrackSegPhi* getTSphi(int station) const;
    
    /// convert sector Id to 8 bit code (= sector center)
    static int convertSector(int);
    
    /// determine charge
    static int getCharge(PtAssMethod);
    
    /// determine pt assignment method
    PtAssMethod getPtMethod() const;
    
    /// calculate bend angle
    int getPtAddress(PtAssMethod) const;
    
    /// build difference of two phi values
    int phiDiff(int stat1, int stat2) const;
    
    /// read phi-assignment look-up tables
    void readPhiLuts();
    
    /// read pt-assignment look-up tables
    void readPtaLuts();

    /// overload output stream operator for pt-assignment methods
    friend ostream& operator<<(ostream& s, PtAssMethod method);

  private:

    L1MuDTSectorProcessor& m_sp;
    int                    m_id;

    L1MuDTAddressArray               m_addArray;
    vector<const L1MuDTTrackSegPhi*> m_TSphi;
    PtAssMethod                      m_ptAssMethod;

    static L1MuDTPhiLut*       thePhiLUTs;  ///< phi-assignment look-up tables
    static L1MuDTPtaLut*       thePtaLUTs;  ///< pt-assignment look-up tables
    static unsigned short      nbit_phi;    ///< # of bits used for pt-assignment
    static unsigned short      nbit_phib;   ///< # of bits used for pt-assignment

};

#endif
