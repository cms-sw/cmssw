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
 *   $Date: 2008/02/18 17:38:04 $
 *   $Revision: 1.4 $
 *
 *   N. Neumeister            CERN EP
 *   J. Troconiz              UAM Madrid
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

#include "L1Trigger/DTTrackFinder/interface/L1AbstractProcessor.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include <FWCore/Framework/interface/ESHandle.h>
#include "CondFormats/L1TObjects/interface/L1MuDTAssParam.h"
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

    /// constructor
    L1MuDTAssignmentUnit(L1MuDTSectorProcessor& sp, int id );

    /// destructor
    virtual ~L1MuDTAssignmentUnit();

    /// run Assignment Unit
    virtual void run(const edm::EventSetup& c);
    
    /// reset Assignment Unit
    virtual void reset();
    
    /// assign phi
    void PhiAU(const edm::EventSetup& c);
    
    /// assign pt and charge
    void PtAU(const edm::EventSetup& c);
    
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
    int getPtAddress(PtAssMethod, int bendcharge=0) const;
    
    /// build difference of two phi values
    int phiDiff(int stat1, int stat2) const;
    
  private:

    L1MuDTSectorProcessor& m_sp;
    int                    m_id;

    L1MuDTAddressArray                    m_addArray;
    std::vector<const L1MuDTTrackSegPhi*> m_TSphi;
    PtAssMethod                           m_ptAssMethod;

    edm::ESHandle< L1MuDTPhiLut > thePhiLUTs;  ///< phi-assignment look-up tables
    edm::ESHandle< L1MuDTPtaLut > thePtaLUTs;  ///< pt-assignment look-up tables
    static unsigned short      nbit_phi;       ///< # of bits used for pt-assignment
    static unsigned short      nbit_phib;      ///< # of bits used for pt-assignment

};

#endif
