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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/L1TObjects/interface/L1MuDTAssParam.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTAddressArray.h"
class L1MuDTPhiLut;
class L1MuDTPtaLut;
class L1MuDTPhiLutRcd;
class L1MuDTPtaLutRcd;
class L1MuDTTrackSegPhi;
class L1MuDTSectorProcessor;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTAssignmentUnit {
public:
  /// constructor
  L1MuDTAssignmentUnit(L1MuDTSectorProcessor& sp, int id, edm::ConsumesCollector);

  /// destructor
  ~L1MuDTAssignmentUnit();

  /// run Assignment Unit
  void run(const edm::EventSetup& c);

  /// reset Assignment Unit
  void reset();

  /// assign phi
  void PhiAU(const edm::EventSetup& c);

  /// assign pt and charge
  void PtAU(const edm::EventSetup& c);

  /// assign quality
  void QuaAU();

  /// set precision of phi and phib
  void setPrecision();

private:
  /// Track Segment Router
  void TSR();

  /// get track segment from a given station
  const L1MuDTTrackSegPhi* getTSphi(int station) const;

  /// convert sector Id to 8 bit code (= sector center)
  int convertSector(int);

  /// determine charge
  int getCharge(PtAssMethod);

  /// determine pt assignment method
  PtAssMethod getPtMethod() const;

  /// calculate bend angle
  int getPtAddress(PtAssMethod, int bendcharge = 0) const;

  /// build difference of two phi values
  int phiDiff(int stat1, int stat2) const;

private:
  L1MuDTSectorProcessor& m_sp;
  int m_id;

  L1MuDTAddressArray m_addArray;
  std::vector<const L1MuDTTrackSegPhi*> m_TSphi;
  PtAssMethod m_ptAssMethod;

  edm::ESGetToken<L1MuDTPhiLut, L1MuDTPhiLutRcd> thePhiToken;
  edm::ESGetToken<L1MuDTPtaLut, L1MuDTPtaLutRcd> thePtaToken;
  edm::ESHandle<L1MuDTPhiLut> thePhiLUTs;  ///< phi-assignment look-up tables
  edm::ESHandle<L1MuDTPtaLut> thePtaLUTs;  ///< pt-assignment look-up tables
  unsigned short nbit_phi;                 ///< # of bits used for pt-assignment
  unsigned short nbit_phib;                ///< # of bits used for pt-assignment
};

#endif
