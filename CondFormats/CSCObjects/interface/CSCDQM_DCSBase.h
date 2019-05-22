/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_DCSBase.h
 *
 *    Description:  CSCDQM DCS Base Objects
 *
 *        Version:  1.0
 *        Created:  05/04/2009 11:10:14 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_DCSBASE_H
#define CSCDQM_DCSBASE_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include <iostream>
#include <sstream>

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

namespace cscdqm {

  /** Type to store Unix timetamp */
  typedef long long TimeType;

  /** Enumeration of Board Types */
  enum DCSBoardType { ANY = 0, ALCT = 1, CFEB = 2, DMB = 3 };

  /** DCSBoardType utility object type */
  struct DCSBoardUtility {
    DCSBoardType boardType;
    DCSBoardUtility(const DCSBoardType boardType_) : boardType(boardType_) {}

    /**
     * @brief  Get DCSBoardType from string
     * @param  board Board name in string format
     * @return DCSBoardType for the string given
     */
    static DCSBoardType getDCSBoard(const std::string board) {
      if (board.compare("ALCT"))
        return ALCT;
      if (board.compare("CFEB"))
        return CFEB;
      if (board.compare("DMB"))
        return DMB;
      return ANY;
    }

    friend std::ostream& operator<<(std::ostream& out, const DCSBoardUtility& b) {
      switch (b.boardType) {
        case ANY:
          return out << "ANY";
        case ALCT:
          return out << "ALCT";
        case CFEB:
          return out << "CFEB";
        case DMB:
          return out << "DMB";
      }
      return out << "?";
    }
  };

  /**
   * DCS Address Type to store and manipulate DCS-related address
   */
  struct DCSAddressType {
    /** Endcap: 1 - plus, 2 - minus */
    unsigned short iendcap;

    /** Station number */
    unsigned short istation;

    /** Ring number */
    unsigned short iring;

    /** Chamber number */
    unsigned int ichamber;

    /** Get CSC Detector Id object from the address */
    CSCDetId getDetId() const { return CSCDetId(iendcap, istation, iring, ichamber); }

    /** Assignment operator */
    DCSAddressType& operator=(const DCSAddressType& a) {
      iendcap = a.iendcap;
      istation = a.istation;
      iring = a.iring;
      ichamber = a.ichamber;
      return *this;
    }

    /** Output stream operator */
    friend std::ostream& operator<<(std::ostream& out, const DCSAddressType& a) {
      std::ostringstream os;
      os << "endcap = " << a.iendcap << " ";
      os << "station = " << a.istation << " ";
      os << "ring = " << a.iring << " ";
      os << "chamber = " << a.ichamber;
      return out << os.str();
    }

    COND_SERIALIZABLE;
  };

}  // namespace cscdqm

#endif
