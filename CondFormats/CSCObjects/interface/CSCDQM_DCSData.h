/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_DCSData.h
 *
 *    Description:  CSCDQM DCS Objects
 *
 *        Version:  1.0
 *        Created:  05/04/2009 11:20:18 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_DCSDATA_H
#define CSCDQM_DCSDATA_H

#include <vector>

#include "CondFormats/CSCObjects/interface/CSCDQM_DCSBase.h"

namespace cscdqm {

  /**
   * Temperature measurement
   */
  struct TempMeasType {
        
    DCSAddressType adr;
    DCSBoardType   board;
    unsigned short boardId;
    int            value;

    friend std::ostream& operator<<(std::ostream& out, const TempMeasType& m) {
      std::ostringstream os;
      os << m.adr << " ";
      os << DCSBoardUtility(m.board);
      if (m.board == CFEB) {
        os << m.boardId;
      }
      os << " temp = " << m.value << "oC";
      return out << os.str();
    }

    TempMeasType& operator=(const TempMeasType& m) {
      adr = m.adr;
      board = m.board;
      boardId = m.boardId;
      value = m.value;
      return *this;
    }

  };

  /**
   * HV V measurement
   */
  struct HVVMeasType {
        
    DCSAddressType adr;
    unsigned int   position;
    int            value;

    friend std::ostream& operator<<(std::ostream& out, const HVVMeasType& m) {
      std::ostringstream os;
      os << m.adr << " ";
      os << "position:" << m.position;
      os << " V = " << m.value << "V";
      return out << os.str();
    }

    HVVMeasType& operator=(const HVVMeasType& m) {
      adr = m.adr;
      position = m.position;
      value = m.value;
      return *this;
    }

  };

  /**
   * LV V measurement
   */
  struct LVVMeasType {
        
    DCSAddressType adr;
    DCSBoardType   board;
    unsigned int   boardId;
    float          nominal_v;

    friend std::ostream& operator<<(std::ostream& out, const LVVMeasType& m) {
      std::ostringstream os;
      os << m.adr << " ";
      os << DCSBoardUtility(m.board);
      if (m.board == CFEB) {
        os << m.boardId;
      }
      os << " " << m.nominal_v << "V";
      return out << os.str();
    }

    LVVMeasType& operator=(const LVVMeasType& m) {
      adr = m.adr;
      board = m.board;
      boardId = m.boardId;
      nominal_v = m.nominal_v;
      return *this;
    }

  };

  /**
   * LV I measurement
   */
  struct LVIMeasType {
        
    DCSAddressType adr;
    DCSBoardType   board;
    unsigned int   boardId;
    float          nominal_v;
    float          value;

    friend std::ostream& operator<<(std::ostream& out, const LVIMeasType& m) {
      std::ostringstream os;
      os << m.adr << " ";
      os << DCSBoardUtility(m.board);
      if (m.board == CFEB) {
        os << m.boardId;
      }
      os << " " << m.nominal_v << "V";
      os << " " << m.value << "A";
      return out << os.str();
    }

    LVIMeasType& operator=(const LVIMeasType& m) {
      adr = m.adr;
      board = m.board;
      boardId = m.boardId;
      nominal_v = m.nominal_v;
      value = m.value;
      return *this;
    }

  };

  /**
   * @class DCSData
   * @brief CSC DCS Data type
   */
  class DCSData {

    public:

      DCSData();
      virtual ~DCSData();

      /** Temperature exceptions */
      std::vector<TempMeasType> temp_meas;

      /** HV V exceptions */
      std::vector<HVVMeasType>  hvv_meas;

      /** LV V exceptions */
      std::vector<LVVMeasType>  lvv_meas;

      /** LV I exceptions */
      std::vector<LVIMeasType>  lvi_meas;

      /** Temperature mode (most frequent value) */
      int  temp_mode;

      /** HV V mode (most frequent value) */
      int  hvv_mode;

      /** LV V mode (most frequent value) */
      bool lvv_mode;

      /** LV I mode (most frequent value) */
      float lvi_mode;

      /** This payload IOV value: timestamp since */
      TimeType iov;
  
      /** This payload last DCS data change value */
      TimeType last_change;

      DCSData& operator=(const DCSData& m) {
        iov = m.iov;
        last_change = m.last_change;
        temp_meas = m.temp_meas;
        hvv_meas = m.hvv_meas;
        lvv_meas = m.lvv_meas;
        lvi_meas = m.lvi_meas;
        temp_mode = m.temp_mode;
        hvv_mode = m.hvv_mode;
        lvv_mode = m.lvv_mode;
        lvi_mode = m.lvi_mode;
        return *this;
      }

  };

}

#endif

