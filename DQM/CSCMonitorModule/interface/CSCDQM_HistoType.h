/*
 * =====================================================================================
 *
 *       Filename:  HistoType.h
 *
 *    Description:  Histo Type Constants
 *
 *        Version:  1.0
 *        Created:  10/03/2008 11:54:31 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_HistoType_H
#define CSCDQM_HistoType_H 

#include <string>
#include <iostream>  

namespace cscdqm {

  typedef char* HistoName; 
  namespace h {
    const HistoName HISTO_SKIP = "0";
  }

  static const char TAG_EMU[] = "EMU";
  static const char TAG_DDU[] = "DDU_%d";
  static const char TAG_CSC[] = "CSC_%03d_%02d";
  static const char TAG_PAR[] = "PARAMETER";

  #include "DQM/CSCMonitorModule/interface/CSCDQM_HistoNames.h"

  /**
   * @class HistoType
   * @brief Abstract Base Histogram Type
   */
  class HistoType {

    private:

      HistoName   id;

    public:

      HistoType(const HistoName& p_id) {
        id  = p_id;
      }
      
      const HistoName&  getId() const { return id;  }

      const std::string getUID() const {
        std::string uid(getTag());
        uid.append("/");
        uid.append(id);
        return uid;
      }

      const bool operator== (const HistoType& t) const {
        if (getUID().compare(t.getUID()) == 0)  return true;
        return false;
      }

      const HistoType& operator= (const HistoType& t) {
        id  = t.getId();
        return *this;
      }

      const bool operator< (const HistoType& t) const {
        return (getUID() < t.getUID());
      }

      friend std::ostream& operator<<(std::ostream& out, const HistoType& t) {
        return out << t.getUID();
      }

      virtual const std::string getTag() const { return ""; }

  };

  /**
   * @class EMUHistoType
   * @brief EMU Level Histogram Type
   */
  class EMUHistoType : public HistoType {

    public:

      EMUHistoType(const HistoName& p_id) : HistoType(p_id) { }
      const std::string getTag() const { return TAG_EMU; }

  };

  /**
   * @class DDUHistoType
   * @brief DDU Level Histogram Type
   */
  class DDUHistoType : public HistoType {

    private:

      unsigned int dduId;

    public:

      DDUHistoType(const HistoName& p_id, const unsigned int p_dduId) : HistoType(p_id) {
        dduId = p_dduId;
      }

      const unsigned int getDDUId() const { return dduId; }

      const std::string getTag() const { return Form(TAG_DDU, dduId); }

  };

  /**
   * @class CSCHistoType
   * @brief CSC Level Histogram Type
   */
  class CSCHistoType : public HistoType {

    private:

      unsigned int crateId;
      unsigned int dmbId;
      unsigned int addId;

    public:

      CSCHistoType(const HistoName& p_id, const unsigned int p_crateId, const unsigned int p_dmbId, const unsigned int p_addId) : 
        HistoType(p_id) {
        crateId = p_crateId;
        dmbId = p_dmbId;
        addId = p_addId;
      }

      const unsigned int getCrateId() const { return crateId; }
      const unsigned int getDMBId() const { return dmbId; }
      const unsigned int getAddId() const { return addId; }

      const std::string getTag() const { return Form(TAG_CSC, crateId, dmbId); }

  };

  /**
   * @class ParHistoType
   * @brief Parameter Histogram Type
   */
  class ParHistoType : public HistoType {

    public:

      ParHistoType(const HistoName& p_id) : HistoType(p_id) { }
      const std::string getTag() const { return TAG_PAR; }

  };

}

#endif

