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

#include "DQM/CSCMonitorModule/interface/CSCDQM_Utility.h"

namespace cscdqm {

  typedef char* HistoName; 
  namespace h {
    const HistoName HISTO_SKIP = "0";
  }

  static const char PATH_DDU[]     = "DDU_%d";
  static const char PATH_CSC[]     = "CSC_%03d_%02d";

  #include "DQM/CSCMonitorModule/interface/CSCDQM_HistoNames.h"

  /**
   * @class HistoType
   * @brief Abstract Base Histogram Type
   */
  class HistoType {

    private:

      HistoName hname;

    public:

      HistoType(const HistoName& p_hname) { hname  = p_hname; }
      
      const HistoName&  getHistoName() const { return hname; }
      virtual const std::string getName() const { return hname; }

      const std::string getFullPath() const {
        std::string uid(getPath());
        uid.append("/");
        uid.append(getName());
        return uid;
      }

      const bool operator== (const HistoType& t) const {
        if (getFullPath().compare(t.getFullPath()) == 0)  return true;
        return false;
      }

      const HistoType& operator= (const HistoType& t) {
        hname  = t.getHistoName();
        return *this;
      }

      const bool operator< (const HistoType& t) const {
        return (getFullPath() < t.getFullPath());
      }

      friend std::ostream& operator<<(std::ostream& out, const HistoType& t) {
        return out << t.getFullPath();
      }

      virtual const std::string getPath() const     { return ""; }
      virtual const unsigned int getCrateId() const { return  0; }
      virtual const unsigned int getDMBId() const   { return  0; }
      virtual const unsigned int getAddId() const   { return  0; }
      virtual const unsigned int getDDUId() const   { return  0; }

  };

  /**
   * @class EMUHistoType
   * @brief EMU Level Histogram Type
   */
  class EMUHistoType : public HistoType {

    public:

      EMUHistoType(const HistoName& p_id) : HistoType(p_id) { }

  };

  /**
   * @class DDUHistoType
   * @brief DDU Level Histogram Type
   */
  class DDUHistoType : public HistoType {

    private:

      unsigned int dduId;

    public:

      DDUHistoType(const HistoName& p_id, const unsigned int p_dduId) : HistoType(p_id) { dduId = p_dduId; }
      const unsigned int getDDUId() const { return dduId; }
      const std::string getPath() const { return getPath(dduId); }

      static const std::string getPath(const unsigned int p_dduId) { 
        return Form(PATH_DDU, p_dduId); 
      }

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

      CSCHistoType(const HistoName& p_id, const unsigned int p_crateId, const unsigned int p_dmbId, const unsigned int p_addId = 0) : 
        HistoType(p_id) {
        crateId = p_crateId;
        dmbId = p_dmbId;
        addId = p_addId;
      }

      const unsigned int getCrateId() const { return crateId; }
      const unsigned int getDMBId() const { return dmbId; }
      const unsigned int getAddId() const { return addId; }
      const std::string  getName() const { return Utility::getNameById(getHistoName(), getAddId()); }
      const std::string  getPath() const { return getPath(crateId, dmbId); }

      static const std::string getPath(const unsigned int p_crateId, const unsigned int p_dmbId) { 
        return Form(PATH_CSC, p_crateId, p_dmbId); 
      }

      const CSCHistoType& operator= (const CSCHistoType& t) {
        HistoType *h1 = const_cast<CSCHistoType*>(this);
        const HistoType *h2 = &t;
        *h1 = *h2;
        crateId = t.getCrateId();
        dmbId   = t.getDMBId();
        addId   = t.getAddId();
        return *this;
      }


  };

  /**
   * @class ParHistoType
   * @brief Parameter Histogram Type
   */
  class ParHistoType : public HistoType {

    public:

      ParHistoType(const HistoName& p_id) : HistoType(p_id) { }

  };

}

#endif

