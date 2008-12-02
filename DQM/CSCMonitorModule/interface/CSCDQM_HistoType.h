/*
 * =====================================================================================
 *
 *       Filename:  HistoType.h
 *
 *    Description:  Histo Type Classes that are being used by EventProcessor
 *    to request histograms. 
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

  /** Type for histogram name constants */
  typedef char* HistoName; 

  /** Type for histogram name constants */
  typedef unsigned int HistoId; 

  namespace h {
    /** Histogram value that implies to skip the histogram */
    const HistoName HISTO_SKIP = "0";
  }

  /** DDU path pattern. Argument is DDU ID */
  static const char PATH_DDU[]     = "DDU_%d";
  /** CSC path pattern. Arguments are Create ID and DMB ID */
  static const char PATH_CSC[]     = "CSC_%03d_%02d";

  #include "DQM/CSCMonitorModule/interface/CSCDQM_HistoNames.h"

  /**
   * @class HistoType
   * @brief Abstract Base Histogram Type
   */
  class HistoType {

    private:

      /** The original, raw name of histogram */
      HistoName hname;

    public:

      /**
       * @brief  Base constructor
       * @param  p_hname Raw histogram name by HistoName
       * @return 
       */
      HistoType(const HistoName& p_hname) { hname  = p_hname; }
      
      /**
       * @brief  Get raw histogram name
       * @return Raw histogram name
       */
      const HistoName&  getHistoName() const { return hname; }

      /**
       * @brief  Get processed histogram name. It can include additional
       * parameter in formated name. This Name is being constructed from raw
       * name and additional parameter. 
       * @return processed full name of the histogram
       */
      virtual const std::string getName() const { return hname; }

      /**
       * @brief  Get full path of the histogram. It is being constructed by
       * appending path and histogam name. 
       * @return full path name of the histogram (processed)
       */
      const std::string getFullPath() const {
        std::string uid(getPath());
        uid.append("/");
        uid.append(getName());
        return uid;
      }

      /**
       * @brief  Comparison (==) operator
       * @param  t Histogram to be compared to
       * @return true if HistoTypes match, false - otherwise
       */
      const bool operator== (const HistoType& t) const {
        if (strcmp(getHistoName(), t.getHistoName()) != 0)  return false;
        if (getDDUId() != t.getDDUId()) return false;
        if (getCrateId() != t.getCrateId()) return false;
        if (getDMBId() != t.getDMBId()) return false;
        if (getAddId() != t.getAddId()) return false;
        return true;
      }

      /**
       * @brief  Assignment (=) operator
       * @param  t Histogram to be taken data from
       * @return resulting histogram (this)
       */
      const HistoType& operator= (const HistoType& t) {
        hname  = t.getHistoName();
        return *this;
      }

      /**
       * @brief  Less (<) operator
       * @param  t Histogram to be compared to
       * @return true if t is "more" than this
       */
      const bool operator< (const HistoType& t) const {
        if (strcmp(getHistoName(), t.getHistoName()) < 0) return true; 
        if (getDDUId() < t.getDDUId())  return true;
        if (getCrateId() < t.getCrateId())  return true;
        if (getDMBId() < t.getDMBId())  return true;
        if (getAddId() < t.getAddId())  return true;
        return false;
      }

      /**
       * @brief  Printing (<<) operator that prints hisotgram full path
       * @param  out output stream
       * @param  t Histogram type to be printed
       * @return output stream
       */
      friend std::ostream& operator<<(std::ostream& out, const HistoType& t) {
        return out << t.getFullPath();
      }

      /**
       * @brief  Get path part of the histogram (used only for DDUs and CSCs) 
       * @return path part of the histogram
       */
      virtual const std::string getPath() const     { return ""; }

      /**
       * @brief  Get CSC Crate ID 
       * @return CSC Crate ID
       */
      virtual const unsigned int getCrateId() const { return  0; }

      /**
       * @brief  Get CSC DMB ID
       * @return CSC DMB ID
       */
      virtual const unsigned int getDMBId() const   { return  0; }

      /**
       * @brief  Get CSC Additional ID (used to store Layer, CLCT, ALCT and
       * other identifiers.
       * @return CSC Additional ID
       */
      virtual const unsigned int getAddId() const   { return  0; }

      /**
       * @brief  Get DDU ID
       * @return DDU ID
       */
      virtual const unsigned int getDDUId() const   { return  0; }

  };

  /**
   * @class EMUHistoType
   * @brief EMU Level Histogram Type
   */
  class EMUHistoType : public HistoType {

    public:

      /**
       * @brief  Constructor. It calls Base constructor inline
       * @param  p_hname Histogram name (passed to Base class)
       * @return 
       */
      EMUHistoType(const HistoName& p_hname) : HistoType(p_hname) { }

  };

  /**
   * @class DDUHistoType
   * @brief DDU Level Histogram Type
   */
  class DDUHistoType : public HistoType {

    private:

      unsigned int dduId;

    public:

      /**
       * @brief  Constructor. It calls Base constructor inline
       * @param  p_hname Histogram name (passed to Base class)
       * @param  p_dduId DDU ID
       * @return 
       */
      DDUHistoType(const HistoName& p_id, const unsigned int p_dduId) : HistoType(p_id) { dduId = p_dduId; }
      const unsigned int getDDUId() const { return dduId; }
      const std::string getPath() const { return getPath(dduId); }

      /**
       * @brief  Static DDU path formatter
       * @param  p_dduId DDU ID
       * @return formatted DDU path 
       */
      static const std::string getPath(const unsigned int p_dduId) { 
        return Form(PATH_DDU, p_dduId); 
      }

      /**
       * @brief  Assignment (=) operator. Calls base assignment operator and
       * assigns DDU-related data
       * @param  t Histogram to be taken data from
       * @return resulting histogram (this)
       */
      const DDUHistoType& operator= (const DDUHistoType& t) {
        HistoType *h1 = const_cast<DDUHistoType*>(this);
        const HistoType *h2 = &t;
        *h1 = *h2;
        dduId   = t.getDDUId();
        return *this;
      }

  };

  /**
   * @class CSCHistoType
   * @brief CSC Level Histogram Type
   */
  class CSCHistoType : public HistoType {

    private:

      /** CSC Crate ID */
      unsigned int crateId;
      /** CSC DMB ID */
      unsigned int dmbId;
      /** CSC Additional ID */
      unsigned int addId;

    public:

      /**
       * @brief  Constructor. It calls Base constructor inline
       * @param  p_hname Histogram name (passed to Base class)
       * @param  p_crateId CSC Crate ID
       * @param  p_dmbId CSC DMB ID
       * @param  p_addId CSC Additional ID, used to store Layer ID, CFEB ID,
       * etc. Used to store processed name identifier. Optional.
       * @return 
       */
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

      /**
       * @brief  Static CSC path formatter
       * @param  p_crateId CSC Crate ID
       * @param  p_dmbId CSC DMB ID
       * @return formatted CSC path 
       */
      static const std::string getPath(const unsigned int p_crateId, const unsigned int p_dmbId) { 
        return Form(PATH_CSC, p_crateId, p_dmbId); 
      }

      /**
       * @brief  Assignment (=) operator. Calls base assignment operator and
       * assigns CSC-related data
       * @param  t Histogram to be taken data from
       * @return resulting histogram (this)
       */
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

      /**
       * @brief  Constructor. It calls Base constructor inline
       * @param  p_hname Histogram name (passed to Base class)
       * @return 
       */
      ParHistoType(const HistoName& p_id) : HistoType(p_id) { }

  };

}

#endif

