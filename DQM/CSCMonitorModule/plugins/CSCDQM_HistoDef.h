/*
 * =====================================================================================
 *
 *       Filename:  HistoDef.h
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

#ifndef CSCDQM_HistoDef_H
#define CSCDQM_HistoDef_H 

#include <string>
#include <iostream>  

#include "CSCDQM_Utility.h"
#include "CSCDQM_Logger.h"

namespace cscdqm {

  /** Type for histogram name constants */
  typedef std::string HistoName; 

  /** Type for histogram id constants */
  typedef unsigned int HistoId; 

  /** Type for detector component (HW) id parameters */
  typedef unsigned int HwId; 

  namespace h {
    /** Histogram value that implies to skip the histogram */
    const HistoName HISTO_SKIP = "0";
  }
  
  /** DDU path pattern. Argument is FED ID */
  static const char PATH_FED[]     = "FED_%03d";

  /** DDU path pattern. Argument is DDU ID */
  static const char PATH_DDU[]     = "DDU_%02d";

  /** CSC path pattern. Arguments are Create ID and DMB ID */
  static const char PATH_CSC[]     = "CSC_%03d_%02d";

  static const TPRegexp REGEXP_ONDEMAND("^.*%d.*$");

  #include "CSCDQM_HistoNames.h"

  /**
   * @class HistoDef
   * @brief Abstract Base Histogram Definition
   */
  class HistoDef {

    private:

      /** Histogram Id */
      HistoId id;

    public:

      /**
       * @brief  Base constructor
       * @param  p_hname Raw histogram name by HistoName
       * @return 
       */
      HistoDef(const HistoId p_id) : id(p_id) { }
      
      /**
       * @brief  Base virtual destructor
       */ 
      virtual ~HistoDef() { }

      /**
       * @brief  Get Histogram ID
       * @return Histogram ID
       */
      const HistoId getId() const { return id; }

      /**
       * @brief  Get raw histogram name
       * @return Raw histogram name
       */
      const HistoName&  getHistoName() const { return h::names[id]; }

      /**
       * @brief  Get processed histogram name. It can include additional
       * parameter in formated name. This Name is being constructed from raw
       * name and additional parameter. 
       * @return processed full name of the histogram
       */
      virtual const std::string getName() const { return getHistoName(); }

      /**
       * @brief  Get full path of the histogram. It is being constructed by
       * appending path and histogam name. 
       * @return full path name of the histogram (processed)
       */
      const std::string getFullPath() const {
        std::string path(getPath());
        if (path.size() > 0) path.append("/");
        path.append(getName());
        return path;
      }

      /**
       * @brief  Comparison (==) operator
       * @param  t Histogram to be compared to
       * @return true if HistoDefs match, false - otherwise
       */
      const bool operator== (const HistoDef& t) const {
        if (getId()      != t.getId())      return false;
        if (getFEDId()   != t.getFEDId())   return false;
        if (getDDUId()   != t.getDDUId())   return false;
        if (getCrateId() != t.getCrateId()) return false;
        if (getDMBId()   != t.getDMBId())   return false;
        if (getAddId()   != t.getAddId())   return false;
        return true;
      }

      /**
       * @brief  Assignment (=) operator
       * @param  t Histogram to be taken data from
       * @return resulting histogram (this)
       */
      const HistoDef& operator= (const HistoDef& t) {
        id  = t.getId();
        return *this;
      }

      /**
       * @brief  Less (<) operator
       * @param  t Histogram to be compared to
       * @return true if t is "more" than this
       */
      const bool operator< (const HistoDef& t) const {
        if (getId()      < t.getId())       return true; 
        if (getFEDId()   < t.getFEDId())    return true;
        if (getDDUId()   < t.getDDUId())    return true;
        if (getCrateId() < t.getCrateId())  return true;
        if (getDMBId()   < t.getDMBId())    return true;
        if (getAddId()   < t.getAddId())    return true;
        return false;
      }

      /**
       * @brief  Printing (<<) operator that prints hisotgram full path
       * @param  out output stream
       * @param  t Histogram type to be printed
       * @return output stream
       */
      friend std::ostream& operator<<(std::ostream& out, const HistoDef& t) {
        return out << t.getFullPath();
      }

      /**
       * @brief  Get path part of the histogram (used only for DDUs and CSCs) 
       * @return path part of the histogram
       */
      virtual const std::string getPath() const { return ""; }

      /**
       * @brief  Get CSC Crate ID 
       * @return CSC Crate ID
       */
      virtual const HwId getCrateId() const { return  0; }

      /**
       * @brief  Get CSC DMB ID
       * @return CSC DMB ID
       */
      virtual const HwId getDMBId() const { return  0; }

      /**
       * @brief  Get CSC Additional ID (used to store Layer, CLCT, ALCT and
       * other identifiers.
       * @return CSC Additional ID
       */
      virtual const HwId getAddId() const { return  0; }

      /**
       * @brief  Get FED ID
       * @return FED ID
       */
      virtual const HwId getFEDId() const { return  0; }

      /**
       * @brief  Get DDU ID
       * @return DDU ID
       */
      virtual const HwId getDDUId() const { return  0; }

      /**
       * @brief  Process Title by Adding appropriate ID
       * @param  p_title Title to process
       * @return processed title
       */
      virtual const std::string processTitle(const std::string& p_title) const {
        return p_title;
      }

      /**
       * @brief  Get Histogram ID by name
       * @param  p_name Histogram name
       * @param  p_id Id to be filled in (return value)
       * @return true if ID was found, false - otherwise
       */
      static const bool getHistoIdByName(const std::string& p_name, HistoId& p_id) {
        for (HistoId i = 0; i < h::namesSize; i++) {
          if (p_name.compare(h::names[i]) == 0) {
            p_id = i;
            return true;
          }
        }
        return false;
      }

      /**
       * @brief  Get Histogram key name by id
       * @param  p_id Histogram id
       * @return Histogram key name
       */
      static const std::string getHistoKeyById(const HistoId& p_id) {
        return h::keys[p_id];
      }

      /**
       * @brief  Process name by applying ID to %d pattern (pattern is stored in REGEXP_ONDEMAND)
       * @param  p_name String value to process
       * @param  p_id ID to include
       * @return processed value
       */
      static const std::string processName(const HistoName& p_name, const HwId p_id) {
        if (Utility::regexMatch(REGEXP_ONDEMAND, p_name)) {
          return Form(p_name.c_str(), p_id);
        }
        return p_name;
      }

  };

  /**
   * @class EMUHistoDef
   * @brief EMU Level Histogram Definition
   */
  class EMUHistoDef : public HistoDef {

    public:

      /**
       * @brief  Constructor. It calls Base constructor inline
       * @param  p_id Histogram id (to be passed to Base class)
       * @return 
       */
      EMUHistoDef(const HistoId p_id) : HistoDef(p_id) { }

  };

 /**
   * @class FEDHistoDef
   * @brief FED Level Histogram Definition
   */
  class FEDHistoDef : public HistoDef {
      
    private:
      
      HwId fedId;

    public:

      /**
       * @brief  Constructor. It calls Base constructor inline
       * @param  p_id Histogram ID (to be passed to Base class)
       * @param  p_fedId FED ID
       * @return 
       */
      FEDHistoDef(const HistoId p_id, const HwId p_fedId) : HistoDef(p_id), fedId(p_fedId) { }
      const HwId getFEDId() const { return fedId; }
      const std::string getPath() const { return getPath(fedId); }

      /**
       * @brief  Static FED path formatter
       * @param  p_fedId FED ID
       * @return formatted FED path 
       */   
      static const std::string getPath(const HwId p_fedId) {
        return Form(PATH_FED, p_fedId);
      } 
      
      /**
       * @brief  Assignment (=) operator. Calls base assignment operator and
       * assigns FEd-related data
       * @param  t Histogram to be taken data from
       * @return resulting histogram (this)
       */
      const FEDHistoDef& operator= (const FEDHistoDef& t) {
        HistoDef *h1 = const_cast<FEDHistoDef*>(this);
        const HistoDef *h2 = &t;
        *h1 = *h2;
        fedId   = t.getFEDId();
        return *this;
      }

      const std::string processTitle(const std::string& p_title) const {
        return processName(p_title.c_str(), getFEDId());
      }
        
  };      

  /**
   * @class DDUHistoDef
   * @brief DDU Level Histogram Definition
   */
  class DDUHistoDef : public HistoDef {

    private:

      HwId dduId;

    public:

      /**
       * @brief  Constructor. It calls Base constructor inline
       * @param  p_id Histogram ID (to be passed to Base class)
       * @param  p_dduId DDU ID
       * @return 
       */
      DDUHistoDef(const HistoId p_id, const HwId p_dduId) : HistoDef(p_id), dduId(p_dduId) { }
      const HwId getDDUId() const { return dduId; }
      const std::string getPath() const { return getPath(dduId); }

      /**
       * @brief  Static DDU path formatter
       * @param  p_dduId DDU ID
       * @return formatted DDU path 
       */
      static const std::string getPath(const HwId p_dduId) { 
        return Form(PATH_DDU, p_dduId); 
      }

      /**
       * @brief  Assignment (=) operator. Calls base assignment operator and
       * assigns DDU-related data
       * @param  t Histogram to be taken data from
       * @return resulting histogram (this)
       */
      const DDUHistoDef& operator= (const DDUHistoDef& t) {
        HistoDef *h1 = const_cast<DDUHistoDef*>(this);
        const HistoDef *h2 = &t;
        *h1 = *h2;
        dduId   = t.getDDUId();
        return *this;
      }

      const std::string processTitle(const std::string& p_title) const {
        return processName(p_title.c_str(), getDDUId());
      }

  };

  /**
   * @class CSCHistoDef
   * @brief CSC Level Histogram Type
   */
  class CSCHistoDef : public HistoDef {

    private:

      /** CSC Crate ID */
      HwId crateId;
      /** CSC DMB ID */
      HwId dmbId;
      /** CSC Additional ID */
      HwId addId;

    public:

      /**
       * @brief  Constructor. It calls Base constructor inline
       * @param  p_hname Histogram id (to be passed to Base class)
       * @param  p_crateId CSC Crate ID
       * @param  p_dmbId CSC DMB ID
       * @param  p_addId CSC Additional ID, used to store Layer ID, CFEB ID,
       * etc. Used to store processed name identifier. Optional.
       * @return 
       */
      CSCHistoDef(const HistoId p_id, const HwId p_crateId, const HwId p_dmbId, const HwId p_addId = 0) : 
        HistoDef(p_id), crateId(p_crateId), dmbId(p_dmbId), addId(p_addId) { }

      const HwId getCrateId() const { return crateId; }
      const HwId getDMBId()   const { return dmbId; }
      const HwId getAddId()   const { return addId; }
      const std::string getName() const { return processName(getHistoName(), getAddId()); }
      const std::string getPath() const { return getPath(crateId, dmbId); }

      /**
       * @brief  Static CSC path formatter
       * @param  p_crateId CSC Crate ID
       * @param  p_dmbId CSC DMB ID
       * @return formatted CSC path 
       */
      static const std::string getPath(const HwId p_crateId, const HwId p_dmbId) { 
        return Form(PATH_CSC, p_crateId, p_dmbId); 
      }

      /**
       * @brief  Assignment (=) operator. Calls base assignment operator and
       * assigns CSC-related data
       * @param  t Histogram to be taken data from
       * @return resulting histogram (this)
       */
      const CSCHistoDef& operator= (const CSCHistoDef& t) {
        HistoDef *h1 = const_cast<CSCHistoDef*>(this);
        const HistoDef *h2 = &t;
        *h1 = *h2;
        crateId = t.getCrateId();
        dmbId   = t.getDMBId();
        addId   = t.getAddId();
        return *this;
      }

      const std::string processTitle(const std::string& p_title) const {
        return processName(p_title.c_str(), getAddId());
      }


  };

  /**
   * @class ParHistoDef
   * @brief Parameter Histogram Definition
   */
  class ParHistoDef : public HistoDef {

    private:

      /**
       * @brief  Parameter name
       */ 
      HistoName name;


    public:

      /**
       * @brief  Constructor. It calls Base constructor inline
       * @param  p_name Histogram name, id will be constructed by using fastHash algorithm and then to be passed to Base class
       * @return 
       */
      ParHistoDef(const HistoName& p_name) : HistoDef(Utility::fastHash(p_name.c_str())), name(p_name) { }

      /**
       * @brief  Constructor. It calls Base constructor inline
       * @param  p_id Histogram id (to be passed to Base class)
       * @return 
       */
      ParHistoDef(const HistoId p_id) : HistoDef(p_id) {
        name = HistoDef::getHistoName();
      }

      const HistoName&  getHistoName() const { return name; }

  };

  static const std::type_info& EMUHistoDefT = typeid(cscdqm::EMUHistoDef);
  static const std::type_info& FEDHistoDefT = typeid(cscdqm::FEDHistoDef);
  static const std::type_info& DDUHistoDefT = typeid(cscdqm::DDUHistoDef);
  static const std::type_info& CSCHistoDefT = typeid(cscdqm::CSCHistoDef);
  static const std::type_info& ParHistoDefT = typeid(cscdqm::ParHistoDef);

}

#endif

