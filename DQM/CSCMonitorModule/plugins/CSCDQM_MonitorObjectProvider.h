/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_MonitorObjectProviderIf.h
 *
 *    Description:  Histo Provider Interface
 *
 *        Version:  1.0
 *        Created:  10/03/2008 10:26:04 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius, valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_MonitorObjectProviderIf_H
#define CSCDQM_MonitorObjectProviderIf_H

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "CSCDQM_HistoDef.h"
#include "CSCDQM_MonitorObject.h"

namespace cscdqm {

  enum HistoType { INT, FLOAT, STRING, H1D, H2D, H3D, PROFILE, PROFILE2D };

  struct HistoBookRequest {

    const HistoDef *hdef;
    HistoType htype;
    std::string ctype;
    std::string folder;
    std::string title;

    int nchX;
    double lowX;
    double highX;

    int nchY;
    double lowY;
    double highY;

    int nchZ;
    double lowZ;
    double highZ;

    int default_int;
    float default_float;
    std::string default_string;

    std::string option;

    HistoBookRequest (const HistoDef& p_hdef, const HistoType& p_htype, const std::string& p_ctype,
                      const std::string& p_folder, const std::string& p_title,
                      const int p_nchX = 0, const double p_lowX = 0, const double p_highX = 0,
                      const int p_nchY = 0, const double p_lowY = 0, const double p_highY = 0,
                      const int p_nchZ = 0, const double p_lowZ = 0, const double p_highZ = 0,
                      const std::string& p_option = "s") {
      hdef = &p_hdef; 
      htype = p_htype;
      ctype = p_ctype;
      folder = p_folder; 
      title = p_title;
      nchX = p_nchX; 
      lowX = p_lowX;
      highX = p_highX; 
      nchY = p_nchY;
      lowY = p_lowY; 
      highY = p_highY;
      nchZ = p_nchZ; 
      lowZ = p_lowZ;
      highZ = p_highZ; 
      option = p_option;
    }

    HistoBookRequest (const HistoDef& p_hdef, const std::string& p_folder, const int p_value) {
      hdef = &p_hdef;
      htype = INT;
      ctype = "INT";
      folder = p_folder;
      title = p_hdef.getHistoName();
      default_int = p_value;
    }

    HistoBookRequest (const HistoDef& p_hdef, const std::string& p_folder, const float p_value) {
      hdef = &p_hdef;
      htype = FLOAT;
      ctype = "FLOAT";
      folder = p_folder;
      title = p_hdef.getHistoName();
      default_float = p_value;
    }

    HistoBookRequest (const HistoDef& p_hdef, const std::string& p_folder, 
                      const std::string& p_title, const std::string& p_value) {
      hdef = &p_hdef;
      htype = STRING;
      ctype = "STRING";
      folder = p_folder;
      title = p_title;
      default_string = p_value;
    }

  };

  /**
   * @class MonitorObjectProvider
   * @brief Interface for Histogram providing objects. Used by Event Processor
   * to retrieve MonitorObject 's and by Collection to book MonitorObject 's
   */
  class MonitorObjectProvider {

    public:
    
      virtual bool getCSCDetId(const unsigned int crateId, const unsigned int dmbId, CSCDetId& detId) const = 0;
      virtual MonitorObject *bookMonitorObject (const HistoBookRequest& p_req) = 0; 
  };

}

#endif
