/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_MonitorObjectIf.h
 *
 *    Description:  Monitor Object interface
 *
 *        Version:  1.0
 *        Created:  10/06/2008 01:49:51 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius, valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_MonitorObjectIf_H
#define CSCDQM_MonitorObjectIf_H

#include <boost/shared_ptr.hpp>
#include <TH1.h>

namespace cscdqm {

  /**
   * @class MonitorObject
   * @brief Monitoring Object interface used to cover Root object and provide
   * common interface to EventProcessor and Collection objects. Used by
   * HistoProvider interface.
   */
  class MonitorObject {
  
    public: 

      virtual void Fill(float x) = 0;
      virtual void Fill(float x, float yw) = 0;
      virtual void Fill(float x, float y, float zw) = 0;
      virtual void Fill(float x, float y, float z, float w) = 0;
      virtual TH1 *getTH1(void) const = 0;
      virtual void SetEntries(const double value) = 0;
      virtual void SetBinContent(const int binX, const double value) = 0;
      virtual void SetBinContent(const int binX, const int binY, const double value) = 0;
      virtual double GetBinContent(const int binX) = 0;
      virtual double GetBinContent(const int binX, int binY) = 0;
      virtual void SetAxisRange(const double from, const double to, const std::string& axis) = 0;
      virtual void setAxisTitle(const std::string title, const int axisN) = 0;
      virtual TObject *getRefRootObject(void) const = 0;

  };

}

#endif
