/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_MonitorObject.h
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

#ifndef CSCDQM_MonitorObject_H
#define CSCDQM_MonitorObject_H

#include <TH1.h>

#include "CSCDQM_Logger.h"
#include "CSCDQM_Lock.h"

namespace cscdqm {

  /**
   * @class MonitorObject
   * @brief Monitoring Object interface used to cover Root object and provide
   * common interface to EventProcessor and Collection objects. Used by
   * HistoProvider interface.
   */
  class MonitorObject : public Lock {
  
    public: 

      virtual void Fill(float x) = 0;
      virtual void Fill(float x, float yw) = 0;
      virtual void Fill(float x, float y, float zw) = 0;
      virtual void Fill(float x, float y, float z, float w) = 0;
      virtual const TH1 *getTH1(void) const = 0;
      virtual TH1 *getTH1Lock(void) = 0; 
      virtual void SetEntries(const double value) = 0;
      virtual const double GetEntries() = 0;
      virtual void SetBinContent(const int binX, const double value) = 0;
      virtual void SetBinContent(const int binX, const int binY, const double value) = 0;
      virtual double GetBinContent(const int binX) = 0;
      virtual double GetBinContent(const int binX, int binY) = 0;
      virtual void SetAxisRange(const double from, const double to, const std::string& axis) = 0;
      virtual void setAxisTitle(const std::string title, const int axisN) = 0;
      virtual const TObject *getRefRootObject(void) const = 0;
      virtual const int GetMaximumBin() = 0;
      virtual void SetMaximum(const double d) = 0;
      virtual void SetNormFactor(const double factor = 1) = 0;
      virtual const double GetBinError(const int bin) = 0;
      virtual void SetBinError(const int bin, const double error) = 0;

  };

}

#endif
