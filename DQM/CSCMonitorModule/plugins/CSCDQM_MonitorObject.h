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
      virtual void SetEntries(double value) = 0;
      virtual const double GetEntries() = 0;
      virtual void SetBinContent(int binX, double value) = 0;
      virtual void SetBinContent(int binX, int binY, double value) = 0;
      virtual double GetBinContent(int binX) = 0;
      virtual double GetBinContent(int binX, int binY) = 0;
      virtual void SetAxisRange(double from, double to, const std::string& axis) = 0;
      virtual void setAxisTitle(std::string title, int axisN) = 0;
      virtual const TObject *getRefRootObject(void) const = 0;
      virtual const int GetMaximumBin() = 0;
      virtual void SetMaximum(double d) = 0;
      virtual void SetNormFactor(double factor = 1) = 0;
      virtual const double GetBinError(int bin) = 0;
      virtual void SetBinError(int bin, double error) = 0;

  };

}

#endif
