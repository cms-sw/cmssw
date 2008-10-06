/*
 * =====================================================================================
 *
 *       Filename:  MonitorObject.h
 *
 *    Description:  Monitor Object envelope
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

#ifndef CSCMonitorObject_H
#define CSCMonitorObject_H

#include <TH1.h>

#include "DQMServices/Core/interface/MonitorElement.h"

class CSCMonitorObject : public MonitorElement {

  public: 

    void SetEntries(const double value) { }
    void SetBinContent(const int binX, const double value) { }
    void SetBinContent(const int binX, const int binY, const double value) { }
    double GetBinContent(const int binX) { return 0.0; }
    double GetBinContent(const int binX, int) { return 0.0; }
    void SetAxisRange(const double from, const double to, const std::string axis) {  }
    TH1* getObject() { return NULL; }

};

#endif
