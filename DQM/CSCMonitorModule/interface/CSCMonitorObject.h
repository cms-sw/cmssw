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
#include "DQM/CSCMonitorModule/interface/CSCDQM_MonitorObjectIf.h"

class CSCMonitorObject : public cscdqm::MonitorObject, public MonitorElement {

  public: 

    CSCMonitorObject(const MonitorElement& me) : MonitorElement(me) { }
    virtual ~CSCMonitorObject() { }

      void Fill(float x) { MonitorElement::Fill(x); }
      void Fill(float x, float yw) { MonitorElement::Fill(x, yw); }
      void Fill(float x, float y, float zw) { MonitorElement::Fill(x, y, zw); }
      void Fill(float x, float y, float z, float w) { MonitorElement::Fill(x, y, z, w); }
      TH1 *getTH1(void) const { return MonitorElement::getTH1(); }
      void SetEntries(const double value) { MonitorElement::setEntries(value); }
      void SetBinContent(const int binX, const double value) { MonitorElement::setBinContent(binX, value);  }
      void SetBinContent(const int binX, const int binY, const double value) { MonitorElement::setBinContent(binX, binY, value); }
      double GetBinContent(const int binX) { return MonitorElement::getBinContent(binX); }
      double GetBinContent(const int binX, int binY) { return MonitorElement::getBinContent(binX, binY); }
      void SetAxisRange(const double from, const double to, const std::string axis) {   }
      void setAxisTitle(const std::string title, const int axisN) { MonitorElement::setAxisTitle(title, axisN);  } 

};

#endif
