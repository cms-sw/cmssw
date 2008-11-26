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

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_MonitorObjectIf.h"

/**
 * @class CSCMonitorObject
 * @brief cscdqm::MonitorObject implementation used in CSCMonitorModuleCmn
 */
class CSCMonitorObject : public cscdqm::MonitorObject {

  private:

    MonitorElement *me;

  public: 

    CSCMonitorObject(MonitorElement* p_me){ me = p_me; }
    virtual ~CSCMonitorObject() { }

    void Fill(float x) { 
      me->Fill(x); 
    }

    void Fill(float x, float yw) { 
      me->Fill(x, yw); 
    }

    void Fill(float x, float y, float zw) { 
      me->Fill(x, y, zw); 
    }

    void Fill(float x, float y, float z, float w) { 
      me->Fill(x, y, z, w); 
    }

    TH1 *getTH1(void) const { 
      return me->getTH1(); 
    }

    void SetEntries(const double value) { 
      me->setEntries(value); 
    }
    
    void SetBinContent(const int binX, const double value) { 
      me->setBinContent(binX, value);  
    }
    
    void SetBinContent(const int binX, const int binY, const double value) { 
      me->setBinContent(binX, binY, value); 
    }
    
    double GetBinContent(const int binX) { 
      return me->getBinContent(binX); 
    }
    
    double GetBinContent(const int binX, int binY) { 
      return me->getBinContent(binX, binY); 
    }
    
    void SetAxisRange(const double from, const double to, const std::string& axis) {
      me->getTH1()->SetAxisRange(from, to, axis.c_str());
    }
    
    void setAxisTitle(const std::string title, const int axisN) { 
      me->setAxisTitle(title, axisN);  
    } 

    TObject *getRefRootObject(void) const {
      return me->getRefRootObject();
    }

};

#endif
