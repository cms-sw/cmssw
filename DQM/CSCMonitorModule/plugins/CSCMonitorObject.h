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
#include "CSCDQM_MonitorObject.h"

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
      cscdqm::LockType lock(mutex);
      me->Fill(x); 
    }

    void Fill(float x, float yw) { 
      cscdqm::LockType lock(mutex);
      me->Fill(x, yw); 
    }

    void Fill(float x, float y, float zw) { 
      cscdqm::LockType lock(mutex);
      me->Fill(x, y, zw); 
    }

    void Fill(float x, float y, float z, float w) { 
      cscdqm::LockType lock(mutex);
      me->Fill(x, y, z, w); 
    }

    void SetEntries(const double value) { 
      cscdqm::LockType lock(mutex);
      me->setEntries(value); 
    }
    
    void SetBinContent(const int binX, const double value) { 
      cscdqm::LockType lock(mutex);
      me->setBinContent(binX, value);  
    }
    
    void SetBinContent(const int binX, const int binY, const double value) { 
      cscdqm::LockType lock(mutex);
      me->setBinContent(binX, binY, value); 
    }
    
    double GetBinContent(const int binX) { 
      cscdqm::LockType lock(mutex);
      double d = me->getBinContent(binX); 
      return d;
    }
    
    double GetBinContent(const int binX, int binY) { 
      cscdqm::LockType lock(mutex);
      double d = me->getBinContent(binX, binY); 
      return d;
    }
    
    void SetAxisRange(const double from, const double to, const std::string& axis) {
      cscdqm::LockType lock(mutex);
      me->getTH1()->SetAxisRange(from, to, axis.c_str());
    }
    
    void setAxisTitle(const std::string title, const int axisN) { 
      cscdqm::LockType lock(mutex);
      me->setAxisTitle(title, axisN);  
    } 

    const int GetMaximumBin() {
      cscdqm::LockType lock(mutex);
      int i = me->getTH1()->GetMaximumBin();
      return i;
    }

    void SetNormFactor(const double factor) {
      cscdqm::LockType lock(mutex);
      me->getTH1()->SetNormFactor(factor);
    }

    const double GetEntries() {
      cscdqm::LockType lock(mutex);
      double d = me->getTH1()->GetEntries();
      return d;
    }

    void SetMaximum(const double d) {
      cscdqm::LockType lock(mutex);
      me->getTH1()->SetMaximum(d);
    }

    const TObject *getRefRootObject(void) const {
      return me->getRefRootObject();
    }

    const double GetBinError(const int bin) {
      cscdqm::LockType lock(mutex);
      double d = me->getTH1()->GetBinError(bin);
      return d;
    }

    void SetBinError(const int bin, const double error) {
      cscdqm::LockType lock(mutex);
      me->getTH1()->SetBinError(bin, error);
    }

    const TH1 *getTH1(void) const {
      return me->getTH1(); 
    }
    
    TH1 *getTH1Lock(void) { 
      return me->getTH1(); 
    }
    

};

#endif
