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
#include "DQM/CSCMonitorModule/interface/CSCDQM_MonitorObject.h"

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
      lock();
      me->Fill(x); 
      unlock();
    }

    void Fill(float x, float yw) { 
      lock();
      me->Fill(x, yw); 
      unlock();
    }

    void Fill(float x, float y, float zw) { 
      lock();
      me->Fill(x, y, zw); 
      unlock();
    }

    void Fill(float x, float y, float z, float w) { 
      lock();
      me->Fill(x, y, z, w); 
      unlock();
    }

    void SetEntries(const double value) { 
      lock();
      me->setEntries(value); 
      unlock();
    }
    
    void SetBinContent(const int binX, const double value) { 
      lock();
      me->setBinContent(binX, value);  
      unlock();
    }
    
    void SetBinContent(const int binX, const int binY, const double value) { 
      lock();
      me->setBinContent(binX, binY, value); 
      unlock();
    }
    
    double GetBinContent(const int binX) { 
      lock();
      double d = me->getBinContent(binX); 
      unlock();
      return d;
    }
    
    double GetBinContent(const int binX, int binY) { 
      lock();
      double d = me->getBinContent(binX, binY); 
      unlock();
      return d;
    }
    
    void SetAxisRange(const double from, const double to, const std::string& axis) {
      lock();
      me->getTH1()->SetAxisRange(from, to, axis.c_str());
      unlock();
    }
    
    void setAxisTitle(const std::string title, const int axisN) { 
      lock();
      me->setAxisTitle(title, axisN);  
      unlock();
    } 

    const int GetMaximumBin() {
      lock();
      int i = me->getTH1()->GetMaximumBin();
      unlock();
      return i;
    }

    void SetNormFactor(const double factor) {
      lock();
      me->getTH1()->SetNormFactor(factor);
      unlock();
    }

    const double GetEntries() {
      lock();
      double d = me->getTH1()->GetEntries();
      unlock();
      return d;
    }

    void SetMaximum(const double d) {
      lock();
      me->getTH1()->SetMaximum(d);
      unlock();
    }

    const TObject *getRefRootObject(void) const {
      return me->getRefRootObject();
    }

    const double GetBinError(const int bin) {
      lock();
      double d = me->getTH1()->GetBinError(bin);
      unlock();
      return d;
    }

    void SetBinError(const int bin, const double error) {
      lock();
      me->getTH1()->SetBinError(bin, error);
      unlock();
    }

    const TH1 *getTH1(void) const { 
      return me->getTH1(); 
    }
    
    TH1 *getTH1Lock(void) { 
      lock();
      return me->getTH1(); 
    }
    

};

#endif
