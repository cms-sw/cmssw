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

#include "DQMServices/Core/interface/DQMStore.h"
#include "CSCDQM_MonitorObject.h"

/**
 * @class CSCMonitorObject
 * @brief cscdqm::MonitorObject implementation used in CSCMonitorModuleCmn
 */
class CSCMonitorObject : public cscdqm::MonitorObject {
protected:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

private:
  MonitorElement *me;

public:
  CSCMonitorObject(MonitorElement *p_me) { me = p_me; }
  ~CSCMonitorObject() override {}

  void Fill(float x) override {
    cscdqm::LockType lock(mutex);
    me->Fill(x);
  }

  void Fill(float x, float yw) override {
    cscdqm::LockType lock(mutex);
    me->Fill(x, yw);
  }

  void Fill(float x, float y, float zw) override {
    cscdqm::LockType lock(mutex);
    me->Fill(x, y, zw);
  }

  void Fill(float x, float y, float z, float w) override {
    cscdqm::LockType lock(mutex);
    me->Fill(x, y, z, w);
  }

  void SetEntries(const double value) override {
    cscdqm::LockType lock(mutex);
    me->setEntries(value);
  }

  void SetBinContent(const int binX, const double value) override {
    cscdqm::LockType lock(mutex);
    me->setBinContent(binX, value);
  }

  void SetBinContent(const int binX, const int binY, const double value) override {
    cscdqm::LockType lock(mutex);
    me->setBinContent(binX, binY, value);
  }

  double GetBinContent(const int binX) override {
    cscdqm::LockType lock(mutex);
    double d = me->getBinContent(binX);
    return d;
  }

  double GetBinContent(const int binX, int binY) override {
    cscdqm::LockType lock(mutex);
    double d = me->getBinContent(binX, binY);
    return d;
  }

  void SetAxisRange(const double from, const double to, const std::string &axis) override {
    cscdqm::LockType lock(mutex);
    me->getTH1()->SetAxisRange(from, to, axis.c_str());
  }

  void setAxisTitle(const std::string title, const int axisN) override {
    cscdqm::LockType lock(mutex);
    me->setAxisTitle(title, axisN);
  }

  const int GetMaximumBin() override {
    cscdqm::LockType lock(mutex);
    int i = me->getTH1()->GetMaximumBin();
    return i;
  }

  void SetNormFactor(const double factor) override {
    cscdqm::LockType lock(mutex);
    me->getTH1()->SetNormFactor(factor);
  }

  const double GetEntries() override {
    cscdqm::LockType lock(mutex);
    double d = me->getTH1()->GetEntries();
    return d;
  }

  void SetMaximum(const double d) override {
    cscdqm::LockType lock(mutex);
    me->getTH1()->SetMaximum(d);
  }

  const TObject *getRefRootObject(void) const override { return nullptr; }

  const double GetBinError(const int bin) override {
    cscdqm::LockType lock(mutex);
    double d = me->getTH1()->GetBinError(bin);
    return d;
  }

  void SetBinError(const int bin, const double error) override {
    cscdqm::LockType lock(mutex);
    me->getTH1()->SetBinError(bin, error);
  }

  const TH1 *getTH1(void) const override { return me->getTH1(); }

  TH1 *getTH1Lock(void) override { return me->getTH1(); }
};

#endif
