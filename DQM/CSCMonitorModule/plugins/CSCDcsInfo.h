/*
 * =====================================================================================
 *
 *       Filename:  CSCDcsInfo.h
 *
 *    Description:  CSC DCS Information
 *
 *        Version:  1.0
 *        Created:  12/09/2008 10:53:27 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDcsInfo_H
#define CSCDcsInfo_H

// system include files
#include <memory>

// FWCore
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// DQM
#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/DQMEDHarvester.h>

class CSCDcsInfo : public DQMEDHarvester {
public:
  explicit CSCDcsInfo(const edm::ParameterSet &);
  ~CSCDcsInfo() override {}

protected:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
  std::map<std::string, MonitorElement *> mos;
};

#endif
