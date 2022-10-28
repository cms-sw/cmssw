/*
 * =====================================================================================
 *
 *       Filename:  CSCDaqInfo.h
 *
 *    Description:  CSC DAQ Information
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

#ifndef CSCDaqInfo_H
#define CSCDaqInfo_H

// system include files
#include <memory>

// FWCore
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// DQM
#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/DQMEDHarvester.h>

class CSCDaqInfo : public DQMEDHarvester {
public:
  explicit CSCDaqInfo(const edm::ParameterSet &);
  ~CSCDaqInfo() override {}

protected:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
  std::map<std::string, MonitorElement *> mos;
};

#endif
