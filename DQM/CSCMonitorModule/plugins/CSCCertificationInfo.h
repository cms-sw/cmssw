/*
 * =====================================================================================
 *
 *       Filename:  CSCCertificationInfo.h
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

#ifndef CSCCertificationInfo_H
#define CSCCertificationInfo_H

// system include files
#include <memory>

// FWCore
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// DQM
#include <DQMServices/Core/interface/DQMEDHarvester.h>
#include <DQMServices/Core/interface/DQMStore.h>

class CSCCertificationInfo : public DQMEDHarvester {
public:
  explicit CSCCertificationInfo(const edm::ParameterSet &);
  ~CSCCertificationInfo() override {}

protected:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
  //    virtual void beginJob() { };
  //    virtual void beginLuminosityBlock(const edm::LuminosityBlock& , const  edm::EventSetup&) { }
  //    virtual void analyze(const edm::Event&, const edm::EventSetup&) { }
  //    virtual void endLuminosityBlock(const edm::LuminosityBlock& , const  edm::EventSetup&) { }
  //    virtual void endJob() { }

  std::map<std::string, MonitorElement *> mos;
};

#endif
