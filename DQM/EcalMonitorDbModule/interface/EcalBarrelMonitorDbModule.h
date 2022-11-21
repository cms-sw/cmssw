#ifndef EcalBarrelMonitorDbModule_H
#define EcalBarrelMonitorDbModule_H

/*
 * \file EcalBarrelMonitorDbModule.h
 *
 * \author G. Della Ricca
 *
 */

#include <string>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RelationalAccess/ISessionProxy.h"
#include "DQMServices/Core/interface/DQMStore.h"

class MonitorElementsDb;

class EcalBarrelMonitorDbModule : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  /// Constructor
  EcalBarrelMonitorDbModule(const edm::ParameterSet &ps);

  /// Destructor
  ~EcalBarrelMonitorDbModule() override;

protected:
  /// Analyze
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

  // BeginJob
  void beginJob(void) override;

  // EndJob
  void endJob(void) override;

private:
  int icycle_;

  DQMStore *dqmStore_;

  std::string prefixME_;

  std::string htmlDir_;

  std::string xmlFile_;

  MonitorElementsDb *ME_Db_;

  unsigned int sleepTime_;

  coral::ISessionProxy *session_;
};

#endif
