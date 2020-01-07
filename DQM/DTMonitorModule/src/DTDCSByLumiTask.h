#ifndef DTDCSByLumiTask_H
#define DTDCSByLumiTask_H

/*
 * \file DTDCSByLumiTask.h
 *
 * \author C. Battilana - CIEMAT
 * \author P. Bellan - INFN PD
 * \author A. Branca = INFN PD
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DQMServices/Core/interface/DQMOneEDAnalyzer.h>

#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/DataRecord/interface/DTHVStatusRcd.h"
#include "CondFormats/DTObjects/interface/DTHVStatus.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <vector>

class DTDCSByLumiTask : public DQMOneLumiEDAnalyzer<> {
public:
  /// Constructor
  DTDCSByLumiTask(const edm::ParameterSet& ps);

  /// Destructor
  ~DTDCSByLumiTask() override;

protected:
  /// Begin Run
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  // Book the histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  /// By Lumi DCS DB Operation
  void dqmBeginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const&) override;

  /// By Lumi DCS DB Operation
  void dqmEndLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup&) override;

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup&) override;

private:
  std::string topFolder() const;

  int theEvents;
  int theLumis;

  bool DTHVRecordFound;

  edm::ESHandle<DTGeometry> theDTGeom;

  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeometryToken_;
  edm::ESGetToken<DTHVStatus, DTHVStatusRcd> dtHVStatusToken_;

  std::vector<MonitorElement*> hActiveUnits;
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
