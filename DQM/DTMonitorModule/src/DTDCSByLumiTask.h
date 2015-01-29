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
#include "DQMServices/Core/interface/MonitorElement.h"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include <FWCore/Framework/interface/LuminosityBlock.h>

#include <vector>

class DTGeometry;
class DQMStore;
class MonitorElement;
class DTHVStatus;

class DTDCSByLumiTask: public DQMEDAnalyzer{

public:

  /// Constructor
  DTDCSByLumiTask(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DTDCSByLumiTask();

protected:

  /// Begin Run
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&);

  // Book the histograms
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  /// By Lumi DCS DB Operation
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// By Lumi DCS DB Operation
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& setup);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

private:

  std::string topFolder() const;

  int theEvents;
  int theLumis;

  bool DTHVRecordFound;

  edm::ESHandle<DTGeometry> theDTGeom;
  // CB Get HV DB and loop on half layers
  edm::ESHandle<DTHVStatus> hvStatus;

  std::vector<MonitorElement*> hActiveUnits;

};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
