#ifndef DTTestPulsesTask_H
#define DTTestPulsesTask_H

/*
 * \file DTTestPulsesTask.h
 *
 * \author M. Zanetti - INFN Padova
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
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>

class DTGeometry;
class DTLayerId;
class DTRangeT0;

class DTTestPulsesTask: public DQMEDAnalyzer{

public:

  /// Constructor
  DTTestPulsesTask(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DTTestPulsesTask();

protected:

  /// BeginRun
  void dqmBeginRun(const edm::Run& , const edm::EventSetup&);

  // Book the histograms
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  /// Book the ME
  void bookHistos(DQMStore::IBooker & ibooker, std::string folder, std::string histoTag);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);


private:

  int nevents;

  edm::ParameterSet parameters;

  edm::ESHandle<DTGeometry> muonGeom;
  edm::ESHandle<DTRangeT0> t0RangeMap;

  edm::EDGetTokenT<DTDigiCollection> dtDigisToken_; // dtunpacker
  std::pair <int, int> t0sPeakRange;

  // My monitor elements
  std::map<int, MonitorElement*> testPulsesProfiles;
  std::map<int, MonitorElement*> testPulsesOccupancies;
  std::map<int, MonitorElement*> testPulsesTimeBoxes;


};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
