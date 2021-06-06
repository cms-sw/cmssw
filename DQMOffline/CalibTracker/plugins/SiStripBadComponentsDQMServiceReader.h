#ifndef SiStripBadComponentsDQMServiceReader_H
#define SiStripBadComponentsDQMServiceReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "CondFormats/DataRecord/interface/SiStripBadStripRcd.h"

#include <sstream>
#include <string>

class TrackerTopology;
class SiStripBadStrip;

class SiStripBadComponentsDQMServiceReader : public edm::EDAnalyzer {
public:
  explicit SiStripBadComponentsDQMServiceReader(const edm::ParameterSet&);
  ~SiStripBadComponentsDQMServiceReader() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void printError(std::stringstream& ss, const bool error, const std::string& errorText);

  std::string detIdToString(DetId detid, const TrackerTopology& tTopo);

private:
  bool printdebug_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<SiStripBadStrip, SiStripBadStripRcd> badStripToken_;
};
#endif
