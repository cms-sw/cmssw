#ifndef SiStripBadComponentsDQMServiceReader_H
#define SiStripBadComponentsDQMServiceReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <sstream>
#include <string>


class TrackerTopology;

class SiStripBadComponentsDQMServiceReader : public edm::EDAnalyzer {

 public:
  explicit SiStripBadComponentsDQMServiceReader( const edm::ParameterSet& );
  ~SiStripBadComponentsDQMServiceReader();

  void analyze( const edm::Event&, const edm::EventSetup& );

  void printError( std::stringstream & ss, const bool error, const std::string & errorText );

  std::string detIdToString(const DetId & detid, const TrackerTopology* tTopo);

 private:
  bool printdebug_;
};
#endif
