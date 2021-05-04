// -*- C++ -*-
//
// Package:    CondTools/SiPhase2Tracker
// Class:      DTCCablingMapTestReader
//
/**\class DTCCablingMapTestReader DTCCablingMapTestReader.cc CondTools/SiPhase2Tracker/plugins/DTCCablingMapTestReader.cc

Description: [one line class summary]

Implementation:
		[Notes on implementation]
*/
//
// Original Author:  Luigi Calligaris, SPRACE, São Paulo, BR
// Created        :  Wed, 27 Feb 2019 21:41:13 GMT
//
//

#include <memory>
#include <utility>
#include <unordered_map>

#include <string>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"

class DTCCablingMapTestReader : public edm::one::EDAnalyzer<> {
public:
  explicit DTCCablingMapTestReader(const edm::ParameterSet&);
  ~DTCCablingMapTestReader() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
};

void DTCCablingMapTestReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.add("DTCCablingMapTestReader", desc);
}

DTCCablingMapTestReader::DTCCablingMapTestReader(const edm::ParameterSet& iConfig) : cablingMapToken_(esConsumes()) {}

DTCCablingMapTestReader::~DTCCablingMapTestReader() {}

void DTCCablingMapTestReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  const auto p_cablingMap = &iSetup.getData(cablingMapToken_);

  {
    ostringstream dump_DetToElink;

    dump_DetToElink << "Det To DTC ELink map elements dump (Python-style):" << endl;
    std::vector<uint32_t> const knownDetIds = p_cablingMap->getKnownDetIds();

    dump_DetToElink << "{";
    for (uint32_t detId : knownDetIds) {
      dump_DetToElink << "(" << detId << " : [";
      auto equal_range = p_cablingMap->detIdToDTCELinkId(detId);

      for (auto it = equal_range.first; it != equal_range.second; ++it)
        dump_DetToElink << "(" << unsigned(it->second.dtc_id()) << ", " << unsigned(it->second.gbtlink_id()) << ", "
                        << unsigned(it->second.elink_id()) << "), ";

      dump_DetToElink << "], ";
    }
    dump_DetToElink << "}" << endl;

    edm::LogInfo("DetToElinkCablingMapDump") << dump_DetToElink.str();
  }

  {
    ostringstream dump_ElinkToDet;

    dump_ElinkToDet << "DTC Elink To Det map elements dump (Python-style):" << endl;
    std::vector<DTCELinkId> const knownDTCELinkIds = p_cablingMap->getKnownDTCELinkIds();

    dump_ElinkToDet << "{";
    for (DTCELinkId const& currentELink : knownDTCELinkIds) {
      dump_ElinkToDet << "(" << unsigned(currentELink.dtc_id()) << ", " << unsigned(currentELink.gbtlink_id()) << ", "
                      << unsigned(currentELink.elink_id()) << ") "
                      << " : ";
      auto detId_it = p_cablingMap->dtcELinkIdToDetId(currentELink);

      dump_ElinkToDet << detId_it->second << ", ";
    }
    dump_ElinkToDet << "}" << endl;

    edm::LogInfo("DetToElinkCablingMapDump") << dump_ElinkToDet.str();
  }
}

void DTCCablingMapTestReader::beginJob() {}

void DTCCablingMapTestReader::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(DTCCablingMapTestReader);
