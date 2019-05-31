// -*- C++ -*-
//
// Package:    CondTools/SiPhase2Tracker
// Class:      DTCCablingMapProducer
//
/**\class DTCCablingMapProducer DTCCablingMapProducer.cc CondTools/SiPhase2Tracker/plugins/DTCCablingMapProducer.cc

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
#include <cstdint>
#include <unordered_map>
#include <utility>

#include "FWCore/Framework/interface/Frameworkfwd.h"
// #include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"

//
// CONSTANTS
//

static constexpr const unsigned int gbt_id_minvalue = 0;
static constexpr const unsigned int gbt_id_maxvalue = 72;
static constexpr const unsigned int elink_id_minvalue = 0;
static constexpr const unsigned int elink_id_maxvalue = 7;

//
// SOME HELPER FUNCTIONS
//

// trim from start (in place)
static inline void ltrim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
static inline void rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string& s) {
  ltrim(s);
  rtrim(s);
}

class DTCCablingMapProducer : public edm::one::EDAnalyzer<> {
public:
  explicit DTCCablingMapProducer(const edm::ParameterSet&);
  ~DTCCablingMapProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  virtual void LoadModulesToDTCCablingMapFromCSV(std::vector<std::string> const&);

private:
  bool generate_fake_valid_gbtlink_and_elinkid_;
  int verbosity_;
  unsigned csvFormat_ncolumns_;
  unsigned csvFormat_idetid_;
  unsigned csvFormat_idtcid_;
  unsigned csvFormat_igbtlinkid_;
  unsigned csvFormat_ielinkid_;
  cond::Time_t iovBeginTime_;
  std::unique_ptr<TrackerDetToDTCELinkCablingMap> pCablingMap_;
  std::string record_;
};

void DTCCablingMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Stores a TrackerDetToDTCELinkCablingMap object into the database from a CSV file.");
  desc.add<bool>("generate_fake_valid_gbtlink_and_elinkid", false);
  desc.add<int>("verbosity", 0);
  desc.add<unsigned>("csvFormat_idetid", 0);
  desc.add<unsigned>("csvFormat_ncolumns", 15);
  desc.add<unsigned>("csvFormat_idtcid", 10);
  desc.add<unsigned>("csvFormat_igbtlinkid", 11);
  desc.add<unsigned>("csvFormat_ielinkid", 12);
  desc.add<long long unsigned int>("iovBeginTime", 1);
  desc.add<std::string>("record", "TrackerDTCCablingMapRcd");
  desc.add<std::vector<std::string>>("modulesToDTCCablingCSVFileNames", std::vector<std::string>());
  descriptions.add("DTCCablingMapProducer", desc);
}

DTCCablingMapProducer::DTCCablingMapProducer(const edm::ParameterSet& iConfig)
    : generate_fake_valid_gbtlink_and_elinkid_(iConfig.getParameter<bool>("generate_fake_valid_gbtlink_and_elinkid")),
      verbosity_(iConfig.getParameter<int>("verbosity")),
      csvFormat_ncolumns_(iConfig.getParameter<unsigned>("csvFormat_ncolumns")),
      csvFormat_idetid_(iConfig.getParameter<unsigned>("csvFormat_idetid")),
      csvFormat_idtcid_(iConfig.getParameter<unsigned>("csvFormat_idtcid")),
      iovBeginTime_(iConfig.getParameter<long long unsigned int>("iovBeginTime")),
      pCablingMap_(std::make_unique<TrackerDetToDTCELinkCablingMap>()),
      record_(iConfig.getParameter<std::string>("record")) {
  LoadModulesToDTCCablingMapFromCSV(iConfig.getParameter<std::vector<std::string>>("modulesToDTCCablingCSVFileNames"));
}

void DTCCablingMapProducer::beginJob() {}

void DTCCablingMapProducer::LoadModulesToDTCCablingMapFromCSV(
    std::vector<std::string> const& modulesToDTCCablingCSVFileNames) {
  using namespace std;

  for (std::string const& csvFileName : modulesToDTCCablingCSVFileNames) {
    edm::FileInPath csvFilePath(csvFileName);

    ifstream csvFile;
    csvFile.open(csvFilePath.fullPath().c_str());

    if (csvFile.is_open()) {
      string csvLine;

      unsigned lineNumber = 0;

      while (std::getline(csvFile, csvLine)) {
        if (verbosity_ >= 1) {
          edm::LogInfo("CSVParser") << "Reading CSV file line: " << ++lineNumber << ": \"" << csvLine << "\"" << endl;
        }

        istringstream csvStream(csvLine);
        vector<string> csvColumn;
        string csvElement;

        while (std::getline(csvStream, csvElement, ',')) {
          trim(csvElement);
          csvColumn.push_back(csvElement);
        }

        if (verbosity_ >= 2) {
          ostringstream splitted_line_info;

          splitted_line_info << "-- split line is: [";

          for (string const& s : csvColumn)
            splitted_line_info << "\"" << s << "\", ";

          splitted_line_info << "]" << endl;

          edm::LogInfo("CSVParser") << splitted_line_info.str();
        }

        if (csvColumn.size() == csvFormat_ncolumns_) {
          // Skip the legend lines
          if (0 == csvColumn[0].compare(std::string("Module DetId/U"))) {
            if (verbosity_ >= 1) {
              edm::LogInfo("CSVParser") << "-- skipping legend line" << endl;
            }
            continue;
          }

          uint32_t detIdRaw;

          try {
            detIdRaw = std::stoi(csvColumn[csvFormat_idetid_]);
          } catch (std::exception const& e) {
            if (verbosity_ >= 0) {
              edm::LogError("CSVParser") << "-- malformed DetId string in CSV file: \"" << csvLine << "\"" << endl;
            }
            throw e;
          }

          unsigned const dtc_id = strtoul(csvColumn[csvFormat_idtcid_].c_str(), nullptr, 10);
          unsigned gbt_id;
          unsigned elink_id;

          if (generate_fake_valid_gbtlink_and_elinkid_) {
            for (gbt_id = gbt_id_minvalue; gbt_id < gbt_id_maxvalue + 1u; ++gbt_id) {
              for (elink_id = elink_id_minvalue; elink_id < elink_id_maxvalue + 1u; ++elink_id) {
                if (!(pCablingMap_->knowsDTCELinkId(DTCELinkId(dtc_id, gbt_id, elink_id))))
                  goto gbtlink_and_elinkid_generator_end;  //break out of this double loop
              }
            }
          gbtlink_and_elinkid_generator_end:
            ((void)0);  // This is a NOP, it's here just to have a valid (although dummy) instruction after the goto tag
          } else {
            gbt_id = strtoul(csvColumn[csvFormat_igbtlinkid_].c_str(), nullptr, 10);
            elink_id = strtoul(csvColumn[csvFormat_ielinkid_].c_str(), nullptr, 10);
          }

          DTCELinkId dtcELinkId(dtc_id, gbt_id, elink_id);

          if (verbosity_ >= 3) {
            edm::LogInfo("CSVParser") << "-- DetId = " << detIdRaw << " (dtc_id, gbt_id, elink_id) = (" << dtc_id << ","
                                      << gbt_id << "," << elink_id << ")" << endl;
          }

          if (pCablingMap_->knowsDTCELinkId(dtcELinkId)) {
            ostringstream message;
            message
                << "Reading CSV file: CRITICAL ERROR, duplicated dtcELinkId entry about (dtc_id, gbt_id, elink_id) = ("
                << dtc_id << "," << gbt_id << "," << elink_id << ")";

            throw cms::Exception(message.str());
          }

          pCablingMap_->insert(dtcELinkId, detIdRaw);
        } else {
          if (verbosity_ >= 3) {
            edm::LogInfo("CSVParser") << "Reading CSV file: Skipped a short line: \"" << csvLine << "\"" << endl;
          }
        }
      }
    } else {
      throw cms::Exception("DTCCablingMapProducer: Unable to open input CSV file") << csvFilePath << endl;
    }

    csvFile.close();
  }
}

void DTCCablingMapProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

void DTCCablingMapProducer::endJob() {
  // 	using namespace edm;
  using namespace std;

  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if (poolDbService.isAvailable()) {
    poolDbService->writeOne(pCablingMap_.release(), iovBeginTime_, record_);
  } else {
    throw cms::Exception("PoolDBService required.");
  }
}

DTCCablingMapProducer::~DTCCablingMapProducer() {}

//define this as a plug-in
DEFINE_FWK_MODULE(DTCCablingMapProducer);
