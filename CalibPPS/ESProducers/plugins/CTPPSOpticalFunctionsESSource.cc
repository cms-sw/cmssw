// Original Author:  Jan Kašpar

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsSetCollection.h"
#include "CondFormats/DataRecord/interface/CTPPSOpticsRcd.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Loads optical functions from ROOT files.
 **/
class CTPPSOpticalFunctionsESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  CTPPSOpticalFunctionsESSource(const edm::ParameterSet &);
  ~CTPPSOpticalFunctionsESSource() override = default;

  std::unique_ptr<LHCOpticalFunctionsSetCollection> produce(const CTPPSOpticsRcd &);
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;

  std::string m_label;

  struct FileInfo {
    double m_xangle;
    std::string m_fileName;
  };

  struct RPInfo {
    std::string m_dirName;
    double m_scoringPlaneZ;
  };

  struct Entry {
    edm::EventRange m_validityRange;
    std::vector<FileInfo> m_fileInfo;
    std::unordered_map<unsigned int, RPInfo> m_rpInfo;
  };

  std::vector<Entry> m_entries;

  bool m_currentEntryValid;
  unsigned int m_currentEntry;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSOpticalFunctionsESSource::CTPPSOpticalFunctionsESSource(const edm::ParameterSet &conf)
    : m_label(conf.getParameter<std::string>("label")), m_currentEntryValid(false), m_currentEntry(0) {
  for (const auto &entry_pset : conf.getParameter<std::vector<edm::ParameterSet>>("configuration")) {
    edm::EventRange validityRange = entry_pset.getParameter<edm::EventRange>("validityRange");

    std::vector<FileInfo> fileInfo;
    for (const auto &pset : entry_pset.getParameter<std::vector<edm::ParameterSet>>("opticalFunctions")) {
      const double &xangle = pset.getParameter<double>("xangle");
      const std::string &fileName = pset.getParameter<edm::FileInPath>("fileName").fullPath();
      fileInfo.push_back({xangle, fileName});
    }

    std::unordered_map<unsigned int, RPInfo> rpInfo;
    for (const auto &pset : entry_pset.getParameter<std::vector<edm::ParameterSet>>("scoringPlanes")) {
      const unsigned int rpId = pset.getParameter<unsigned int>("rpId");
      const std::string dirName = pset.getParameter<std::string>("dirName");
      const double z = pset.getParameter<double>("z");
      const RPInfo entry = {dirName, z};
      rpInfo.emplace(rpId, entry);
    }

    m_entries.push_back({validityRange, fileInfo, rpInfo});
  }

  setWhatProduced(this, m_label);
  findingRecord<CTPPSOpticsRcd>();
}

//----------------------------------------------------------------------------------------------------

void CTPPSOpticalFunctionsESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
                                                   const edm::IOVSyncValue &iosv,
                                                   edm::ValidityInterval &oValidity) {
  for (unsigned int idx = 0; idx < m_entries.size(); ++idx) {
    const auto &entry = m_entries[idx];

    // is within an entry ?
    if (edm::contains(entry.m_validityRange, iosv.eventID())) {
      m_currentEntryValid = true;
      m_currentEntry = idx;
      oValidity = edm::ValidityInterval(edm::IOVSyncValue(entry.m_validityRange.startEventID()),
                                        edm::IOVSyncValue(entry.m_validityRange.endEventID()));
      return;
    }
  }

  // not within any entry
  m_currentEntryValid = false;
  m_currentEntry = 0;

  edm::LogInfo("") << "No configuration entry found for event " << iosv.eventID()
                   << ", no optical functions will be available.";

  const edm::EventID start(iosv.eventID().run(), iosv.eventID().luminosityBlock(), iosv.eventID().event());
  const edm::EventID end(iosv.eventID().run(), iosv.eventID().luminosityBlock(), iosv.eventID().event());
  oValidity = edm::ValidityInterval(edm::IOVSyncValue(start), edm::IOVSyncValue(end));
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<LHCOpticalFunctionsSetCollection> CTPPSOpticalFunctionsESSource::produce(const CTPPSOpticsRcd &) {
  // prepare output, empty by default
  auto output = std::make_unique<LHCOpticalFunctionsSetCollection>();

  // fill the output
  if (m_currentEntryValid) {
    const auto &entry = m_entries[m_currentEntry];

    for (const auto &fi : entry.m_fileInfo) {
      std::unordered_map<unsigned int, LHCOpticalFunctionsSet> xa_data;

      for (const auto &rpi : entry.m_rpInfo) {
        LHCOpticalFunctionsSet fcn(fi.m_fileName, rpi.second.m_dirName, rpi.second.m_scoringPlaneZ);
        xa_data.emplace(rpi.first, std::move(fcn));
      }

      output->emplace(fi.m_xangle, xa_data);
    }
  }

  // commit the output
  return output;
}

//----------------------------------------------------------------------------------------------------

void CTPPSOpticalFunctionsESSource::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("label", "")->setComment("label of the optics record");

  edm::ParameterSetDescription config_desc;

  config_desc.add<edm::EventRange>("validityRange", edm::EventRange())->setComment("interval of validity");

  edm::ParameterSetDescription of_desc;
  of_desc.add<double>("xangle")->setComment("half crossing angle value in urad");
  of_desc.add<edm::FileInPath>("fileName")->setComment("ROOT file with optical functions");
  std::vector<edm::ParameterSet> of;
  config_desc.addVPSet("opticalFunctions", of_desc, of)
      ->setComment("list of optical functions at different crossing angles");

  edm::ParameterSetDescription sp_desc;
  sp_desc.add<unsigned int>("rpId")->setComment("associated detector DetId");
  sp_desc.add<std::string>("dirName")->setComment("associated path to the optical functions file");
  sp_desc.add<double>("z")->setComment("longitudinal position at scoring plane/detector");
  std::vector<edm::ParameterSet> sp;
  config_desc.addVPSet("scoringPlanes", sp_desc, sp)->setComment("list of sensitive planes/detectors stations");

  std::vector<edm::ParameterSet> config;
  desc.addVPSet("configuration", config_desc, sp)->setComment("list of configuration blocks");

  descriptions.add("ctppsOpticalFunctionsESSource", desc);
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(CTPPSOpticalFunctionsESSource);
