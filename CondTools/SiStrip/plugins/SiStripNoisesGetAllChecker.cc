#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <iostream>
#include <exception>

class SiStripNoisesGetAllChecker : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripNoisesGetAllChecker(const edm::ParameterSet&);
  ~SiStripNoisesGetAllChecker() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noisesToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  void checkModuleNoise(const SiStripNoises&, const uint32_t detID, uint16_t maxNStrips);
};

SiStripNoisesGetAllChecker::SiStripNoisesGetAllChecker(const edm::ParameterSet&)
    : noisesToken_(esConsumes()), tkGeomToken_(esConsumes()) {}

void SiStripNoisesGetAllChecker::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  const auto& siStripNoises = iSetup.getData(noisesToken_);

  const auto& tkGeom = &iSetup.getData(tkGeomToken_);
  const auto& tkDets = tkGeom->dets();

  edm::LogInfo("SiStripNoisesGetAllChecker") << "Starting to loop over all SiStrip modules...";

  // Get all DetIDs associated with SiStripNoises
  std::vector<uint32_t> detIDs;
  siStripNoises.getDetIds(detIDs);

  size_t exceptionCounts{0};
  for (const auto& detID : detIDs) {
    uint16_t maxNStrips{0};
    auto det = std::find_if(tkDets.begin(), tkDets.end(), [detID](auto& elem) -> bool {
      return (elem->geographicalId().rawId() == detID);
    });
    const StripTopology& p = dynamic_cast<const StripGeomDetUnit*>(*det)->specificTopology();
    maxNStrips = p.nstrips();

    try {
      checkModuleNoise(siStripNoises, detID, maxNStrips);
    } catch (const std::exception& e) {
      // Increment the exception counter if checkModuleNoise itself throws an exception
      edm::LogError("SiStripNoisesGetAllChecker")
          << "Exception in checkModuleNoise for detID " << detID << ": " << e.what();
      ++exceptionCounts;
    } catch (...) {
      edm::LogError("SiStripNoisesGetAllChecker") << "Unknown exception in checkModuleNoise for detID " << detID;
      ++exceptionCounts;
    }
  }

  std::ostringstream message;

  // Define the box width
  const int boxWidth = 50;

  message << "\n"
          << std::string(boxWidth, '*') << "\n"
          << "* " << std::setw(boxWidth - 4) << std::left << "SiStripNoisesGetAllChecker Summary"
          << " *\n"
          << std::string(boxWidth, '*') << "\n"
          << "* " << std::setw(boxWidth - 4) << std::left
          << ("Completed loop over " + std::to_string(detIDs.size()) + " SiStrip modules.") << " *\n"
          << "* " << std::setw(boxWidth - 4) << std::left
          << ("Encountered " + std::to_string(exceptionCounts) + " exceptions.") << " *\n"
          << std::string(boxWidth, '*');

  edm::LogSystem("SiStripNoisesGetAllChecker") << message.str();
}

void SiStripNoisesGetAllChecker::checkModuleNoise(const SiStripNoises& siStripNoises,
                                                  const uint32_t detID,
                                                  uint16_t maxNStrips) {
  try {
    SiStripNoises::Range detNoiseRange = siStripNoises.getRange(detID);
    std::vector<float> noises;
    noises.resize(maxNStrips);
    siStripNoises.allNoises(noises, detNoiseRange);
    edm::LogInfo("SiStripNoisesGetAllChecker") << "Successfully processed detID: " << detID;
  } catch (const std::exception& e) {
    edm::LogError("SiStripNoisesGetAllChecker") << "Exception caught for detID " << detID << ": " << e.what();
    throw;  // Re-throw the exception to allow the outer loop to handle it
  } catch (...) {
    edm::LogError("SiStripNoisesGetAllChecker") << "Unknown exception caught for detID " << detID;
    throw;  // Re-throw the unknown exception
  }
}

// Define this as a plug-in
DEFINE_FWK_MODULE(SiStripNoisesGetAllChecker);
