// -*- C++ -*-
//
// Package:    CalibTracker/SiPhase2TrackerESProducers
// Class:      SiPhase2BadStripConfigurableFakeESSource
//
/**\class SiPhase2BadStripConfigurableFakeESSource SiPhase2BadStripConfigurableFakeESSource.h CalibTracker/SiPhase2TrackerESProducers/plugins/SiPhase2BadStripConfigurableFakeESSource.cc

 Description: "fake" SiStripBadStrip ESProducer - configurable list of bad modules

 Implementation:
     Adapted to Phase-2 from CalibTracker/SiStripESProducers/plugins/fake/SiStripBadModuleConfigurableFakeESSource.cc
*/

// system include files
#include <memory>

// user include files
#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// neede for the random number generation
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/JamesRandom.h"

class SiPhase2BadStripConfigurableFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiPhase2BadStripConfigurableFakeESSource(const edm::ParameterSet&);
  ~SiPhase2BadStripConfigurableFakeESSource() override = default;

  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue& iov,
                      edm::ValidityInterval& iValidity) override;

  typedef std::unique_ptr<SiStripBadStrip> ReturnType;
  ReturnType produce(const SiPhase2OuterTrackerBadStripRcd&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  std::map<unsigned short, unsigned short> clusterizeBadChannels(
      const std::vector<Phase2TrackerDigi::PackedDigiType>& maskedChannels);

  // configurables
  bool printDebug_;
  float badComponentsFraction_;

  // random engine
  std::unique_ptr<CLHEP::HepRandomEngine> engine_;

  // es tokens
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
};

SiPhase2BadStripConfigurableFakeESSource::SiPhase2BadStripConfigurableFakeESSource(const edm::ParameterSet& iConfig)
    : engine_(new CLHEP::HepJamesRandom(iConfig.getParameter<unsigned int>("seed"))) {
  auto cc = setWhatProduced(this);
  trackTopoToken_ = cc.consumes();
  geomToken_ = cc.consumes();

  printDebug_ = iConfig.getUntrackedParameter<bool>("printDebug", false);
  badComponentsFraction_ = iConfig.getParameter<double>("badComponentsFraction");

  if (badComponentsFraction_ > 1. || badComponentsFraction_ < 0.) {
    throw cms::Exception("Inconsistent configuration")
        << "[SiPhase2BadStripChannelBuilder::c'tor] the requested fraction of bad components is unphysical. \n";
  }

  findingRecord<SiPhase2OuterTrackerBadStripRcd>();
}

void SiPhase2BadStripConfigurableFakeESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                              const edm::IOVSyncValue& iov,
                                                              edm::ValidityInterval& iValidity) {
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiPhase2BadStripConfigurableFakeESSource::ReturnType SiPhase2BadStripConfigurableFakeESSource::produce(
    const SiPhase2OuterTrackerBadStripRcd& iRecord) {
  using namespace edm::es;
  using Phase2TrackerGeomDetUnit = PixelGeomDetUnit;

  //TrackerTopology const& tTopo = iRecord.get(trackTopoToken_);
  TrackerGeometry const& tGeom = iRecord.get(geomToken_);

  auto badStrips = std::make_unique<SiStripBadStrip>();

  // early return with nullptr if fraction is == 0.f
  if (badComponentsFraction_ == 0.f) {
    return badStrips;
  }

  LogDebug("SiPhase2BadStripConfigurableFakeESSource")
      << " There are " << tGeom.detUnits().size() << " modules in this geometry.";

  int counter{0};
  for (auto const& det_u : tGeom.detUnits()) {
    const DetId detid = det_u->geographicalId();
    uint32_t rawId = detid.rawId();
    int subid = detid.subdetId();
    if (detid.det() == DetId::Detector::Tracker) {
      const Phase2TrackerGeomDetUnit* pixdet = dynamic_cast<const Phase2TrackerGeomDetUnit*>(det_u);
      assert(pixdet);
      if (subid == StripSubdetector::TOB || subid == StripSubdetector::TID) {
        if (tGeom.getDetectorType(rawId) == TrackerGeometry::ModuleType::Ph2PSS ||
            tGeom.getDetectorType(rawId) == TrackerGeometry::ModuleType::Ph2SS) {
          const PixelTopology& topol(pixdet->specificTopology());

          const int nrows = topol.nrows();
          const int ncols = topol.ncolumns();

          LogDebug("SiPhase2BadStripConfigurableFakeESSource")
              << "DetId: " << rawId << " subdet: " << subid << " nrows: " << nrows << " ncols: " << ncols;

          // auxilliary vector to check if the channels were already used
          std::vector<Phase2TrackerDigi::PackedDigiType> usedChannels;

          size_t nmaxBadStrips = std::floor(nrows * ncols * badComponentsFraction_);

          while (usedChannels.size() < nmaxBadStrips) {
            unsigned short badStripRow = std::floor(CLHEP::RandFlat::shoot(engine_.get(), 0, nrows));
            unsigned short badStripCol = std::floor(CLHEP::RandFlat::shoot(engine_.get(), 0, ncols));
            const auto& badChannel = Phase2TrackerDigi::pixelToChannel(badStripRow, badStripCol);
            if (std::find(usedChannels.begin(), usedChannels.end(), badChannel) == usedChannels.end()) {
              usedChannels.push_back(badChannel);
            }
          }

          //usedChannels.push_back(Phase2TrackerDigi::pixelToChannel(0,1)); // useful for testing

          const auto badChannelsGroups = this->clusterizeBadChannels(usedChannels);

          LogDebug("SiPhase2BadStripConfigurableFakeESSource")
              << rawId << " (" << counter << ") "
              << " masking " << nmaxBadStrips << " strips, used channels size: " << usedChannels.size()
              << ", clusters size: " << badChannelsGroups.size();

          std::vector<unsigned int> theSiStripVector;

          // loop over the groups of bad strips
          for (const auto& [first, consec] : badChannelsGroups) {
            unsigned int theBadChannelsRange;
            theBadChannelsRange = badStrips->encodePhase2(first, consec);

            if (printDebug_) {
              edm::LogInfo("SiPhase2BadStripConfigurableFakeESSource")
                  << "detid " << rawId << " \t"
                  << " firstBadStrip " << first << "\t "
                  << " NconsecutiveBadStrips " << consec << "\t "
                  << " packed integer " << std::hex << theBadChannelsRange << std::dec;
            }
            theSiStripVector.push_back(theBadChannelsRange);
          }

          SiStripBadStrip::Range range(theSiStripVector.begin(), theSiStripVector.end());
          if (!badStrips->put(rawId, range))
            edm::LogError("SiPhase2BadStripConfigurableFakeESSource")
                << "[SiPhase2BadStripConfigurableFakeESSource::produce] detid already exists";

          counter++;

        }  // if it's a strip module
      }    // if it's OT
    }      // if it's Tracker
  }        // loop on DetIds

  LogDebug("SiPhase2BadStripConfigurableFakeESSource") << "end of the detId loops";

  return badStrips;
}

// poor-man clusterizing algorithm
std::map<unsigned short, unsigned short> SiPhase2BadStripConfigurableFakeESSource::clusterizeBadChannels(
    const std::vector<Phase2TrackerDigi::PackedDigiType>& maskedChannels) {
  // Here we will store the result
  std::map<unsigned short, unsigned short> result{};
  std::map<int, std::string> printresult{};

  // Sort and remove duplicates.
  std::set data(maskedChannels.begin(), maskedChannels.end());

  // We will start the evaluation at the beginning of our data
  auto startOfSequence = data.begin();

  // Find all sequences
  while (startOfSequence != data.end()) {
    // Find first value that is not greater than one
    auto endOfSequence =
        std::adjacent_find(startOfSequence, data.end(), [](const auto& v1, const auto& v2) { return v2 != v1 + 1; });
    if (endOfSequence != data.end())
      std::advance(endOfSequence, 1);

    auto consecutiveStrips = std::distance(startOfSequence, endOfSequence);
    result[*startOfSequence] = consecutiveStrips;

    if (printDebug_) {
      // Build resulting string
      std::ostringstream oss{};
      bool writeDash = false;
      for (auto it = startOfSequence; it != endOfSequence; ++it) {
        oss << (writeDash ? "-" : "") << std::to_string(*it);
        writeDash = true;
      }

      // Copy result to map
      for (auto it = startOfSequence; it != endOfSequence; ++it)
        printresult[*it] = oss.str();
    }

    // Continue to search for the next sequence
    startOfSequence = endOfSequence;
  }

  if (printDebug_) {
    // Show result on the screen. Or use the map in whichever way you want.
    for (const auto& [value, text] : printresult)
      LogDebug("SiPhase2BadStripConfigurableFakeESSource")
          << std::left << std::setw(2) << value << " -> " << text << "\n";
  }
  return result;
}

void SiPhase2BadStripConfigurableFakeESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Configurable Fake Phase-2 Outer Tracker Bad Strip ESSource");
  desc.add<unsigned int>("seed", 1)->setComment("random seed");
  desc.addUntracked<bool>("printDebug", false)->setComment("maximum amount of print-outs");
  desc.add<double>("badComponentsFraction", 0.01)->setComment("fraction of bad components to populate the ES");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiPhase2BadStripConfigurableFakeESSource);
