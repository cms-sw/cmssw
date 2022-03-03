/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Edoardo Bossini
 *   Filip Dej
 *   Laurent Forthomme
 *   Christopher Misan
 *
 * NOTE:
 *   Given implementation handles calibration files in JSON format,
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/PPSObjects/interface/PPSTimingCalibrationLUT.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationLUTRcd.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;

//------------------------------------------------------------------------------

class PPSTimingCalibrationLUTESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  PPSTimingCalibrationLUTESSource(const edm::ParameterSet&);

  edm::ESProducts<std::unique_ptr<PPSTimingCalibrationLUT> > produce(const PPSTimingCalibrationLUTRcd&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

  /// Extract calibration data from JSON file (PPS horizontal diamond)
  std::unique_ptr<PPSTimingCalibrationLUT> parsePPSDiamondLUTJsonFile() const;

  const std::string filename_;
};

//------------------------------------------------------------------------------

PPSTimingCalibrationLUTESSource::PPSTimingCalibrationLUTESSource(const edm::ParameterSet& iConfig)
    : filename_(iConfig.getParameter<edm::FileInPath>("calibrationFile").fullPath()) {
  setWhatProduced(this);
  findingRecord<PPSTimingCalibrationLUTRcd>();
}

//------------------------------------------------------------------------------

edm::ESProducts<std::unique_ptr<PPSTimingCalibrationLUT> > PPSTimingCalibrationLUTESSource::produce(
    const PPSTimingCalibrationLUTRcd&) {
  return edm::es::products(parsePPSDiamondLUTJsonFile());
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationLUTESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                     const edm::IOVSyncValue&,
                                                     edm::ValidityInterval& oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

//------------------------------------------------------------------------------

std::unique_ptr<PPSTimingCalibrationLUT> PPSTimingCalibrationLUTESSource::parsePPSDiamondLUTJsonFile() const {
  pt::ptree mother_node;
  pt::read_json(filename_, mother_node);

  PPSTimingCalibrationLUT::BinMap binMap;
  for (pt::ptree::value_type& node : mother_node.get_child("calib")) {
    PPSTimingCalibrationLUT::Key key;

    key.sector = node.second.get<int>("sector");
    key.station = node.second.get<int>("station");
    key.plane = node.second.get<int>("plane");
    key.channel = node.second.get<int>("channel");
    std::vector<double> values;
    for (pt::ptree::value_type& sample : node.second.get_child("samples")) {
      values.emplace_back(std::stod(sample.second.data(), nullptr));
      binMap[key] = values;
    }
  }
  return std::make_unique<PPSTimingCalibrationLUT>(binMap);
}

void PPSTimingCalibrationLUTESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::FileInPath>("calibrationFile", edm::FileInPath())->setComment("file with calibrations");

  descriptions.add("ppsTimingCalibrationLUTESSource", desc);
}

DEFINE_FWK_EVENTSETUP_SOURCE(PPSTimingCalibrationLUTESSource);
