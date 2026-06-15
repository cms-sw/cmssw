#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsDataSequence.h"

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsMethods.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

#include "CalibPPS/ESProducers/interface/CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon.h"

#include <catch2/catch_all.hpp>

//----------------------------------------------------------------------------------------------------

int CompareCorrections(const CTPPSRPAlignmentCorrectionData& c1, const CTPPSRPAlignmentCorrectionData c2) {
  if (c1.getShX() != c2.getShX())
    return 2;
  if (c1.getShY() != c2.getShY())
    return 2;
  if (c1.getShZ() != c2.getShZ())
    return 2;

  if (c1.getRotX() != c2.getRotX())
    return 2;
  if (c1.getRotY() != c2.getRotY())
    return 2;
  if (c1.getRotZ() != c2.getRotZ())
    return 2;

  return 0;
}

//----------------------------------------------------------------------------------------------------

TEST_CASE("CTPPSRPAlignmentCorrectionsDataSequence", "[CTPPSRPAlignmentCorrectionsDataSequence]") {
  // build sample alignment data
  CTPPSRPAlignmentCorrectionsData ad;

  ad.addRPCorrection(TotemRPDetId(1, 0, 3),
                     CTPPSRPAlignmentCorrectionData(1., 2., 3., 1e-3, 2e-3, 3e-3));  // silicon RP
  ad.addSensorCorrection(TotemRPDetId(1, 0, 3, 2),
                         CTPPSRPAlignmentCorrectionData(4., 5., 6., 4e-3, 5e-3, 6e-3));  // silicon plane

  ad.addRPCorrection(CTPPSPixelDetId(1, 2, 3), CTPPSRPAlignmentCorrectionData(1., -2., 0., 0., 0., 3e-3));  // pixel RP
  ad.addSensorCorrection(CTPPSPixelDetId(1, 2, 3, 1),
                         CTPPSRPAlignmentCorrectionData(-1., +0.5, 0., 0., 0., -0.2e-3));  // pixel plane

  ad.addRPCorrection(CTPPSDiamondDetId(1, 2, 4),
                     CTPPSRPAlignmentCorrectionData(1., -2., 0., 0., 0., 3.));  // diamond RP
  ad.addSensorCorrection(CTPPSDiamondDetId(1, 2, 4, 3),
                         CTPPSRPAlignmentCorrectionData(-1., +0.5, 0., 0., 0., -0.2e-3));  // diamond plane

  ad.addRPCorrection(
      TotemRPDetId(0, 0, 2),
      CTPPSRPAlignmentCorrectionData(1., -2., 0., 0., 0., 3e-3));  // silicon RPs with no sensor corrections
  ad.addRPCorrection(TotemRPDetId(0, 0, 3), CTPPSRPAlignmentCorrectionData(1., -2., 0., 0., 0., 3e-3));

  // prepare sequence
  CTPPSRPAlignmentCorrectionsDataSequence ads;
  edm::EventID event_end(123, 456, 1);
  ads.insert(edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue(event_end)), ad);

  // write alignment data into XML file
  CTPPSRPAlignmentCorrectionsMethods::writeToXML(
      ads, "alignment_xml_io_test.xml", false, false, true, true, true, true);

  SECTION("from XML") {
    // load alignment data from XML file
    const CTPPSRPAlignmentCorrectionsDataSequence& adsl =
        CTPPSRPAlignmentCorrectionsMethods::loadFromXML("alignment_xml_io_test.xml");

    // there should be exactly one element in the sequence
    REQUIRE(adsl.size() == 1);

    SECTION("check loaded iov") {
      const auto& iovl = adsl.begin()->first;
      REQUIRE(iovl.first() == edm::IOVSyncValue::beginOfTime());
      REQUIRE(iovl.last().eventID().run() == event_end.run());
      REQUIRE(iovl.last().eventID().luminosityBlock() == event_end.luminosityBlock());
    }
    SECTION("compare build and loaded data for 1 RP") {
      unsigned int id = TotemRPDetId(1, 0, 3);
      const CTPPSRPAlignmentCorrectionData& a = ad.getRPCorrection(id);
      const CTPPSRPAlignmentCorrectionData& al = adsl.begin()->second.getRPCorrection(id);

      REQUIRE(CompareCorrections(a, al) == 0);
    }

    SECTION("compare build and loaded data for 1 sensor") {
      unsigned int id = TotemRPDetId(1, 0, 3, 2);
      const CTPPSRPAlignmentCorrectionData& a = ad.getSensorCorrection(id);
      const CTPPSRPAlignmentCorrectionData& al = adsl.begin()->second.getSensorCorrection(id);

      REQUIRE(CompareCorrections(a, al) == 0);
    }
  }
}

namespace edm {
  //These are helpful if a REQUIRE clause fails and it prints the values
  std::ostream& operator<<(std::ostream& ost, edm::IOVSyncValue const& sv) {
    ost << " { " << sv.eventID() << " } ";
    return ost;
  }
  std::ostream& operator<<(std::ostream& ost, edm::ValidityInterval const& iov) {
    ost << "( " << iov.first() << ", " << iov.last() << " )";
    return ost;
  }
}  // namespace edm

TEST_CASE("CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon", "[CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon]") {
  CTPPSRPAlignmentCorrectionsData ad;

  ad.addRPCorrection(TotemRPDetId(1, 0, 3),
                     CTPPSRPAlignmentCorrectionData(1., 2., 3., 1e-3, 2e-3, 3e-3));  // silicon RP
  ad.addSensorCorrection(TotemRPDetId(1, 0, 3, 2),
                         CTPPSRPAlignmentCorrectionData(4., 5., 6., 4e-3, 5e-3, 6e-3));  // silicon plane

  auto iov = [](edm::EventID const& begin, edm::EventID const& end) {
    return edm::ValidityInterval{edm::IOVSyncValue{begin}, edm::IOVSyncValue{end}};
  };

  CTPPSRPAlignmentCorrectionsDataSequence ads;
  ads.insert(iov({2, 1, 0}, {2, 100, 0}), ad);
  ads.insert(iov({3, 1, 0}, {3, 100, 0}), ad);

  using common = CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon;
  SECTION("Test IOV handling") {
    REQUIRE(edm::ValidityInterval{edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue{edm::EventID{2, 0, 0}}} ==
            common::intervalFor(edm::IOVSyncValue::beginOfTime(), ads, false));
    REQUIRE(iov({2, 1, 0}, {2, 100, 0}) == common::intervalFor(edm::IOVSyncValue{edm::EventID{2, 1, 0}}, ads, false));
    REQUIRE(iov({2, 1, 0}, {2, 100, 0}) == common::intervalFor(edm::IOVSyncValue{edm::EventID{2, 20, 0}}, ads, false));
    REQUIRE(iov({2, 200, 0}, {3, 0, 0}) == common::intervalFor(edm::IOVSyncValue{edm::EventID{2, 200, 0}}, ads, false));
    REQUIRE(iov({3, 1, 0}, {3, 100, 0}) == common::intervalFor(edm::IOVSyncValue{edm::EventID{3, 1, 0}}, ads, false));
    REQUIRE(edm::ValidityInterval{edm::IOVSyncValue::endOfTime(), edm::IOVSyncValue::endOfTime()} ==
            common::intervalFor(edm::IOVSyncValue::endOfTime(), ads, false));
  }
  SECTION("Test data IOV handling") {
    REQUIRE(nullptr == common::dataFor(edm::IOVSyncValue::beginOfTime(), ads));
    REQUIRE(nullptr != common::dataFor(edm::IOVSyncValue{edm::EventID{2, 1, 0}}, ads));
    REQUIRE(nullptr != common::dataFor(edm::IOVSyncValue{edm::EventID{2, 20, 0}}, ads));
    REQUIRE(nullptr == common::dataFor(edm::IOVSyncValue{edm::EventID{2, 200, 0}}, ads));
    REQUIRE(nullptr != common::dataFor(edm::IOVSyncValue{edm::EventID{3, 1, 0}}, ads));
    REQUIRE(nullptr == common::dataFor(edm::IOVSyncValue::endOfTime(), ads));
  }
}
