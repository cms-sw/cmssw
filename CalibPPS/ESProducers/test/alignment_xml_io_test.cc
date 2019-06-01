#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSRPAlignmentCorrectionsDataSequence.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSRPAlignmentCorrectionsMethods.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

//----------------------------------------------------------------------------------------------------

int CompareCorrections(const CTPPSRPAlignmentCorrectionData &c1, const CTPPSRPAlignmentCorrectionData c2) {
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

int main() {
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

  // load alignment data from XML file
  const CTPPSRPAlignmentCorrectionsDataSequence &adsl =
      CTPPSRPAlignmentCorrectionsMethods::loadFromXML("alignment_xml_io_test.xml");

  // there should be exactly one element in the sequence
  if (adsl.size() != 1)
    return 1;

  // check loaded iov
  const auto &iovl = adsl.begin()->first;
  if (iovl.first() != edm::IOVSyncValue::beginOfTime() || iovl.last().eventID().run() != event_end.run() ||
      iovl.last().eventID().luminosityBlock() != event_end.luminosityBlock())
    return 2;

  // compare build and loaded data for 1 RP
  {
    unsigned int id = TotemRPDetId(1, 0, 3);
    const CTPPSRPAlignmentCorrectionData &a = ad.getRPCorrection(id);
    const CTPPSRPAlignmentCorrectionData &al = adsl.begin()->second.getRPCorrection(id);

    if (CompareCorrections(a, al) != 0)
      return 3;
  }

  // compare build and loaded data for 1 sensor
  {
    unsigned int id = TotemRPDetId(1, 0, 3, 2);
    const CTPPSRPAlignmentCorrectionData &a = ad.getSensorCorrection(id);
    const CTPPSRPAlignmentCorrectionData &al = adsl.begin()->second.getSensorCorrection(id);

    if (CompareCorrections(a, al) != 0)
      return 4;
  }

  return 0;
}