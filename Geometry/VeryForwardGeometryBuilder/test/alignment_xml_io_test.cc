#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"
#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsDataSequence.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/RPAlignmentCorrectionsMethods.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

int main()
{
  // build sample alignment data
  RPAlignmentCorrectionsData ad;

  ad.addRPCorrection(TotemRPDetId(1, 0, 3), RPAlignmentCorrectionData(1., -2., 0., 0., 0., 3e-3));                  // silicon RP
  ad.addSensorCorrection(TotemRPDetId(1, 0, 3, 2), RPAlignmentCorrectionData(-1., +0.5, 0., 0., 0., -0.2e-3));      // silicon plane

  ad.addRPCorrection(CTPPSPixelDetId(1, 2, 3), RPAlignmentCorrectionData(1., -2., 0., 0., 0., 3e-3));               // pixel RP
  ad.addSensorCorrection(CTPPSPixelDetId(1, 2, 3, 1), RPAlignmentCorrectionData(-1., +0.5, 0., 0., 0., -0.2e-3));   // pixel plane

  ad.addRPCorrection(CTPPSDiamondDetId(1, 2, 4), RPAlignmentCorrectionData(1., -2., 0., 0., 0., 3.));               // diamond RP
  ad.addSensorCorrection(CTPPSDiamondDetId(1, 2, 4, 3), RPAlignmentCorrectionData(-1., +0.5, 0., 0., 0., -0.2e-3)); // diamond plane

  ad.addRPCorrection(TotemRPDetId(0, 0, 2), RPAlignmentCorrectionData(1., -2., 0., 0., 0., 3e-3));                  // silicon RPs with no sensor corrections
  ad.addRPCorrection(TotemRPDetId(0, 0, 3), RPAlignmentCorrectionData(1., -2., 0., 0., 0., 3e-3));

  // write alignment data into XML file
  // TODO write all elements
  RPAlignmentCorrectionsMethods::writeToXML(ad, "alignment_xml_io_test.xml");

  // load alignment data from XML file
  const RPAlignmentCorrectionsDataSequence &adsl = RPAlignmentCorrectionsMethods::loadFromXML("alignment_xml_io_test.xml");

  // there should be exactly one element in the sequence
  if (adsl.size() != 1)
    return 1;

  // compare build and loaded data for 1 sensor
  unsigned int id = TotemRPDetId(1, 0, 3, 2);
  const RPAlignmentCorrectionData &a = ad.getSensorCorrection(id);
  const RPAlignmentCorrectionData &al = adsl.begin()->second.getSensorCorrection(id);

  printf("a:\n");
  a.print();
  printf("al:\n");
  al.print();

  if (a.getShX() != al.getShX()) return 2;
  if (a.getShY() != al.getShY()) return 2;
  if (a.getShZ() != al.getShZ()) return 2;

  if (a.getRotX() != al.getRotX()) return 2;
  if (a.getRotY() != al.getRotY()) return 2;
  if (a.getRotZ() != al.getRotZ()) return 2;

  return 0;
}
