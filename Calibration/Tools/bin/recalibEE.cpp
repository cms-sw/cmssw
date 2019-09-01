#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibTools.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalBarrel.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TFile.h"

int main() {
  calibXMLwriter endcapWriter(EcalEndcap);
  CaloMiscalibMapEcal map;
  std::string endcapfile =
      "/afs/cern.ch/user/p/presotto/CMSSW_1_6_0/src/CalibCalorimetry/CaloMiscalibTools/data/ecal_endcap_startup.xml";
  map.prefillMap();
  MiscalibReaderFromXMLEcalEndcap endcapreader(map);
  if (!endcapfile.empty())
    endcapreader.parseXMLMiscalibFile(endcapfile);

  EcalIntercalibConstants* constants = new EcalIntercalibConstants(map.get());
  const EcalIntercalibConstantMap& imap = constants->getMap();

  std::string endcapfile2 = "EEcalib.xml";
  CaloMiscalibMapEcal map2;
  map2.prefillMap();
  MiscalibReaderFromXMLEcalEndcap endcapreader2(map2);
  if (!endcapfile2.empty())
    endcapreader2.parseXMLMiscalibFile(endcapfile2);
  EcalIntercalibConstants* constants2 = new EcalIntercalibConstants(map2.get());
  const EcalIntercalibConstantMap& imap2 = constants2->getMap();
  for (int x = 1; x <= 100; ++x)
    for (int y = 1; y < 100; ++y) {
      if (!EEDetId::validDetId(x, y, -1))
        continue;
      EEDetId ee(x, y, -1, EEDetId::XYMODE);
      endcapWriter.writeLine(ee, *(imap.find(ee.rawId())) * *(imap2.find(ee.rawId())));
      if (!EEDetId::validDetId(x, y, 1))
        continue;
      EEDetId e2(x, y, 1, EEDetId::XYMODE);
      endcapWriter.writeLine(e2, *(imap.find(e2.rawId())) * *(imap2.find(e2.rawId())));
    }
}
