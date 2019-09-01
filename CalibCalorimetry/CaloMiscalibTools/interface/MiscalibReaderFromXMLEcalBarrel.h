#ifndef _MISCALIB_READER_FROM_XML_ECAL_BARREL_H
#define _MISCALIB_READER_FROM_XML_ECAL_BARREL_H

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXML.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"

class MiscalibReaderFromXMLEcalBarrel : public MiscalibReaderFromXML {
public:
  MiscalibReaderFromXMLEcalBarrel(CaloMiscalibMapEcal &map) : MiscalibReaderFromXML(map){};

  DetId parseCellEntry(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute) override;

  EBDetId getCellFromAttributes(int ieta, int iphi);
};

#endif
