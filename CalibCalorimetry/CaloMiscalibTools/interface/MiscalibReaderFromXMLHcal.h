#ifndef _MISCALIB_READER_FROM_XML_HCAL_BARREL_H
#define _MISCALIB_READER_FROM_XML_HCAL_BARREL_H

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXML.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapHcal.h"

class MiscalibReaderFromXMLHcal : public MiscalibReaderFromXML {
public:
  MiscalibReaderFromXMLHcal(CaloMiscalibMapHcal &map) : MiscalibReaderFromXML(map){};

  DetId parseCellEntry(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute) override;

  HcalDetId getCellFromAttributes(int idet, int ieta, int iphi, int idepth);
};

#endif
