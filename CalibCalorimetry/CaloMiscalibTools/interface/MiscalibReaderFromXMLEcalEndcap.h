#ifndef _MISCALIB_READER_FROM_XML_ECAL_ENDCAP_H
#define _MISCALIB_READER_FROM_XML_ECAL_ENDCAP_H

#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXML.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

class MiscalibReaderFromXMLEcalEndcap : public MiscalibReaderFromXML {
public:
  MiscalibReaderFromXMLEcalEndcap(CaloMiscalibMapEcal &map)
      : MiscalibReaderFromXML(map){};

  DetId
  parseCellEntry(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute) override;

  EEDetId getCellFromAttributes(int ix, int iy, int iz);
};

#endif
