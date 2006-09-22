#ifndef _MISCALIB_READER_FROM_XML_ECAL_ENDCAP_H
#define _MISCALIB_READER_FROM_XML_ECAL_ENDCAP_H

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXML.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"



class MiscalibReaderFromXMLEcalEndcap : public MiscalibReaderFromXML
{
 public:
  MiscalibReaderFromXMLEcalEndcap(CaloMiscalibMapEcal & map):MiscalibReaderFromXML(map){};

  virtual DetId parseCellEntry(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute);

  EEDetId getCellFromAttributes(int ix, int iy, int iz);

};

#endif

