#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"


DetId MiscalibReaderFromXMLEcalEndcap::parseCellEntry(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute)
{

EEDetId cell= MiscalibReaderFromXMLEcalEndcap::getCellFromAttributes(
                                            getIntAttribute(attribute,"module_index"),
                                            getIntAttribute(attribute,"crystal_index"),
                                            getIntAttribute(attribute,"z_index")
                                            );
return cell;
}

EEDetId MiscalibReaderFromXMLEcalEndcap::getCellFromAttributes(int isc, int ic, int iz)
{
EEDetId cell(isc,ic,iz,1);
return cell;
}

