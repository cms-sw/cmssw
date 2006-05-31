#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalBarrel.h"



DetId MiscalibReaderFromXMLEcalBarrel::parseCellEntry(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute)
{

EBDetId cell= MiscalibReaderFromXMLEcalBarrel::getCellFromAttributes(
                                            getIntAttribute(attribute,"eta_index"),
                                            getIntAttribute(attribute,"phi_index")
                                            );
return cell;
}

EBDetId MiscalibReaderFromXMLEcalBarrel::getCellFromAttributes(int ieta, int iphi)
{
EBDetId cell(ieta,iphi);
return cell;
}

