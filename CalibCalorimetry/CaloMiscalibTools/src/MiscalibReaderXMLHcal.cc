#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLHcal.h"



DetId MiscalibReaderFromXMLHcal::parseCellEntry(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute)
{

HcalDetId cell= MiscalibReaderFromXMLHcal::getCellFromAttributes(
                                            getIntAttribute(attribute,"det_index"),
                                            getIntAttribute(attribute,"eta_index"),
                                            getIntAttribute(attribute,"phi_index"),
                                            getIntAttribute(attribute,"depth_index")
                                            );
return cell;
}

HcalDetId MiscalibReaderFromXMLHcal::getCellFromAttributes(int idet, int ieta, int iphi, int idepth)
{
  try
    {
      HcalDetId cell((HcalSubdetector) idet, ieta, iphi, idepth);
      return cell;
    }
  catch (...)
    {
      std::cout << "Null coordinates = "<< idet << "," << ieta << "," << iphi << "," << idepth << std::endl;
      return HcalDetId(0);
    }
}

