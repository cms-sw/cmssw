#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"


DetId MiscalibReaderFromXMLEcalEndcap::parseCellEntry(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attribute)
{

EEDetId cell= MiscalibReaderFromXMLEcalEndcap::getCellFromAttributes(
                                            getIntAttribute(attribute,"x_index"),
                                            getIntAttribute(attribute,"y_index"),
                                            getIntAttribute(attribute,"z_index")
                                            );
return cell;
}

EEDetId MiscalibReaderFromXMLEcalEndcap::getCellFromAttributes(int ix, int iy, int iz)
{

       try 
         {
	   if (EEDetId::validDetId(ix, iy, iz)) {
	     EEDetId cell(ix,iy,iz);
	     return cell;
	   } else {
	     return (EEDetId) NULL;
	   }
         }
    
           catch (...)
	  
        {
          std::cout << "Null coordinates = "<< ix << "," << iy << "," << iz << std::endl;
	  return (EEDetId) NULL;
        }
	    
	    
}

