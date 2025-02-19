#ifndef CastorRawCollections_h
#define CastorRawCollections_h

/** \class CastorRawCollections
 *
 * CastorRawCollections 
 *
 * \author Alan Campbell     
 *
 * \version   1st Version April 18, 2008  
 *
 ************************************************************/


#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

class CastorRawCollections 
{
public:

   CastorRawCollections();
   ~CastorRawCollections();
    std::vector<CastorDataFrame>* castorCont;
    std::vector<HcalCalibDataFrame>* calibCont;
    std::vector<CastorTriggerPrimitiveDigi>* tpCont;
	std::vector<HcalTTPDigi>* ttp;

};

#endif
