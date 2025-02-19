#ifndef CastorCollections_h
#define CastorCollections_h

/** \class CastorCollections
 *
 * CastorCollections 
 *
 * \author Alan Campbell   
 *
 * \version   1st Version April 18, 2008  
 *
 ************************************************************/


#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

class CastorCollections 
{
public:

   CastorCollections();
   ~CastorCollections();
    const CastorDigiCollection* castorCont;
    const HcalCalibDigiCollection* calibCont;
    const HcalTrigPrimDigiCollection* tpCont;
};

#endif
