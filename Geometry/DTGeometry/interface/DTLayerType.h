#ifndef DTLAYERTYPE_H
#define DTLAYERTYPE_H

/** \class DTLayerType
 *
 *  DetType for a Drift Tube GeomDetUnit (the DTLayer).
 *
 *  $date   : 23/01/2006 18:24:59 CET $
 *  $Revision: 1.2 $
 *  \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

class DTLayerType : public GeomDetType {

  public:

/* Constructor */ 
    DTLayerType() ;

/* Operations */ 
    virtual const Topology& topology() const;

  private:

};
#endif // DTLAYERTYPE_H

