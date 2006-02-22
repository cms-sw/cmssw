#ifndef DTLAYERTYPE_H
#define DTLAYERTYPE_H

/** \class DTLayerType
 *
 *  DetType for a Drift Tube GeomDetUnit (the DTLayer).
 *
 *  $date   : 23/01/2006 18:24:59 CET $
 *  $Revision: 1.3 $
 *  \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

/* Collaborating Class Declarations */

/* C++ Headers */

/* ====================================================================== */

/* Class DTLayerType Interface */

class DTLayerType : public GeomDetType {

  public:

/* Constructor */ 
    DTLayerType() ;

/* Destructor */ 

/* Operations */ 
    virtual const Topology& topology() const;

  private:

  protected:

};
#endif // DTLAYERTYPE_H

