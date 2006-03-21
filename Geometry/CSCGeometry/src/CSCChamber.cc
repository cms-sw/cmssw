/** \file
 *
 *  $Date: $
 *  $Revision: $
 */

#include <Geometry/CSCGeometry/interface/CSCChamber.h>


CSCChamber::~CSCChamber(){
  // Delete all layers
  for (std::vector<const GeomDet*>::const_iterator i=theComponents.begin();
  i!=theComponents.end(); ++i){
    delete (*i);
  }
}

