/** \file
 *
 *  $Date: 2006/03/21 16:50:53 $
 *  $Revision: 1.1 $
 */

#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>


CSCChamber::~CSCChamber(){
  // Delete all layers
  for (std::vector<const GeomDet*>::const_iterator i=theComponents.begin();
       i!=theComponents.end(); ++i){
    delete (*i);
  }
}


void CSCChamber::addComponent( int n, const GeomDet* gd ) { 
	
  if ((n>0) && (n<7)) 
    theComponents[n-1] = gd; 
  else 
    edm::LogError("CSC") << "Each chamber has only SIX layers.";
}

const CSCLayer* CSCChamber::layer(CSCDetId id) const {
  return layer(id.layer());
}
  
const CSCLayer* CSCChamber::layer(int ilay) const{
  	
  if ((ilay>0) && (ilay<7)) 
    return dynamic_cast<const CSCLayer*> (theComponents[ilay-1]);
  else {
    return 0;
  }
}
