/** \file
 *
 *  $Date: 2006/07/14 14:37:20 $
 *  $Revision: 1.4 $
 */

#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>


CSCChamber::~CSCChamber(){
  // Delete all layers
  for (std::vector<const CSCLayer*>::const_iterator i=theComponents.begin();
       i!=theComponents.end(); ++i){
    delete (*i);
  }
}


std::vector<const GeomDet*> 
CSCChamber::components() const {
  return std::vector <const GeomDet*>(theComponents.begin(),theComponents.end());
}


const GeomDet* CSCChamber::component(DetId id) const {
  return layer(CSCDetId(id.rawId()));
}


void CSCChamber::addComponent( int n, const CSCLayer* gd ) { 
	
  if ((n>0) && (n<7)) 
    theComponents[n-1] = gd; 
  else 
    edm::LogError("CSC") << "Each chamber has only SIX layers.";
}

const CSCLayer* CSCChamber::layer(CSCDetId iid) const {
  if (iid.chamberId()!=id()) return 0; // not in this chamber
  return layer(iid.layer());
}
  
const CSCLayer* CSCChamber::layer(int ilay) const{
  	
  if ((ilay>0) && (ilay<7)) 
    return theComponents[ilay-1];
  else {
    return 0;
  }
}
