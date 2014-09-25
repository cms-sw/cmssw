#include "RecoTauTag/TauTagTools/interface/ECALBounds.h"
namespace {
  constexpr float epsilon = 0.001; // should not matter at all
  const Surface::RotationType rot; // unit rotation matrix
}


const BoundCylinder  ECALBounds::theCylinder(ECALBounds::barrel_innerradius(), Surface::PositionType(0,0,0),rot, 
				 new SimpleCylinderBounds(ECALBounds::barrel_innerradius()-epsilon, 
						       	  ECALBounds::barrel_innerradius()+epsilon, 
							 -ECALBounds::barrel_halfLength(), 
							  ECALBounds::barrel_halfLength()));


const BoundDisk  ECALBounds::theNegativeDisk(Surface::PositionType(0,0,-ECALBounds::endcap_innerZ()),rot, 
		   new SimpleDiskBounds(0,ECALBounds::endcap_outerradius(),-epsilon,epsilon));

const BoundDisk  ECALBounds::thePositiveDisk( Surface::PositionType(0,0,ECALBounds::endcap_innerZ()),rot, 
		   new SimpleDiskBounds(0,ECALBounds::endcap_outerradius(),-epsilon,epsilon));
