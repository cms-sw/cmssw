
// FILE IS GENERATED. DON'T MODIFY!
// source: global.xml
#ifndef DDAlgorithm_global_angular_h
#define DDAlgorithm_global_angular_h

#include "DetectorDescription/ExprAlgo/interface/AlgoImpl.h"
//#include "CLHEP/Units/GlobalSystemOfUnits.h"
// I tried the following on CERN lxplus and still no MAX_DOUBLE was defined.
// so I tried DBL_MAX which does exist, but I do not know the source of this.
// So, in keeping with everything else I saw:
#define MAX_DOUBLE DBL_MAX
//#include <climits>
//#include <cfloat>

#include <iostream>
#include <vector>
#include <string>

/**

	     theta, phi, angle ... axis of rotation. Object will be placed in a plane going through
	                    the origin having the axis (theta,phi) perpendicular to itself.
	     startAngle, rangeAngle ... position rotated object starting at startAngle going to 
	                                startAngle+rangeAngle.
					startAngle = 0 denotes the x-axis in the local frame in
					               case of (phi=0 and theta=0), otherwise it
						       denotes the 
					
	   
*/



//divrange// each algorithm implementation is suffixed by _n, n=0,1,2,...
class global_angular_0 : public AlgoImpl
{
public:
  global_angular_0(AlgoPos*,std::string label);
  ~global_angular_0();
  
  bool checkParameters();
  
  DDTranslation translation();
  
  DDRotationMatrix  rotation();
  
  int copyno() const;
  void checkTermination();
// no XML counterpart, but usefull.
  void stream(std::ostream  &) const;
  
  // parameters according to XML-attribute 'name' in <Parameter> tags 
  std::vector<double> rotate_, center_, rotateSolid_;
  bool alignSolid_;
  int n_,startCopyNo_,incrCopyNo_;
  double startAngle_, rangeAngle_, radius_;
  double delta_; // angular distance between two placements
  DDRotationMatrix solidRot_; // rotation of the solid
  DDRotationMatrix planeRot_; // rotation of the plane
};


/**************************************************************************
 
 The following Code gets only generated IF the code-generator
 has the capability of generating evaluation/checking code for
 the specification of the parameters in the algorithm XML.
 
 If the following is not generated, there will be not automatic
 checking of the basic properties of the user-supplied parameters
 (bounds, occurences, default values, ...)
 
***************************************************************************/

#include "DetectorDescription/ExprAlgo/interface/AlgoCheck.h"

class global_angular_Check : public AlgoCheck
{
public:
  // only ctor-code is generated!
  global_angular_Check();
  ~global_angular_Check() { };
};

#endif


