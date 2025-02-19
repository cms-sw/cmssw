
// FILE IS GENERATED. DON'T MODIFY!
// source: global.xml
#ifndef DDAlgorithm_global_linear_h
#define DDAlgorithm_global_linear_h

#include "DetectorDescription/ExprAlgo/interface/AlgoImpl.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
// I tried the following on CERN lxplus and still no MAX_DOUBLE was defined.
// so I tried DBL_MAX which does exist, but I do not know the source of this.
// So, in keeping with everything else I saw:
#include <cfloat>
#include <string>
#include <iostream>


#define MAX_DOUBLE DBL_MAX
//#include <climits>


/**

	    Linear positioning according to the formula:
	     c = {s, s+i, s+2*i, .. <=e} ... current number in alg.range
	     dir[theta,phi] ... unit-std::vector in direction theta,phi
	     offset ... an offset distance in direction dir(theta,phi)
	     delta ... distance between two subsequent positions along dir[theta,phi]
	     base ... a 3d-point where the offset is calculated from
	              base is optional, if omitted base=(0,0,0)
	     3Dpoint(c) = base + (offset + c*delta)*dir[theta,phi]
	     
	     The algorithm will have 2 implementations:
	     The first will be selected if base is not present in the input parameters,
	     the second will be selected if base is present in the input parameters.
	     (This is just an example and 2 implementations are not really
	      necessary. They are only here to demonstrate the capability of
	      having mulitiple implementations of the same algorithm)
	     
*/



//base in the input parameters// each algorithm implementation is suffixed by _n, n=0,1,2,...
class global_linear_0 : public AlgoImpl
{
public:
  global_linear_0(AlgoPos*,std::string label);
  ~global_linear_0();
  
  bool checkParameters();
  
  DDTranslation translation();
  
  DDRotationMatrix  rotation();
  
  

  

// no XML counterpart, but usefull.
  void stream(std::ostream  &) const;

};

//without base in input parameters// each algorithm implementation is suffixed by _n, n=0,1,2,...
class global_linear_1 : public AlgoImpl
{
public:
  global_linear_1(AlgoPos*,std::string label);
  ~global_linear_1();
  
  bool checkParameters();
  
  DDTranslation translation();
  
  DDRotationMatrix  rotation();
  
  
// no XML counterpart, but usefull.
  void stream(std::ostream  &) const;

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

class global_linear_Check : public AlgoCheck
{
public:
  // only ctor-code is generated!
  global_linear_Check();
  ~global_linear_Check() { }
};

#endif


