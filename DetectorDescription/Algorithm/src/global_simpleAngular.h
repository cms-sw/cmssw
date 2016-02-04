
// FILE IS GENERATED. DON'T MODIFY!
// source: global.xml
#ifndef DDAlgorithm_global_simpleAngular_h
#define DDAlgorithm_global_simpleAngular_h

#include "DetectorDescription/ExprAlgo/interface/AlgoImpl.h"
//#include "CLHEP/Units/GlobalSystemOfUnits.h"
// I tried the following on CERN lxplus and still no MAX_DOUBLE was defined.
// so I tried DBL_MAX which does exist, but I do not know the source of this.
// So, in keeping with everything else I saw:
#include <cfloat>
#include <string>
#include <iostream>


#define MAX_DOUBLE DBL_MAX
//#include <climits>


/**

   (...THIS COULD BE DOXYGEN CONFORMANT ...)
   Angular positioning according to the formula:

      offset ... an angle offset from which to start the positioning

      radius ... an offset distance in the radial direction.
    
      delta  ... angular delta between each position.

      number ... number of times to position within the range
      rotate ... boolean text, T or F, True or False, 1 or zero.  Use this
                 to indicate if the self logical part should be rotated.
                 Since the DDD uses double or std::string, I'm using std::string.
                 default = "T"
      orientation ... std::string: rotation matrix name to be used to orient the
                      part within the mother.  This is a DDLrRotation.

      The algorithm has (at least) two implementations.
      
      The first is selected if delta is specified and number is NOT specified.
         In this algorithm, the positioning is relatively straight-forward.
         
      The second is selected if number is specified and delta is NOT specified.
         This should use the first one by specifying a delta based on dividing
	 up 360 degrees by the number of positionings desired.  For this one, 
	 CheckTermination is used to determine when to stop making them.

      Some methods are only specific to this algorithm (I HOPE)

*/



//number no delta// each algorithm implementation is suffixed by _n, n=0,1,2,...
class global_simpleAngular_0 : public AlgoImpl
{
public:
  global_simpleAngular_0(AlgoPos*,std::string label);
  ~global_simpleAngular_0();
  
  bool checkParameters();
  
  DDTranslation translation();
  
  DDRotationMatrix  rotation();
  
  

  void checkTermination();
// no XML counterpart, but usefull.
  void stream(std::ostream  &) const;

};

//delta no number// each algorithm implementation is suffixed by _n, n=0,1,2,...
class global_simpleAngular_1 : public AlgoImpl
{
public:
  global_simpleAngular_1(AlgoPos*,std::string label);
  ~global_simpleAngular_1();
  
  bool checkParameters();
  
  DDTranslation translation();
  
  DDRotationMatrix  rotation();
  
  

  void checkTermination();
// no XML counterpart, but usefull.
  void stream(std::ostream  &) const;

};

//delta AND number// each algorithm implementation is suffixed by _n, n=0,1,2,...
class global_simpleAngular_2 : public AlgoImpl
{
public:
  global_simpleAngular_2(AlgoPos*,std::string label);
  ~global_simpleAngular_2();
  
  bool checkParameters();
  
  DDTranslation translation();
  
  DDRotationMatrix  rotation();
  
  

  void checkTermination();
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

class global_simpleAngular_Check : public AlgoCheck
{
public:
  // only ctor-code is generated!
  global_simpleAngular_Check();
  ~global_simpleAngular_Check() { };
};

#endif


