
///////////////////////////////////////////////////////////////////////////
// GENERATED FILE, DO NOT MODIFY
///////////////////////////////////////////////////////////////////////////


#include "DetectorDescription/Algorithm/src/AlgoInit.h"
#include "DetectorDescription/Core/interface/DDAlgo.h"
#include "DetectorDescription/ExprAlgo/interface/AlgoPos.h"
// include all from XML generated algorithms


#include "global_linear.h"


#include "global_angular.h"


#include "global_simpleAngular.h"


#include "DetectorDescription/Algorithm/src/presh_detectors.h"



// add a line for each generated algorithm to register the
// algorithm with DDCore
void AlgoInit()
{
  static bool isInit = false;
  if (isInit) return;
  isInit = true;
  
  // placeholder pointer
  AlgoPos * algo = 0;
  
  // generated:
  //   for each <Algorithm> in each ADL-instance the code generator has to provide code
  //   similar to that below ...
   
  //   create the representation of the algorithm  


  algo = new AlgoPos(new global_linear_Check);
  //   create the implementations of the algorithm & register them with the representation


  new global_linear_0(algo,"base in the input parameters");


  new global_linear_1(algo,"without base in input parameters");


  //   register the representation at DDCore
  DDalgo(DDName("linear", "global"), algo);


  algo = new AlgoPos(new global_angular_Check);
  //   create the implementations of the algorithm & register them with the representation
  

  new global_angular_0(algo,"divrange");


  //   register the representation at DDCore
  DDalgo(DDName("angular", "global"), algo);


  algo = new AlgoPos(new global_simpleAngular_Check);
  //   create the implementations of the algorithm & register them with the representation


  new global_simpleAngular_0(algo,"number no delta");


  new global_simpleAngular_1(algo,"delta no number");


  new global_simpleAngular_2(algo,"delta AND number");


  //   register the representation at DDCore
  DDalgo(DDName("simpleAngular", "global"), algo);

}


