#ifndef DD_TrackerRingAlgo_h
#define DD_TrackerRingAlgo_h

/*
  Tilts and positions n copies of a module at prescribed phi values
  within a ring. The module can also be flipped if requested.

  (radius, Phi, Z) refers to the cylindrical coordinates in the global frame of reference.
  
  A module's tilt angle is defined with respect to the global frame of reference's Z axis.
  Example, in the outer tracker : For a straight barrel module, tiltAngle = 0°.
  For a module in the endcaps, tiltAngle = 90°.
  tiltAngle ∈ [0, 90°].
  Please note that parameter tiltAngle has to be set regardless of any sign consideration,
  to the absolute value of the module's tilt angle.

  == Example of use : ==

  <Algorithm name="track:DDTrackerRingAlgo">
  <rParent name="tracker:Ring5Layer1Plus"/>
  <String name="ChildName" value="tracker:BModule5Layer1"/>
  <Numeric name="N" value="9"/>
  <Numeric name="StartCopyNo" value="1"/>
  <Numeric name="IncrCopyNo" value="2"/>
  <Numeric name="RangeAngle" value="360*deg"/>
  <Numeric name="StartAngle" value="90*deg"/>
  <Numeric name="Radius" value="247"/>
  <Vector name="Center" type="numeric" nEntries="3">0,0,-5.45415</Vector>
  <Numeric name="IsZPlus" value="1"/>
  <Numeric name="TiltAngle" value="47*deg"/>
  <Numeric name="IsFlipped" value="1"/>
  </Algorithm>
*/

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTrackerRingAlgo : public DDAlgorithm {
 
public:
  // Constructor and Destructor
  DDTrackerRingAlgo(); 
  virtual ~DDTrackerRingAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  int           n;              //Number of copies
  int           startCopyNo;    //Start Copy number
  int           incrCopyNo;     //Increment in Copy number
  double        rangeAngle;     //Range in Phi angle
  double        startAngle;     //Start Phi angle
  double        radius;         //Radius
  std::vector<double> center;   //Phi values
  bool          isZPlus;        //Is Z positive ?
  double        tiltAngle;      //Module's tilt angle (absolute value)
  bool          isFlipped;      //Is the module flipped ?
  double        delta;          //Increment in Phi

  std::string   idNameSpace;    //Namespace of this and ALL sub-parts
  std::string   childName;      //Child name
};

#endif
