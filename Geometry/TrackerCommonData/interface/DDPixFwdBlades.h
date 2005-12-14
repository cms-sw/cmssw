#ifndef DDPixFwdBlades_h
#define DDPixFwdBlades_h

/* 

== CMS Forward Pixels Geometry ==

 @version 2.01.01 Dec 06, 2005
 @created Dmitry Onoprienko

== ALGORITHM DESCRIPTION: ==

  Algorithm for placing one-per-blade components

== Example of use : ==

<Algorithm name="track:DDPixFwdBlades">
  <rParent name="pixfwdDisk:PixelForwardDiskZMinus"/>
  <Numeric name="BladeAngle"    value="-[BladeAngle]" />
  <Numeric name="ZPlane"        value="0." />
  <Numeric name="BladeZShift"   value="-[BladeZShift]" />
  <String  name="Child"         value="pixfwdPanel:PixelForwardPanel3Left"/>
  <Numeric name="ChildX"        value="[radiusPanel]"/>
  <Numeric name="ChildY"        value="0."/>
  <Numeric name="ChildZ"        value="-[zPanel]"/>
  <String  name="ChildRotation" value="pixfwdDisk:Y180"/>
  <String  name="FlagString"     value="LRRRRLRRRRRLLRRRRLRRRRRL" />
  <String  name="FlagSelector"  value="L" />
</Algorithm>

== Parameters : ==

  "BladeAngle" - angle of blade rotation around its axis (20 degrees in current PixelForward design)
  "ZPlane" - Z of of the disk central plane in the parent volume reference frame
  "BladeZShift" - shift in Z between adjacent blades. 
                  Positive if the first blade (lowest phi) is shifted in +Z direction
  "Child" - name of a child volume being places (shold be in the form "file:volume")
  "ChildX" - child X in the "blade frame"
  "ChildY" - child Y in the "blade frame"
  "ChildZ" - child Z in the "blade frame"
  "ChildRotation" - rotation of the child volume with respect to the "blade frame". [OPTIONAL]
  "FlagString" - string of 24 characters, used to indicate blades into which the child volume 
                 should be placed. [OPTIONAL]
  "FlagSelector" - 1 character string, key to interpreting "FlagString".
                   Positions in "BladeFlag" that have this character will get the child volume.
  
  Blade frame: center at (0, 0, zPlane+k*bladeZShift) in disk frame, Z perpendicular to blade
  plane and pointing away from IP, X along blade axis pointing away from beam line.

*/

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDPixFwdBlades : public DDAlgorithm {
 
public:

  // Constructors & Destructor :  --------------------------------------------------------

  DDPixFwdBlades(); 
  virtual ~DDPixFwdBlades();
  
  // Initialization & Execution :  -------------------------------------------------------
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute();
  
  // -------------------------------------------------------------------------------------

private:

  // -- Parameters :  --------------------------------------------------------------------

  int           nBlades;            // Number of blades
  double        bladeAngle;         // Angle of blade rotation around axis perpendicular to beam
  double        zPlane;             // Common shift in Z for all blades (with respect to disk center plane)
  double        bladeZShift;        // Shift in Z between the axes of two adjacent blades

  std::string   flagString;         // String of flags
  std::string   flagSelector;       // Character that means "yes" in flagString
  
  std::string   childName;          // Child volume name
  
  double        childX;             // X of child volume center with respect to blade frame
  double        childY;             // Y of child volume center with respect to blade frame
  double        childZ;             // Z of child volume center with respect to blade frame
  std::string   childRotationName;  // Child volume rotation with respect to blade frame

  // -------------------------------------------------------------------------------------

  std::string   idNameSpace;    //Namespace of this and ALL sub-parts
  
  static std::map<std::string, int> copyNumbers;

  // -- Helper functions :  --------------------------------------------------------------
  
  int issueCopyNumber();
  
  // -------------------------------------------------------------------------------------

};

#endif
