#ifndef DDRoot_h
#define DDRoot_h

#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Base/interface/Singleton.h"

//!  Defines the root of the CompactView
/**
  DDRoot will define the root of the geometrical hierarchy. The root also
  defines the base of the global coordinates.
  /todo provide possibility to have different roots for different parallel geometries
  /todo prohibit multiple calls for one geometry (the root can only be defined once!)
*/
class DDRoot
{
public:
  DDRoot();
  ~DDRoot();
  //! set the root by using its qualified name DDName
  void set(const DDName & rootName);
  
  //! set DDLogicalPart root to the root 
  void set(const DDLogicalPart & root);
  
  //! returns the root of the geometrical hierarchy
  DDLogicalPart root() const;

private:
  DDLogicalPart root_;
};

typedef DDI::Singleton<DDRoot> DDRootDef;
#endif
