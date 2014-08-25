#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDRoot.h"

DDRoot::DDRoot()
{ }

DDRoot::~DDRoot()
{ }

void DDRoot::set(const DDName & name)
{
   DCOUT('C',"DDRoot::set() root=" << name);
   root_ = DDLogicalPart(name);
}

void DDRoot::set(const DDLogicalPart & root)
{
   DCOUT('C',"DDRoot::set() root=" << root);
   root_ = root;
}

/**
  To find out, whether the root was already defined or not:
  \code
    DDLogicalPart root;
    if(root=DDRoot::instance().root()) { // ok, root was already defined
      // so something here ...
    }
    else { // root has not been defined yet!
      // do something else
    }      
   \endcode 
*/ 
DDLogicalPart DDRoot::root() const { return root_; } 
