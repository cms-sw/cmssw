#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DDRoot::DDRoot() {}

DDRoot::~DDRoot() {}

void DDRoot::set(const DDName& name) { root_ = DDLogicalPart(name); }

void DDRoot::set(const DDLogicalPart& root) { root_ = root; }

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
