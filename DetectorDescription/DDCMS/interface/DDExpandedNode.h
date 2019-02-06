#ifndef DETECTOR_DESCRIPTION_DD_EXPANDED_NODE_H
#define DETECTOR_DESCRIPTION_DD_EXPANDED_NODE_H

#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include <Math/Rotation3D.h>
#include <Math/Vector3D.h>

#include <DD4hep/Volumes.h>

namespace cms {

  using DDVolume = dd4hep::Volume;
  using DDTranslation = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
  using DDRotationMatrix = ROOT::Math::Rotation3D;
    
  struct DDExpandedNode {
    
    DDVolume volume; // logicalpart to provide access to solid & material information
    DDTranslation trans;   // absolute translation
    DDRotationMatrix rot;  // absolute rotation
    int copyNo;
    int siblingno;         // internal sibling-numbering from 0 to max-sibling
    DDSpecPar specpar;
    
    DDExpandedNode(DDVolume const& aVolume, DDTranslation const& aTrans,
		   DDRotationMatrix const& aRot, int aCopyNo, int num)
      : volume(aVolume),
        trans(aTrans),
        rot(aRot),
        copyNo(aCopyNo),
        siblingno(num) {
      }
  };
}

#endif
