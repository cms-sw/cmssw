#ifndef GEOMETRY_RECO_GEOMETRY_DT_GEOMETRY_BUILDER_H
#define GEOMETRY_RECO_GEOMETRY_DT_GEOMETRY_BUILDER_H

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include <Math/Rotation3D.h>
#include <Math/Vector3D.h>

namespace dd4hep {
  class Detector;
}

class DTGeometry;
class DTChamber;
class DTSuperLayer;
class DTLayer;

namespace cms {

  class DDDetector;
  struct DDFilteredView;
  struct MuonNumbering;
  struct DDSpecPar;
  
  class DTGeometryBuilder {
  public:
    DTGeometryBuilder() {}
    ~DTGeometryBuilder() {}
    
    using Detector = dd4hep::Detector;
    using DDRotationMatrix = ROOT::Math::Rotation3D;
    using DDTranslation = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
    using DD3Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
    using DDSpecParRefs = std::vector<const DDSpecPar*>;

    void build(DTGeometry&,
               const DDDetector*, 
               const MuonNumbering&,
	       const DDSpecParRefs&);
  private:
    void buildGeometry(DDFilteredView&,
		       const DDSpecPar&,
		       DTGeometry&, const MuonNumbering&) const;

    /// create the chamber
    DTChamber* buildChamber(const DDFilteredView&,
			    const DDTranslation&,
			    const DDRotationMatrix&,
                            const MuonNumbering&) const;
    
    /// create the SL
    DTSuperLayer* buildSuperLayer(const DDFilteredView&,
				  DTChamber*,
				  const DDTranslation&,
				  const DDRotationMatrix&,
				  const MuonNumbering&) const;

    /// create the layer
    DTLayer* buildLayer(const DDFilteredView&,
			DTSuperLayer*,
			const DDTranslation&,
			const DDRotationMatrix&,
			const MuonNumbering&) const;

    using RCPPlane = ReferenceCountingPointer<Plane>;

    RCPPlane plane(const DDTranslation&,
		   const DDRotationMatrix&,
		   Bounds* bounds) const;
  };
}

#endif
