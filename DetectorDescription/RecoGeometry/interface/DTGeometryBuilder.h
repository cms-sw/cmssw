#ifndef GEOMETRY_RECO_GEOMETRY_DT_GEOMETRY_BUILDER_H
#define GEOMETRY_RECO_GEOMETRY_DT_GEOMETRY_BUILDER_H

namespace cms {

  class DTGeometryBuilder {
  public:
    DTGeometryBuilder() {}
    ~DTGeometryBuilder() {}
    
    using Detector = dd4hep::Detector;
    using DD3Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;

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
