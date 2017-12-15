#ifndef GeometryHGCalGeometryHGCellGeometry_h
#define GeometryHGCalGeometryHGCellGeometry_h

#include "Geometry/CaloGeometry/interface/FlatTrd.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Plane3D.h>
#include <CLHEP/Geometry/Vector3D.h>
#include <CLHEP/Geometry/Transform3D.h>
#include <vector>

class HGCellGeometry : public FlatTrd {
public:

  typedef CaloCellGeometry::CCGFloat CCGFloat ;
  typedef CaloCellGeometry::Pt3D     Pt3D     ;
  typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
  typedef CaloCellGeometry::Tr3D     Tr3D     ;

  HGCellGeometry(void);
  
  HGCellGeometry(const HGCellGeometry& tr) ;
  
  HGCellGeometry& operator=(const HGCellGeometry& tr) ;
  
  HGCellGeometry(const HGCalTopology*, CornersMgr*  cMgr ,
		 const GlobalPoint& fCtr ,
		 const GlobalPoint& bCtr ,
		 const GlobalPoint& cor1 ,
		 const CCGFloat*    parV );
  
  HGCellGeometry(const HGCalTopology*, const CornersVec& corn ,
		 const CCGFloat*   par    ) ;
  
  HGCellGeometry(const HGCalTopology*, const FlatTrd& tr, const Pt3D& local) ;

  ~HGCellGeometry() override ;
  
  const GlobalPoint getPosition(const DetId&) const override;

  const std::vector<GlobalPoint> getCorners(const DetId&) const override;

  const HGCalTopology& topology() const {return *topo_;}

private:

  const HGCalTopology*  topo_;
};

std::ostream& operator<<( std::ostream& s, const HGCellGeometry& cell ) ;

#endif
