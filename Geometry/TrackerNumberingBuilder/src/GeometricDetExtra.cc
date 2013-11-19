//#include "CondFormats/GeometryObjects/interface/PGeometricDetExtra.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDetExtra.h"
//#include "DetectorDescription/Core/interface/DDExpandedView.h"
/* #include "DetectorDescription/Base/interface/DDRotationMatrix.h" */
/* #include "DetectorDescription/Base/interface/DDTranslation.h" */
/* #include "DetectorDescription/Core/interface/DDSolidShapes.h" */
/* #include "DataFormats/GeometrySurface/interface/Surface.h" */
/* #include "DataFormats/GeometrySurface/interface/Bounds.h" */
//#include "DataFormats/DetId/interface/DetId.h"

//#include <vector>
//#include "FWCore/ParameterSet/interface/types.h"

/**
 * Constructors to be used when looping over DDD
 */
GeometricDetExtra::GeometricDetExtra( GeometricDet const * gd, DetId id, GeoHistory& gh,  double vol, double dens, double wgt, double cpy, const std::string& mat, const std::string& name, bool dd )
  : _mygd(gd), _parents(gh), _volume(vol), _density(dens), _weight(wgt), _copy((int)(cpy)), _material(mat), _name(name), _fromDD(dd), _geographicalId(id)
{ 
}

GeometricDetExtra::~GeometricDetExtra()
{ }

// copy-ctor
GeometricDetExtra::GeometricDetExtra(const GeometricDetExtra& src)
    : _mygd(src._mygd), _parents(src._parents), _density(src._density), _weight(src._weight),
      _copy(src._copy), _material(src._material), _name(src._name), _fromDD(src._fromDD),
      _geographicalId(src._geographicalId) {}
// copy assignment operator
GeometricDetExtra&
GeometricDetExtra::operator=(const GeometricDetExtra& rhs) {
    GeometricDetExtra temp(rhs);
    temp.swap(*this);
    return *this;
}
// public swap function
void GeometricDetExtra::swap(GeometricDetExtra& other) {
    std::swap(_mygd, other._mygd);
    std::swap(_parents, other._parents);
    std::swap(_density, other._density);
    std::swap(_weight, other._weight);
    std::swap(_copy, other._copy);
    std::swap(_material, other._material);
    std::swap(_name, other._name);
    std::swap(_fromDD, other._fromDD);
    std::swap(_geographicalId, other._geographicalId);
}
