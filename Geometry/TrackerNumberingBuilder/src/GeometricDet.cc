#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "Geometry/TrackerNumberingBuilder/interface/TrackerShapeToBounds.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include "DetectorDescription/Core/interface/DDMaterial.h"

/**
 * What to do in the destructor?
 * For the moment nothing, I do not want to destroy all the daughters!
 */
GeometricDet::~GeometricDet(){}

GeometricDet::GeometricDet(nav_type navtype, GeometricEnumType type) : _ddd(navtype), _type(type){ 
  //
  // I need to find the params by myself :(
  //
  DDCompactView cpv;
  DDExpandedView ev(cpv);
  ev.goTo(_ddd);
  _params = ((ev.logicalPart()).solid()).parameters();
  _trans = ev.translation();
  _rot = ev.rotation();
  _shape = ((ev.logicalPart()).solid()).shape();
  _ddname = ((ev.logicalPart()).ddname()).name();
  _parents = ev.geoHistory();
  _volume   = ((ev.logicalPart()).solid()).volume();
  _density  = ((ev.logicalPart()).material()).density();
  //  _weight  = (ev.logicalPart()).weight();
  _weight   = _density * ( _volume / 1000.); // volume mm3->cm3
  _copy     = ev.copyno();
  _material = ((ev.logicalPart()).material()).name();
}

GeometricDet::GeometricDet(DDExpandedView* fv, GeometricEnumType type) : _type(type) {
  //
  // Set by hand the _ddd
  //
  _ddd = fv->navPos();
  _params = ((fv->logicalPart()).solid()).parameters();  
  _trans = fv->translation();
  _rot = fv->rotation();
  _shape = ((fv->logicalPart()).solid()).shape();
  _ddname = ((fv->logicalPart()).ddname()).name();
  _parents = fv->geoHistory();
  _volume   = ((fv->logicalPart()).solid()).volume();  
  _density  = ((fv->logicalPart()).material()).density();
  //  _weight   = (fv->logicalPart()).weight();  
  _weight   = _density * ( _volume / 1000.); // volume mm3->cm3
  _copy     = fv->copyno();
  _material = ((fv->logicalPart()).material()).name();
}

GeometricDet::GeometricDet(DDFilteredView* fv, GeometricEnumType type) : _type(type){
  //
  // Set by hand the _ddd
  //
  _ddd = fv->navPos();
  _params = ((fv->logicalPart()).solid()).parameters();
  _trans = fv->translation();
  _rot = fv->rotation();
  _shape = ((fv->logicalPart()).solid()).shape();
  _ddname = ((fv->logicalPart()).ddname()).name();
  _parents = fv->geoHistory();
  _volume   = ((fv->logicalPart()).solid()).volume();
  _density  = ((fv->logicalPart()).material()).density();
  //  _weight   = (fv->logicalPart()).weight();
  _weight   = _density * ( _volume / 1000.); // volume mm3->cm3
  _copy     = fv->copyno();
  _material = ((fv->logicalPart()).material()).name();
}

GeometricDet::ConstGeometricDetContainer GeometricDet::components() const{
  ConstGeometricDetContainer _temp;
  for (GeometricDetContainer::const_iterator it = _container.begin();
       it != _container.end(); it++){
    _temp.push_back(*it);
  }
  return _temp;
}

GeometricDet::ConstGeometricDetContainer GeometricDet::deepComponents() const {
  //
  // iterate on all the components ;)
  //
  ConstGeometricDetContainer _temp;
  if (isLeaf())
    _temp.push_back(const_cast<GeometricDet*>(this));
  else {
    for (GeometricDetContainer::const_iterator it = _container.begin();
	 it != _container.end(); it++){
      ConstGeometricDetContainer _temp2 =  (**it).deepComponents();
      copy(_temp2.begin(), _temp2.end(), back_inserter(_temp));
    }
  }
  return _temp;
}


DetId GeometricDet::geographicalID() const {
  return _geographicalID;
}

void GeometricDet::addComponents(GeometricDetContainer cont){
  for( GeometricDetContainer::iterator ig = cont.begin();
      ig != cont.end();ig++){
    _container.push_back(*ig);
  }
}


void GeometricDet::addComponent(GeometricDet* det){
  _container.push_back(det);
}

void GeometricDet::deleteComponents(){
  _container.erase(_container.begin(),_container.end());
}


void GeometricDet::deepDeleteComponents(){
  for (GeometricDetContainer::iterator it = _container.begin();
       it != _container.end(); it++){
    (const_cast<GeometricDet*>(*it))->deepDeleteComponents();
    delete (*it);
  }
  clearComponents();  
}

GeometricDet::Position GeometricDet::positionBounds() const{

  Position _pos(float(_trans.x()/cm), 
		float(_trans.y()/cm), 
		float(_trans.z()/cm));
  return _pos;
}

GeometricDet::Rotation GeometricDet::rotationBounds() const{

  Rotation _rotation(float(_rot.xx()),float(_rot.yx()),float(_rot.zx()),
		     float(_rot.xy()),float(_rot.yy()),float(_rot.zy()),
		     float(_rot.xz()),float(_rot.yz()),float(_rot.zz())); 
  
  return _rotation;
}

const Bounds * GeometricDet::bounds() const{
  const std::vector<double>& par = _params;
  Bounds * bounds = 0;
  TrackerShapeToBounds shapeToBounds;
  bounds = shapeToBounds.buildBounds(_shape,par);
  return bounds;
}

GeometricDet::GeometricDetContainer GeometricDet::components(){
  return _container;
}
