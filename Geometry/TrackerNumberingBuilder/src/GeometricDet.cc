#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/TrackerShapeToBounds.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <boost/bind.hpp>

#include <cfloat>
#include <vector>
#include <string>

namespace {

  std::string strue("true");

  template<typename DDView>
  double  getDouble(const  char * s,  DDView const & ev) {
    DDValue val(s);
    std::vector<const DDsvalues_type *> result;
    ev.specificsV(result);
    std::vector<const DDsvalues_type *>::iterator it = result.begin();
    bool foundIt = false;
    for (; it != result.end(); ++it)
      {
	foundIt = DDfetch(*it,val);
	if (foundIt) break;
      }    
    if (foundIt)
      { 
	const std::vector<std::string> & temp = val.strings(); 
	if (temp.size() != 1)
	  {
	    throw cms::Exception("Configuration") << "I need 1 "<< s << " tags";
	  }
	return double(::atof(temp[0].c_str())); 
      }
    return 0;
  }

  template<typename DDView>
  std::string  getString(const  char * s,  DDView const & ev) {
    DDValue val(s);
    std::vector<const DDsvalues_type *> result;
    ev.specificsV(result);
    std::vector<const DDsvalues_type *>::iterator it = result.begin();
    bool foundIt = false;
    for (; it != result.end(); ++it)
    {
	foundIt = DDfetch(*it,val);
	if (foundIt) break;

    }    
    if (foundIt)
    { 
	const std::vector<std::string> & temp = val.strings(); 
	if (temp.size() != 1)
	{
	  throw cms::Exception("Configuration") << "I need 1 "<< s << " tags";
	}
	return temp[0]; 
    }
    return "NotFound";
  }
}


/**
 * What to do in the destructor?
 * destroy all the daughters!
 */
GeometricDet::~GeometricDet(){
  //std::cout << "~GeometricDet5" << std::endl;
  deleteComponents();
}
#ifdef GEOMETRICDETDEBUG
// for use outside CMSSW framework only since it asks for a default DDCompactView...
GeometricDet::GeometricDet(DDnav_type const & navtype, GeometricEnumType type) :
  _ddd(navtype.begin(),navtype.end()), _type(type){ 
  //
  // I need to find the params by myself :(
  //
  //std::cout << "GeometricDet1" << std::endl;
  _fromDD = true;
  DDCompactView cpv; // bad, bad, bad!
  DDExpandedView ev(cpv);
  ev.goTo(navtype);
  _params = ((ev.logicalPart()).solid()).parameters();
  _trans = ev.translation();
  _phi = _trans.Phi();
  _rho = _trans.Rho();
  _rot = ev.rotation();
  _shape = ((ev.logicalPart()).solid()).shape();
  _ddname = ((ev.logicalPart()).ddname()).name();
  _parents = GeoHistory(ev.geoHistory().begin(),ev.geoHistory().end()) ;
  _volume   = ((ev.logicalPart()).solid()).volume();
  _density  = ((ev.logicalPart()).material()).density();
  //  _weight  = (ev.logicalPart()).weight();
  _weight   = _density * ( _volume / 1000.); // volume mm3->cm3
  _copy     = ev.copyno();
  _material = ((ev.logicalPart()).material()).name().fullname();
  _radLength = getDouble("TrackerRadLength",ev);
  _xi = getDouble("TrackerXi",ev);
  _pixROCRows = getDouble("PixelROCRows",ev);
  _pixROCCols = getDouble("PixelROCCols",ev);
  _pixROCx = getDouble("PixelROC_X",ev);
  _pixROCy = getDouble("PixelROC_Y",ev);
  _stereo =  getString("TrackerStereoDetectors",ev)==strue;
  _siliconAPVNum = getDouble("SiliconAPVNumber",ev);

}

GeometricDet::GeometricDet(DDExpandedView* fv, GeometricEnumType type) :  _type(type) {
  //
  // Set by hand the _ddd
  //
  //std::cout << "GeometricDet2" << std::endl;
  _fromDD = true;
  _ddd = nav_type(fv->navPos().begin(),fv->navPos().end() );
  _params = ((fv->logicalPart()).solid()).parameters();  
  _trans = fv->translation();
  _phi = _trans.Phi();
  _rho = _trans.Rho();
  _rot = fv->rotation();
  _shape = ((fv->logicalPart()).solid()).shape();
  _ddname = ((fv->logicalPart()).ddname()).name();
  _parents = GeoHistory(fv->geoHistory().begin(),fv->geoHistory().end()) ;
  _volume   = ((fv->logicalPart()).solid()).volume();  
  _density  = ((fv->logicalPart()).material()).density();
  //  _weight   = (fv->logicalPart()).weight();  
  _weight   = _density * ( _volume / 1000.); // volume mm3->cm3
  _copy     = fv->copyno();
  _material = ((fv->logicalPart()).material()).name().fullname();
  _radLength = getDouble("TrackerRadLength",*fv);
  _xi = getDouble("TrackerXi",*fv);
  _pixROCRows = getDouble("PixelROCRows",*fv);
  _pixROCCols = getDouble("PixelROCCols",*fv);
  _pixROCx = getDouble("PixelROC_X",*fv);
  _pixROCy = getDouble("PixelROC_Y",*fv);
  _stereo =  getString("TrackerStereoDetectors",*fv)=="true";
  _siliconAPVNum = getDouble("SiliconAPVNumber",*fv);

}
#endif

GeometricDet::GeometricDet(DDFilteredView* fv, GeometricEnumType type) : 
  //
  // Set by hand the _ddd
  //
  _trans(fv->translation()),
  _phi(_trans.Phi()),
  _rho(_trans.Rho()),
  _rot(fv->rotation()),
  _shape(((fv->logicalPart()).solid()).shape()),
  _ddname(((fv->logicalPart()).ddname()).name()),
  _type(type),
  _params(((fv->logicalPart()).solid()).parameters()),
  //  want this :) _ddd(fv->navPos().begin(),fv->navPos().end()),
#ifdef GEOMTRICDETDEBUG
  _parents(fv->geoHistory().begin(),fv->geoHistory().end()),
  _volume(((fv->logicalPart()).solid()).volume()),
  _density(((fv->logicalPart()).material()).density()),
  //  _weight   = (fv->logicalPart()).weight();
  _weight(_density * ( _volume / 1000.)), // volume mm3->cm3
  _copy(fv->copyno()),
  _material(((fv->logicalPart()).material()).name().fullname()),
#endif
  _radLength(getDouble("TrackerRadLength",*fv)),
  _xi(getDouble("TrackerXi",*fv)),
  _pixROCRows(getDouble("PixelROCRows",*fv)),
  _pixROCCols(getDouble("PixelROCCols",*fv)),
  _pixROCx(getDouble("PixelROC_X",*fv)),
  _pixROCy(getDouble("PixelROC_Y",*fv)),
  _stereo(getString("TrackerStereoDetectors",*fv)==strue),
  _siliconAPVNum(getDouble("SiliconAPVNumber",*fv))
#ifdef GEOMTRICDETDEBUG
  ,
  _fromDD(true)
#endif
{
  //std::cout << "GeometricDet3" << std::endl;
  //  workaround instead of this at initialization _ddd(fv->navPos().begin(),fv->navPos().end()),
  const DDFilteredView::nav_type& nt = fv->navPos();
  _ddd = nav_type(nt.begin(), nt.end());
}

// PGeometricDet is persistent version... make it... then come back here and make the
// constructor.
GeometricDet::GeometricDet ( const PGeometricDet::Item& onePGD, GeometricEnumType type) :
  _trans(onePGD._x, onePGD._y, onePGD._z),
  _phi(onePGD._phi), //_trans.Phi()),
  _rho(onePGD._rho), //_trans.Rho()),
  _rot(onePGD._a11, onePGD._a12, onePGD._a13, 
       onePGD._a21, onePGD._a22, onePGD._a23,
       onePGD._a31, onePGD._a32, onePGD._a33),
  _shape(DDSolidShapesName::index(onePGD._shape)),
  _ddd(), 
  _ddname(onePGD._name, onePGD._ns),//, "fromdb");
  _type(type),
  _params(),
  _geographicalID(onePGD._geographicalID),
#ifdef GEOMTRICDETDEBUG
  _parents(), // will remain empty... hate wasting the space but want all methods to work.
  _volume(onePGD._volume),
  _density(onePGD._density),
  _weight(onePGD._weight),
  _copy(onePGD._copy),
  _material(onePGD._material),
#endif
  _radLength(onePGD._radLength),
  _xi(onePGD._xi),
  _pixROCRows(onePGD._pixROCRows),
  _pixROCCols(onePGD._pixROCCols),
  _pixROCx(onePGD._pixROCx),
  _pixROCy(onePGD._pixROCy),
  _stereo(onePGD._stereo),
  _siliconAPVNum(onePGD._siliconAPVNum)
#ifdef GEOMTRICDETDEBUG
  , // mind the tricky comma is needed.
  _fromDD(false)
#endif
{
  //std::cout << "GeometricDet4" << std::endl;
  
  if(onePGD._shape==1||onePGD._shape==3){ //The parms vector is neede only in the case of box or trap shape
    _params.reserve(11);
    _params.push_back(onePGD._params0);
    _params.push_back(onePGD._params1);
    _params.push_back(onePGD._params2);
    _params.push_back(onePGD._params3);
    _params.push_back(onePGD._params4);
    _params.push_back(onePGD._params5);
    _params.push_back(onePGD._params6);
    _params.push_back(onePGD._params7);
    _params.push_back(onePGD._params8);
    _params.push_back(onePGD._params9);
    _params.push_back(onePGD._params10);
  }
 
  _ddd.reserve(onePGD._numnt);
  _ddd.push_back(onePGD._nt0);
  _ddd.push_back(onePGD._nt1);
  _ddd.push_back(onePGD._nt2);
  _ddd.push_back(onePGD._nt3);
  if ( onePGD._numnt > 4 ) {
    _ddd.push_back(onePGD._nt4);
    if ( onePGD._numnt > 5 ) {
      _ddd.push_back(onePGD._nt5);
      if ( onePGD._numnt > 6 ) {
	_ddd.push_back(onePGD._nt6);
	if ( onePGD._numnt > 7 ) {
	  _ddd.push_back(onePGD._nt7);
	  if ( onePGD._numnt > 8 ) {
	    _ddd.push_back(onePGD._nt8);
	    if ( onePGD._numnt > 9 ) {
	      _ddd.push_back(onePGD._nt9);
	      if ( onePGD._numnt > 10 ) {
		_ddd.push_back(onePGD._nt10);
	      }}}}}}
  }
 
}

GeometricDet::ConstGeometricDetContainer GeometricDet::deepComponents() const {
  //
  // iterate on all the components ;)
  //
  //std::cout << "deepComponents1" << std::endl;
  ConstGeometricDetContainer _temp;
  deepComponents(_temp);
  return _temp;
}

void GeometricDet::deepComponents(GeometricDetContainer & cont) const {
  //std::cout << "const deepComponents2" << std::endl;
  if (isLeaf())
    cont.push_back(const_cast<GeometricDet*>(this));
  else 
    std::for_each(_container.begin(),_container.end(), 
		  boost::bind(&GeometricDet::deepComponents,_1,boost::ref(cont))
		  );
}


void GeometricDet::addComponents(GeometricDetContainer const & cont){
  //std::cout << "addComponents" << std::endl;
  if (_container.empty()) {
    _container=cont;
    return;
  }
  _container.reserve(_container.size()+cont.size());
  std::copy(cont.begin(), cont.end(), back_inserter(_container));
}


void GeometricDet::addComponent(GeometricDet* det){
  //std::cout << "deepComponent" << std::endl;
  _container.push_back(det);
}

namespace {
  struct Deleter {
    void operator()(GeometricDet const* det) const { delete const_cast<GeometricDet*>(det);}
  };
}

void GeometricDet::deleteComponents(){
  //std::cout << "deleteComponents" << std::endl;
  std::for_each(_container.begin(),_container.end(),Deleter()); 
  _container.clear();
}


GeometricDet::Position GeometricDet::positionBounds() const{
  //std::cout << "positionBounds" << std::endl;
  Position _pos(float(_trans.x()/cm), 
		float(_trans.y()/cm), 
		float(_trans.z()/cm));
  return _pos;
}

GeometricDet::Rotation GeometricDet::rotationBounds() const{
  //std::cout << "rotationBounds" << std::endl;
  DD3Vector x, y, z;
  _rot.GetComponents(x, y, z);
  Rotation _rotation(float(x.X()),float(x.Y()),float(x.Z()),
		     float(y.X()),float(y.Y()),float(y.Z()),
		     float(z.X()),float(z.Y()),float(z.Z())); 
  return _rotation;
}

Bounds * GeometricDet::bounds() const{
  //std::cout << "bounds" << std::endl;
  const std::vector<double>& par = _params;
  TrackerShapeToBounds shapeToBounds;
  return shapeToBounds.buildBounds(_shape,par);
}

