#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/TrackerShapeToBounds.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "CondFormats/GeometryObjects/interface/PGeometricTimingDet.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <boost/bind.hpp>

#include <cfloat>
#include <vector>
#include <string>

namespace {

  const std::string strue("true");

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
GeometricTimingDet::~GeometricTimingDet(){
  deleteComponents();
}
#ifdef GEOMETRICDETDEBUG
// for use outside CMSSW framework only since it asks for a default DDCompactView...
GeometricTimingDet::GeometricTimingDet(DDnav_type const & navtype, GeometricTimingEnumType type) :
  ddd_(navtype.begin(),navtype.end()), type_(type){ 
  //
  // I need to find the params by myself :(
  //
  //std::cout << "GeometricTimingDet1" << std::endl;
  fromDD_ = true;
  DDCompactView cpv; // bad, bad, bad!
  DDExpandedView ev(cpv);
  ev.goTo(navtype);
  params_ = ((ev.logicalPart()).solid()).parameters();
  trans_ = ev.translation();
  phi_ = trans_.Phi();
  rho_ = trans_.Rho();
  rot_ = ev.rotation();
  shape_ = ((ev.logicalPart()).solid()).shape();
  ddname_ = ((ev.logicalPart()).ddname()).name();
  parents_ = GeoHistory(ev.geoHistory().begin(),ev.geoHistory().end()) ;
  volume_   = ((ev.logicalPart()).solid()).volume();
  density_  = ((ev.logicalPart()).material()).density();
  //  _weight  = (ev.logicalPart()).weight();
  weight_   = density_ * ( volume_ / 1000.); // volume mm3->cm3
  copy_     = ev.copyno();
  material_ = ((ev.logicalPart()).material()).name().fullname();
  radLength_ = getDouble("TrackerRadLength",ev);
  xi_ = getDouble("TrackerXi",ev);
  pixROCRows_ = getDouble("PixelROCRows",ev);
  pixROCCols_ = getDouble("PixelROCCols",ev);
  pixROCx_ = getDouble("PixelROC_X",ev);
  pixROCy_ = getDouble("PixelROC_Y",ev);
  stereo_ =  getString("TrackerStereoDetectors",ev)==strue;
  siliconAPVNum_ = getDouble("SiliconAPVNumber",ev);

}

GeometricTimingDet::GeometricTimingDet(DDExpandedView* fv, GeometricTimingEnumType type) :  type_(type) {
  //
  // Set by hand the ddd_
  //
  //std::cout << "GeometricTimingDet2" << std::endl;
  fromDD_ = true;
  ddd_ = nav_type(fv->navPos().begin(),fv->navPos().end() );
  params_ = ((fv->logicalPart()).solid()).parameters();  
  trans_ = fv->translation();
  phi_ = trans_.Phi();
  rho_ = trans_.Rho();
  rot_ = fv->rotation();
  shape_ = ((fv->logicalPart()).solid()).shape();
  ddname_ = ((fv->logicalPart()).ddname()).name();
  parents_ = GeoHistory(fv->geoHistory().begin(),fv->geoHistory().end()) ;
  volume_   = ((fv->logicalPart()).solid()).volume();  
  density_  = ((fv->logicalPart()).material()).density();
  //  weight_   = (fv->logicalPart()).weight();  
  weight_   = density_ * ( volume_ / 1000.); // volume mm3->cm3
  copy_     = fv->copyno();
  material_ = ((fv->logicalPart()).material()).name().fullname();
  radLength_ = getDouble("TrackerRadLength",*fv);
  xi_ = getDouble("TrackerXi",*fv);
  pixROCRows_ = getDouble("PixelROCRows",*fv);
  pixROCCols_ = getDouble("PixelROCCols",*fv);
  pixROCx_ = getDouble("PixelROC_X",*fv);
  pixROCy_ = getDouble("PixelROC_Y",*fv);
  stereo_ =  getString("TrackerStereoDetectors",*fv)=="true";
  siliconAPVNum_ = getDouble("SiliconAPVNumber",*fv);

}
#endif

GeometricTimingDet::GeometricTimingDet(DDFilteredView* fv, GeometricTimingEnumType type) : 
  //
  // Set by hand the ddd_
  //
  trans_(fv->translation()),
  phi_(trans_.Phi()),
  rho_(trans_.Rho()),
  rot_(fv->rotation()),
  shape_(((fv->logicalPart()).solid()).shape()),
  ddname_(((fv->logicalPart()).ddname()).name()),
  type_(type),
  params_(((fv->logicalPart()).solid()).parameters()),
  //  want this :) ddd_(fv->navPos().begin(),fv->navPos().end()),
#ifdef GEOMTRICDETDEBUG
  parents_(fv->geoHistory().begin(),fv->geoHistory().end()),
  volume_(((fv->logicalPart()).solid()).volume()),
  density_(((fv->logicalPart()).material()).density()),
  //  _weight   = (fv->logicalPart()).weight();
  weight_(density_ * ( volume_ / 1000.)), // volume mm3->cm3
  copy_(fv->copyno()),
  material_(((fv->logicalPart()).material()).name().fullname()),
#endif
  radLength_(getDouble("TrackerRadLength",*fv)),
  xi_(getDouble("TrackerXi",*fv)),
  pixROCRows_(getDouble("PixelROCRows",*fv)),
  pixROCCols_(getDouble("PixelROCCols",*fv)),
  pixROCx_(getDouble("PixelROC_X",*fv)),
  pixROCy_(getDouble("PixelROC_Y",*fv)),
  stereo_(getString("TrackerStereoDetectors",*fv)==strue),
  siliconAPVNum_(getDouble("SiliconAPVNumber",*fv))
#ifdef GEOMTRICDETDEBUG
  ,
  fromDD_(true)
#endif
{
  const DDFilteredView::nav_type& nt = fv->navPos();
  ddd_ = nav_type(nt.begin(), nt.end());
}

// PGeometricTimingDet is persistent version... make it... then come back here and make the
// constructor.
GeometricTimingDet::GeometricTimingDet ( const PGeometricTimingDet::Item& onePGD, GeometricTimingEnumType type) :
  trans_(onePGD.x_, onePGD.y_, onePGD.z_),
  phi_(onePGD.phi_), //_trans.Phi()),
  rho_(onePGD.rho_), //_trans.Rho()),
  rot_(onePGD.a11_, onePGD.a12_, onePGD.a13_, 
       onePGD.a21_, onePGD.a22_, onePGD.a23_,
       onePGD.a31_, onePGD.a32_, onePGD.a33_),
  shape_(static_cast<DDSolidShape>(onePGD.shape_)),
  ddd_(), 
  ddname_(onePGD.name_, onePGD.ns_),//, "fromdb");
  type_(type),
  params_(),
  geographicalID_(onePGD.geographicalID_),
#ifdef GEOMTRICDETDEBUG
  parents_(), // will remain empty... hate wasting the space but want all methods to work.
  volume_(onePGD.volume_),
  density_(onePGD.density_),
  weight_(onePGD.weight_),
  copy_(onePGD.copy_),
  material_(onePGD.material_),
#endif
  radLength_(onePGD.radLength_),
  xi_(onePGD.xi_),
  pixROCRows_(onePGD.pixROCRows_),
  pixROCCols_(onePGD.pixROCCols_),
  pixROCx_(onePGD.pixROCx_),
  pixROCy_(onePGD.pixROCy_),
  stereo_(onePGD.stereo_),
  siliconAPVNum_(onePGD.siliconAPVNum_)
#ifdef GEOMTRICDETDEBUG
  , // mind the tricky comma is needed.
  fromDD_(false)
#endif
{
  //std::cout << "GeometricTimingDet4" << std::endl;
  
  if(onePGD.shape_==1||onePGD.shape_==3){ //The parms vector is neede only in the case of box or trap shape
    params_.reserve(11);
    params_.emplace_back(onePGD.params_0);
    params_.emplace_back(onePGD.params_1);
    params_.emplace_back(onePGD.params_2);
    params_.emplace_back(onePGD.params_3);
    params_.emplace_back(onePGD.params_4);
    params_.emplace_back(onePGD.params_5);
    params_.emplace_back(onePGD.params_6);
    params_.emplace_back(onePGD.params_7);
    params_.emplace_back(onePGD.params_8);
    params_.emplace_back(onePGD.params_9);
    params_.emplace_back(onePGD.params_10);
  }
 
  ddd_.reserve(onePGD.numnt_);
  ddd_.emplace_back(onePGD.nt0_);
  ddd_.emplace_back(onePGD.nt1_);
  ddd_.emplace_back(onePGD.nt2_);
  ddd_.emplace_back(onePGD.nt3_);
  if ( onePGD.numnt_ > 4 ) {
    ddd_.emplace_back(onePGD.nt4_);
    if ( onePGD.numnt_ > 5 ) {
      ddd_.emplace_back(onePGD.nt5_);
      if ( onePGD.numnt_ > 6 ) {
	ddd_.emplace_back(onePGD.nt6_);
	if ( onePGD.numnt_ > 7 ) {
	  ddd_.emplace_back(onePGD.nt7_);
	  if ( onePGD.numnt_ > 8 ) {
	    ddd_.emplace_back(onePGD.nt8_);
	    if ( onePGD.numnt_ > 9 ) {
	      ddd_.emplace_back(onePGD.nt9_);
	      if ( onePGD.numnt_ > 10 ) {
		ddd_.emplace_back(onePGD.nt10_);
	      }}}}}}
  }
 
}

GeometricTimingDet::ConstGeometricTimingDetContainer GeometricTimingDet::deepComponents() const {
  //
  // iterate on all the components ;)
  //
  ConstGeometricTimingDetContainer temp;
  deepComponents(temp);
  return temp;
}

void GeometricTimingDet::deepComponents(ConstGeometricTimingDetContainer & cont) const {
  if (isLeaf()) {
    cont.emplace_back(this);
  }  
  else 
    std::for_each(container_.begin(),container_.end(), 
		  [&](const GeometricTimingDet* iDet) {
		    iDet->deepComponents(cont);
		  }
		  );
}

void GeometricTimingDet::addComponents(GeometricTimingDetContainer const & cont){
  container_.reserve(container_.size()+cont.size());
  std::copy(cont.begin(), cont.end(), back_inserter(container_));
}

void GeometricTimingDet::addComponents(ConstGeometricTimingDetContainer const & cont){
  container_.reserve(container_.size()+cont.size());
  std::copy(cont.begin(), cont.end(), back_inserter(container_));
}

void GeometricTimingDet::addComponent(GeometricTimingDet* det){
  container_.emplace_back(det);
}

namespace {
  struct Deleter {
    void operator()(GeometricTimingDet const* det) const { delete const_cast<GeometricTimingDet*>(det);}
  };
}

void GeometricTimingDet::deleteComponents(){
  std::for_each(container_.begin(),container_.end(),Deleter()); 
  container_.clear();
}


GeometricTimingDet::Position GeometricTimingDet::positionBounds() const{
  Position pos(float(trans_.x()/cm), 
	       float(trans_.y()/cm), 
	       float(trans_.z()/cm));
  return pos;
}

GeometricTimingDet::Rotation GeometricTimingDet::rotationBounds() const{
  DD3Vector x, y, z;
  rot_.GetComponents(x, y, z);
  Rotation rotation(float(x.X()),float(x.Y()),float(x.Z()),
		    float(y.X()),float(y.Y()),float(y.Z()),
		    float(z.X()),float(z.Y()),float(z.Z())); 
  return rotation;
}

std::unique_ptr<Bounds> GeometricTimingDet::bounds() const{
  const std::vector<double>& par = params_;
  TrackerShapeToBounds shapeToBounds;
  return std::unique_ptr<Bounds>(shapeToBounds.buildBounds(shape_,par));
}

