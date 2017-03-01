/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*	Jan Kašpar (jan.kaspar@gmail.com) 
*	CMSSW developers (based on GeometricDet class)
*
****************************************************************************/

#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionData.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

DetGeomDesc::DetGeomDesc(nav_type navtype, GeometricEnumType type) : _ddd(navtype), _type(type)
{ 
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
	_weight   = _density * ( _volume / 1000.); // volume mm3->cm3
	_copy     = ev.copyno();
	_material = ((ev.logicalPart()).material()).name();
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc::DetGeomDesc(DDExpandedView* fv, GeometricEnumType type) : _type(type)
{
	_ddd = fv->navPos();
	_params = ((fv->logicalPart()).solid()).parameters();  
	_trans = fv->translation();
	_rot = fv->rotation();
	_shape = ((fv->logicalPart()).solid()).shape();
	_ddname = ((fv->logicalPart()).ddname()).name();
	_parents = fv->geoHistory();
	_volume   = ((fv->logicalPart()).solid()).volume();  
	_density  = ((fv->logicalPart()).material()).density();
	_weight   = _density * ( _volume / 1000.); // volume mm3->cm3
	_copy     = fv->copyno();
	_material = ((fv->logicalPart()).material()).name();
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc::DetGeomDesc(DDFilteredView* fv, GeometricEnumType type) : _type(type)
{
	_ddd = fv->navPos();
	_params = ((fv->logicalPart()).solid()).parameters();
	_trans = fv->translation();
	_rot = fv->rotation();
	_shape = ((fv->logicalPart()).solid()).shape();
	_ddname = ((fv->logicalPart()).ddname()).name();
	_parents = fv->geoHistory();
	_volume   = ((fv->logicalPart()).solid()).volume();
	_density  = ((fv->logicalPart()).material()).density();
	_weight   = _density * ( _volume / 1000.); // volume mm3->cm3
	_copy     = fv->copyno();
	_material = ((fv->logicalPart()).material()).name();
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc::DetGeomDesc(const DetGeomDesc &ref)
{
	(*this) = ref;
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc& DetGeomDesc::operator= (const DetGeomDesc &ref)
{
	_ddd			= ref._ddd;
	_params			= ref._params;
	_trans			= ref._trans;
	_rot			= ref._rot;
	_shape			= ref._shape;
	_ddname			= ref._ddname;
	_parents		= ref._parents;
	_volume			= ref._volume;
	_density		= ref._density;
	_weight			= ref._weight;
	_copy			= ref._copy;
	_material		= ref._material;
	_geographicalID	= ref._geographicalID;
	_type			= ref._type;

	return (*this);
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc::~DetGeomDesc()
{
	deepDeleteComponents();
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc::Container DetGeomDesc::components()
{
	return _container;
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc::ConstContainer DetGeomDesc::components() const
{
	ConstContainer _temp;
	for (Container::const_iterator it = _container.begin(); it != _container.end(); it++) {
		_temp.push_back(*it);
	}
	return _temp;
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc::ConstContainer DetGeomDesc::deepComponents() const
{
  ConstContainer _temp;
  if (isLeaf())
    _temp.push_back(const_cast<DetGeomDesc*>(this));
  else {
    for (Container::const_iterator it = _container.begin();
	 it != _container.end(); it++){
      ConstContainer _temp2 =  (**it).deepComponents();
      copy(_temp2.begin(), _temp2.end(), back_inserter(_temp));
    }
  }
  return _temp;
}


//----------------------------------------------------------------------------------------------------

void DetGeomDesc::addComponents(Container cont)
{
	for( Container::iterator ig = cont.begin(); ig != cont.end();ig++) {
		_container.push_back(*ig);
	}
}

//----------------------------------------------------------------------------------------------------

void DetGeomDesc::addComponent(DetGeomDesc* det)
{
	_container.push_back(det);
}

//----------------------------------------------------------------------------------------------------

void DetGeomDesc::deleteComponents()
{
	_container.erase(_container.begin(), _container.end());
}

//----------------------------------------------------------------------------------------------------

void DetGeomDesc::deepDeleteComponents()
{
	for (Container::iterator it = _container.begin(); it != _container.end(); it++) {
		( const_cast<DetGeomDesc*>(*it) )->deepDeleteComponents();
		delete (*it);
	}
	clearComponents();  
}

//----------------------------------------------------------------------------------------------------

void DetGeomDesc::ApplyAlignment(const RPAlignmentCorrectionData &t)
{
    //cout << " DetGeomDesc::ApplyAlignment > before: " << _trans << ",  " << _rot << endl;
	_rot = t.getRotationMatrix() * _rot;
	_trans = t.getTranslation() + _trans;
    //cout << " DetGeomDesc::ApplyAlignment > after: " << _trans << ",  " << _rot << endl;
}
