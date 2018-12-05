/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*	Jan Kašpar (jan.kaspar@gmail.com) 
*	CMSSW developers (based on GeometricDet class)
*
****************************************************************************/

#include <utility>

#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionData.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

DetGeomDesc::DetGeomDesc( DDFilteredView* fv ) :
  m_trans( fv->translation()),
  m_rot( fv->rotation()),
  m_name((( fv->logicalPart()).ddname()).name()),
  m_params((( fv->logicalPart()).solid()).parameters()),
  m_copy( fv->copyno()),
  m_z( fv->geoHistory().back().absTranslation().z())
{}

//----------------------------------------------------------------------------------------------------
DetGeomDesc::DetGeomDesc(const DetGeomDesc &ref)
{
	(*this) = ref;
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc& DetGeomDesc::operator= ( const DetGeomDesc &ref )
{
  m_params = ref.m_params;
  m_trans = ref.m_trans;
  m_rot = ref.m_rot;
  m_name = ref.m_name;
  m_copy = ref.m_copy;
  m_geographicalID = ref.m_geographicalID;
  m_z = ref.m_z;
  return (*this);
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc::~DetGeomDesc()
{
  deepDeleteComponents();
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc::Container DetGeomDesc::components() const
{
  return m_container;
}

//----------------------------------------------------------------------------------------------------

void DetGeomDesc::addComponent( DetGeomDesc* det )
{
  m_container.emplace_back( det );
}

//----------------------------------------------------------------------------------------------------

void DetGeomDesc::deleteComponents()
{
  m_container.erase( m_container.begin(), m_container.end());
}

//----------------------------------------------------------------------------------------------------

void DetGeomDesc::deepDeleteComponents()
{
  for( auto & it : m_container ) {
    it->deepDeleteComponents();
    delete it;
  }
  clearComponents();
}

//----------------------------------------------------------------------------------------------------

void DetGeomDesc::applyAlignment( const RPAlignmentCorrectionData &t )
{
  m_rot = t.getRotationMatrix() * m_rot;
  m_trans = t.getTranslation() + m_trans;
}
