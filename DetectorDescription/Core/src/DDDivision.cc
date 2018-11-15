#include "DetectorDescription/Core/interface/DDDivision.h"

#include <ostream>

#include "DetectorDescription/Core/src/Division.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using DDI::Division;
  
std::ostream & 
operator<<( std::ostream & os, const DDDivision & div )
{
  DDBase<DDName,Division*>::def_type defined( div.isDefined());
  if( defined.first ) {
    os << *( defined.first ) << " ";
    if( defined.second ) {
      div.rep().stream( os ); 
    }
    else {
      os << "* division not defined * ";  
    }
  }  
  else {
    os << "* division not declared * ";  
  }  
  return os;
}

DDDivision::DDDivision()
  : DDBase< DDName, std::unique_ptr<DDI::Division> >()
{}

DDDivision::DDDivision( const DDName & name)
  : DDBase< DDName, std::unique_ptr<DDI::Division> >()
{
  create( name );
}

DDDivision::DDDivision( const DDName & name,
			const DDLogicalPart & parent,
			const DDAxes axis,
			const int nReplicas,
			const double width,
			const double offset )
  : DDBase< DDName, std::unique_ptr<DDI::Division> >()
{
  create( name, std::make_unique<Division>( parent, axis, nReplicas, width, offset )); 
} 

DDDivision::DDDivision( const DDName & name,
			const DDLogicalPart & parent,
			const DDAxes axis,
			const int nReplicas,
			const double offset )
{
  create( name, std::make_unique<Division>( parent, axis, nReplicas, offset )); 
}

DDDivision::DDDivision( const DDName & name,
			const DDLogicalPart & parent,
			const DDAxes axis,
			const double width,
			const double offset )
{
  create( name, std::make_unique<Division>( parent, axis, width, offset )); 
}

DDAxes
DDDivision::axis() const
{
  return rep().axis();
}

int
DDDivision::nReplicas() const
{
  return rep().nReplicas();
}

double
DDDivision::width() const
{
  return rep().width();
}

double
DDDivision::offset() const
{
  return rep().offset();
}

const DDLogicalPart &
DDDivision::parent() const
{
  return rep().parent();
}

