#include "DetectorDescription/Core/interface/DDDivision.h"
#include "Division.h"

#include "DetectorDescription/Base/interface/DDdebug.h"

using DDI::Division;
  
std::ostream & 
operator<<(std::ostream & os, const DDDivision & div)
{
  DDBase<DDName,Division*>::def_type defined(div.isDefined());
  if (defined.first) {
    os << *(defined.first) << " ";
    if (defined.second) {
      div.rep().stream(os); 
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

DDDivision::DDDivision() : DDBase<DDName, DDI::Division*>()
{ }

DDDivision::DDDivision( const DDName & name) : DDBase<DDName,DDI::Division*>()
{
  prep_ = StoreT::instance().create(name);
}

DDDivision::DDDivision( const DDName & name,
			const DDLogicalPart & parent,
			const DDAxes axis,
			const int nReplicas,
			const double width,
			const double offset ) :  DDBase<DDName,DDI::Division*>()
{
  DCOUT('C', "create Division name=" << name << " parent=" << parent.name() << " axis=" << DDAxesNames::name(axis) << " nReplicas=" << nReplicas << " width=" << width << " offset=" << offset);
  prep_ = StoreT::instance().create(name, new Division(parent, axis, nReplicas, width, offset)); 
} 

DDDivision::DDDivision( const DDName & name,
			const DDLogicalPart & parent,
			const DDAxes axis,
			const int nReplicas,
			const double offset )
{
  DCOUT('C', "create Division name=" << name << " parent=" << parent.name() << " axis=" << DDAxesNames::name(axis) << " nReplicas=" << nReplicas << " offset=" << offset);
  prep_ = StoreT::instance().create(name, new Division(parent, axis, nReplicas, offset)); 
}

DDDivision::DDDivision( const DDName & name,
			const DDLogicalPart & parent,
			const DDAxes axis,
			const double width,
			const double offset )
{
  DCOUT('C', "create Division name=" << name << " parent=" << parent.name() << " axis=" << DDAxesNames::name(axis) << " width=" << width << " offset=" << offset);
  prep_ = StoreT::instance().create(name, new Division(parent, axis, width, offset)); 
}

DDAxes DDDivision::axis() const
{
  return rep().axis();
}

int DDDivision::nReplicas() const
{
  return rep().nReplicas();
}

double DDDivision::width() const
{
  return rep().width();
}

double DDDivision::offset() const
{
  return rep().offset();
}

const DDLogicalPart & DDDivision::parent() const
{
  return rep().parent();
}

