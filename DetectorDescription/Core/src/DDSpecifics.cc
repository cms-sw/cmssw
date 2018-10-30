#include "DetectorDescription/Core/interface/DDSpecifics.h"

#include <ostream>

#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/src/Specific.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using DDI::Specific;

DDSpecifics::DDSpecifics()
  : DDBase< DDName, std::unique_ptr<Specific> >()
{ }

DDSpecifics::DDSpecifics( const DDName & name )
  : DDBase< DDName, std::unique_ptr<Specific> >()
{
  create( name );
}

DDSpecifics::DDSpecifics(const DDName & name,
                         const std::vector<std::string> & partSelections,
	      		 const DDsvalues_type & svalues,
			 bool doRegex)
  : DDBase< DDName, std::unique_ptr<Specific> >()
{
  create( name, std::make_unique<Specific>( partSelections, svalues, doRegex ));   
  std::vector<std::pair<DDLogicalPart,std::pair<const DDPartSelection*, const DDsvalues_type*> > > v;
  rep().updateLogicalPart(v);
  for( auto& it : v ) {
    if( it.first.isDefined().second ) {
      it.first.addSpecifics( it.second );
    }
    else {
      throw cms::Exception("DDException") << "Definition of LogicalPart missing! name="
					  << it.first.ddname().fullname();
    }
  }
} 

const std::vector<DDPartSelection> &
DDSpecifics::selection() const
{ 
  return rep().selection(); 
}

const DDsvalues_type &
DDSpecifics::specifics() const
{ 
  return rep().specifics(); 
}         

/** node() will only work, if
    - there is only one PartSelection std::string
    - the PartSelection std::string specifies exactly one full path concatenating
      always direct children including their copy-number
    and will return (true,const DDGeoHistory&) if the std::string matches an
    expanded-part in the ExpandedView, else it will return
    (false, xxx), whereas xxx is a history which is not valid.
*/      
std::pair<bool, DDExpandedView>
DDSpecifics::node() const
{
  return rep().node();
}
  	
std::ostream & operator<<( std::ostream  & os, const DDSpecifics & sp)
{
  DDBase<DDName,std::unique_ptr<DDI::Specific>>::def_type defined(sp.isDefined());
  if (defined.first) {
    os << *(defined.first) << " ";
    if (defined.second) {
      sp.rep().stream(os); 
    }
    else {
      os << "* specific not defined * ";  
    }
  }  
  else {
    os << "* specific not declared * ";  
  }  
  return os;
}

std::ostream & operator<<( std::ostream & os, const std::vector<std::string> & v)
{
  for( const auto& it : v ) {
    os << it << std::endl;
  }
  return os;
}
