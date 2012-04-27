#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "Specific.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"

#include <utility>

// Message logger.
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using DDI::Specific;

//DDBase<DDName,Specific*>::StoreT::pointer_type 
//  DDBase<DDName,Specific*>::StoreT::instance_ = 0;

DDSpecifics::DDSpecifics() : DDBase<DDName,Specific*>()
{ }


DDSpecifics::DDSpecifics(const DDName & name) : DDBase<DDName,Specific*>()
{
  prep_ = StoreT::instance().create(name);
}


DDSpecifics::DDSpecifics(const DDName & name,
                         const selectors_type & partSelections,
	      		 const DDsvalues_type & svalues,
			 bool doRegex)
 : DDBase<DDName,Specific*>()
{
  prep_ = StoreT::instance().create(name, new Specific(partSelections,svalues,doRegex));   
  typedef std::vector<std::pair<DDLogicalPart,std::pair<DDPartSelection*,DDsvalues_type*> > > strange_type;
  strange_type v;
  rep().updateLogicalPart(v);
  strange_type::iterator it = v.begin();
  for(; it != v.end(); ++it) {
    if (it->first.isDefined().second) {
      it->first.addSpecifics(it->second);
      DCOUT('C', "add specifics to LP: partsel=" << *(it->second.first) );
    }
    else {
      std::string serr("Definition of LogicalPart missing! name=");
      serr+= it->first.ddname().fullname();
      throw DDException(serr);
    }
  }
} 
    

DDSpecifics::~DDSpecifics() { }


const std::vector<DDPartSelection> & DDSpecifics::selection() const //
{ 
  return rep().selection(); 
}
  

const DDsvalues_type & DDSpecifics::specifics() const
{ 
  return rep().specifics(); 
}         

bool DDSpecifics::nodes(DDNodes & result) const 
{
   return rep().nodes(result);
}

/** node() will only work, if
    - there is only one PartSelection std::string
    - the PartSelection std::string specifies exactly one full path concatenating
      always direct children including their copy-number
    and will return (true,const DDGeoHistory&) if the std::string matches an
    expanded-part in the ExpandedView, else it will return
    (false, xxx), whereas xxx is a history which is not valid.
*/      
std::pair<bool,DDExpandedView> DDSpecifics::node() const
{
  return rep().node();
}
  	
// void DDSpecifics::clear()
// {
//  StoreT::instance().clear();
// }

			 			
std::ostream & operator<<( std::ostream  & os, const DDSpecifics & sp)
{
  DDBase<DDName,DDI::Specific*>::def_type defined(sp.isDefined());
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
  std::vector<std::string>::const_iterator it = v.begin();
  for (; it!=v.end(); ++it) {
    os << *it << std::endl;
  }
  return os;
}
