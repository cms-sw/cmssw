#ifndef DDLogicalPart_h
#define DDLogicalPart_h

#include <iosfwd>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/Singleton.h"
#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDEnums.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"

class DDLogicalPart;
class DDMaterial;
class DDPartSelection;
class DDSolid;
class DDSpecifics;
class DDValue;
namespace DDI {
class LogicalPart;
}  // namespace DDI

std::ostream & operator<<( std::ostream &, const DDLogicalPart &);

//! A DDLogicalPart aggregates information concerning material, solid and sensitveness ...
/** ... of a unpositioned volume. DDLogicalPart provides thus an interface to its XML
    representation <LogicalPart ... </LogicalPart>. 

    An object of this class is a reference-object and thus lightweighted. 
    It can be copied by value without having a large overhead.
    Assigning to the reference-object invalidates the object which was referred to
    before. Assigning also effects all other instances of this class which 
    were created using the same value of DDName. In fact, the value of DDName
    identifies a LogicalPart uniquely.
    
    \b Current \b Restriction: Only the \a name part of DDName identifies the LogicalPart.
    
    
    <h3> General properties of reference objects:</h3>
    
    Three kind of reference objects can
    be distinguished:
    \arg \b Uninitialized reference objects (related to default constructor)
    \arg \b Initialized reference objects (related to the constructor taken DDName as argument)
    \arg \b Defined reference objects
    
    An \b unititialized reference object is somehow comparable to an anonymous
    structure. The default constructor (no arguments) is used to create it. No
    DDName was specified. It's not very useful unless you assign an \b initialized
    or \b defined reference object.
    
    An \b initialized reference object is a reference object which was created
    first using only the constructor taking a single DDName as an argument. It's
    comparable to a variable declaration with default initialization (like 
    \c std::vector \c < \c int \c > \c v). The default object is registered using
    the DDName as index in some from the user hidden registry.
    After an \b initialized reference object
    has been created it can be used (copied, assigned to, beeing assigned to, ..)
    like a built in type like \c int. As soon as an \b defined reference
    object with the same DDName is create, all already existing \b initialized
    reference object become \b defined reference object immidiately (some kind
    of one-definition-rule).
    
    A \b defined reference object is a reference object which was created using
    a constructor taking DDName and additional arguments or using appropriate
    factory functions (e.g. DDbox) returning a \b defined reference object.
    As soon as one \b defined reference object \c A1 is created with 
    a unique DDName \c N1,
    every reference object \c A2 
    created by the constructor which only take DDName \c N2
    as an argument and \c N2 == \c N1, is also a \b defined
    reference object referring to the same object than \c A1. 
    Hence \c A1 == \c A2 is true. Further any previously created
    initialized reference objects having the same DDName also become references
    to the newly created defined reference object \c A1.
    
    To find out whether an instance of a reference object is defined or not,
    \c operator \c bool can be used, i.e.
    \code
    DDLogicalPart(DDName("CMS","cms.xml")) cms;
    if (cms) {
      // cms is a defined reference object
    }
    else {
      // cms is a (default) initialized reference object
    }
    \endcode
*/    
class DDLogicalPart : public DDBase<DDName,DDI::LogicalPart*>
{
 public:  
  //! The default constructor provides an uninitialzed reference object. 
  DDLogicalPart( void ) : DDBase<DDName,DDI::LogicalPart*>(){ }   
  
  //! Creates a reference object referring to the appropriate XML specification.
  DDLogicalPart( const DDName & name );
    
  //! Registers (creates) a reference object representing a LogicalPart. 
  DDLogicalPart( const DDName & name,
		 const DDMaterial & material,
		 const DDSolid & solid,
		 DDEnums::Category cat = DDEnums::unspecified );
  
  //! Returns the categorization of the DDLogicalPart (sensitive detector element, cable, ...)  
  DDEnums::Category category( void ) const;
  
  //! Returns a reference object of the material this LogicalPart is made of 
  const DDMaterial & material( void ) const;
  
  //! Returns a reference object of the solid being the shape of this LogicalPart 
  const DDSolid & solid( void ) const; 
  
  //! Weight of the LogicalPart viewed as a component, if cached, else -1
  double & weight( void );
  
  //! returns the specific-data attached to the LogicalPart only (not to a DDExpandedNode)
  std::vector<const DDsvalues_type *> specifics( void ) const;
  
  //! returns the merged-specifics, i.e. the last specified specifics of this logical-part
  DDsvalues_type mergedSpecifics( void ) const;
  
  //! \b don't \b use, internal only /todo make it private
  void addSpecifics( const std::pair<const DDPartSelection*, const DDsvalues_type*> &);
  void removeSpecifics( const std::pair<DDPartSelection*,DDsvalues_type*> &);
  const std::vector< std::pair<const DDPartSelection*, const DDsvalues_type*> > & attachedSpecifics( void ) const;
  bool hasDDValue( const DDValue & ) const;
};

// some helpers .... (not very clean, redesign!!)
std::pair<bool, std::string>
DDIsValid( const std::string & ns, const std::string & name, std::vector<DDLogicalPart> & result, bool doRegex = true );
// std::maps name to std::vector of namespaces
// 2009-2010 re-write...FIX: Understand how this is used by DDSpecifics and FIX
typedef DDI::Singleton<std::map<std::string, std::vector<DDName> > > LPNAMES;
void DD_NC( const DDName & );

#endif
