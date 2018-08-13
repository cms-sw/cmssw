#include "DetectorDescription/Core/interface/DDLogicalPart.h"

#include <ostream>

#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/src/LogicalPart.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class DDValue;

void
DD_NC( const DDName & n )
{
  auto & ns = LPNAMES::instance()[n.name()];

  bool alreadyIn( false );
  for( const auto& p : ns )
  {
    if( p.ns() == n.ns())
    {
      alreadyIn = true;
      break;
    } 
  }
  if( !alreadyIn )
  {
    ns.emplace_back( n );
  }  
}

std::ostream & 
operator<<( std::ostream & os, const DDLogicalPart & part )
{
  DDBase<DDName,DDI::LogicalPart*>::def_type defined( part.isDefined());
  if( defined.first )
  {
    os << *(defined.first) << " ";
    if( defined.second )
    {
      part.rep().stream( os ); 
    }
    else
    {
      os << "* logicalpart not defined * ";  
    }
  }  
  else
  {
    os << "* logicalpart not declared * ";  
  }  
  return os;
}

// =================================================================================

/** 
   In order to use an uninitialized reference object one has to assign
   to it an initialized object of the same class.
      
   Example: 
   \code 
      DDLogicalPart world;  // uninitialized (anonymous) reference object
      world = DDLogicalPart(DDName("CMS","cms.xml"));  
      // now world refers to an initialized object, which in turn is not
      // necessarily defined yet.
   \endcode
*/
// now inlined... 

/** One has to distinguish two cases:
  
      \par The object which should be referred to has already been defined ...
       ... using the constructor:
       \code DDLogicalPart(const DDName &, const DDMaterial, 
       const DDSolid, bool sens)\endcode
       This will be the case for example after XML has been parsed. The XML parser calls the constructor 
       described below and thus registers a new object using DDName to identify it uniquely. The 
       using this constructor one fetches the reference object. Assigning to this reference object
       invalidates the object being refered before (and redirects all reference objects using the same value
       of their DDName already in use to the newly assigned reference object).
      \par The object which should be referred to has not yet been defined 
       In this case this constructor registeres a valid object. But this object is not
       yet defined (i.e. no material nor a solid has been attached to it). Nevertheless
       the reference object can be used (copied ...) everywhere. If, at a later stage,
       a defined reference object with the same DDName is created, all already existing reference objects
       become references to this newly created reference object (one definition rule).
       
       Example:
       
       \code 
       ... // code for DDMaterial (material) definition and DDSolid (solid) defintion goes here
       DDName detName("Detector","logparts"); // define a unique  name
       DDLogicalPart detDeclaration(detName); // detName-corresponding object not defined yet!
       std::vector<DDLogicalPart> vec; 
       vec.emplace_back(det);  // use reference object in a std::vector
       // now define ad detName-corresponding object (it will be internally registered) 
       DDLogicalPart detDefinition(detName, material, solid, false); 
       // now also vec[0] automatically becomes a reference to detDefinition!
       // both got  
       \endcode 
*/        
DDLogicalPart::DDLogicalPart( const DDName & name )
  : DDBase< DDName, std::unique_ptr<DDI::LogicalPart> >()
{ 
  create( name );
  DD_NC( name );
}

/** 
   An object representing a logicalpart uniquely identified by its DDName \a name
   will be created. If reference objects of the same \a name already exist, they
   will refere to the newly created object. DDMaterial \a material and DDSolid \a solid
   are themselves reference objects to a material and solid specification. The need
   not be defined yet as long as they were constructed using unique DDName-objects.
      
   This constructor is intended to be called by the \b XML \b parsing software, not
   by the DDD user. It decouples the input technologies (i.e. XML) and forms the transition
   to the runtime DDD representation.
   However, it could also be used for 'programming' a detector description.
*/    
DDLogicalPart::DDLogicalPart(const DDName & ddname,
                             const DDMaterial & material,
                             const DDSolid & solid,
		             DDEnums::Category cat) 
  : DDBase< DDName, std::unique_ptr<DDI::LogicalPart> >() 
{ 
  create( ddname, std::make_unique<DDI::LogicalPart>( material, solid, cat ));
   DD_NC(ddname);
}

DDEnums::Category DDLogicalPart::category() const
{ 
  return rep().category(); 
}

const DDMaterial & DDLogicalPart::material() const 
{
  return rep().material();
}  

const DDSolid & DDLogicalPart::solid() const
{
  return rep().solid();
}

/**
 The method will only return specific data attached to a DDLogicalPart. 
 If DDL-XML is used to define specific data, the path-attribute of <PartSelector> addressing only
 LogicalParts only consists of a "//" and the name of the LogicalPart (or a regexp for the name):
 \code
   <SpecPar name="Color">
    <PartSelector path="//BarrelDetector"/>
    <PartSelector path="//ForwardSector1.*Cable."/>
    <Parameter name="rgb" value="0.5"/>
    <Parameter name="rgb" value="0.1+0.2"/>
    <Parameter name="rgb" value="[colors:blue1]/>
    <Parameter name="visible" value="true"/>
   </SpecPar>
 \endcode 
 The above XML assigns specific data to a DDLogicalPart with name "BarrelDetector" and to all
 DDLogicalParts whose names match the regexp "ForwardSector1.*Cable.", e.g. "ForwardSector123abCable7"
 Two parameters are attached as specific data: "rgb" - a std::vector of three values, and 
 "visible" - a std::vector of one value.
 
 The method DDLogicalPart::specifics() now returns a std::vector<const DDsvalues_type *> V which
 correspond to these two values. Every entry in V comes from a different <SpecPar> tag.
 In our example above, V would have size 1, e.g. V.size() == 1.
 
 A <Paramter> is std::mapped to DDValue. 'value' of <Parameter> is kept as a std::string and as a double.
 If the std::string does not evaluate correctly to double, 0 is the assigned.
 
 Here's the code to retrieve the 'rgb' Parameter:
 \code
   void someFunc(DDLogicalPart aLp) {

     // want to know, whether the specific parameter 'Color' is attached to aLp

     // each <SpecPar> for this LogicalPart will create one entry in the result_type std::vector
     // each entry in the result_type std::vector contains all Paramters defined in one SpecPar-tag
     // We assume now, that we have only one SpecPar ...
     typedef std::vector<const DDsvalues_type *> result_type;
     result_type result = aLp.specifics();
     if (result.size()==1) {
       DDValue val("Color");
       bool foundIt=false;
       foundIt = DDfetch(result[0],val) // DDfetch is a utility function to retrieve values from a DDsvalues_type*
      if (foundIt) { // val contains the result
        const std::vector<std::string> & strVec = val.std::string();
       // strVec[0] == "0.5"
       // strVec[1] == "0.1+0.2"
       const std::vector<double> & dblVec = val.doubles(); 
       // dblVec[0] == double value of Constant 'red' 0.5
... 
      // do something here ...
     }
   } 
 \endcode
*/
std::vector<const DDsvalues_type *> DDLogicalPart::specifics() const
{
  std::vector<const DDsvalues_type*> result;
  rep().specificsV(result);
  return result;
}

DDsvalues_type DDLogicalPart::mergedSpecifics() const
{
  DDsvalues_type  result;
  rep().mergedSpecificsV(result);
  return result;
}  

// for internal use only
void
DDLogicalPart::addSpecifics(const std::pair<const DDPartSelection*, const DDsvalues_type*> & s)
{
   rep().addSpecifics(s);
}

void
DDLogicalPart::removeSpecifics(const std::pair<DDPartSelection*,DDsvalues_type*> & s)
{
   rep().removeSpecifics(s);
}

bool
DDLogicalPart::hasDDValue(const DDValue & v) const
{
  return rep().hasDDValue(v);
}

// finds out whether a DDLogicalPart is registered & already valid (initialized)
// - returns (true,""), if so; result holds the corresponding DDLogicalPart
// - returns (false,"some status message") otherwise
// - nm corresponds to a regular expression, but will be anchored ( ^regexp$ )
// - ns corresponds to a regular expression, but will be anchored ( ^regexp$ )
#include <regex.h>
#include <cstddef>

namespace
{
  struct Regex
  {
    explicit Regex( const std::string & s )
      : m_ok( false ),
	me( s )
      {
	size_t p = me.find(".");
	m_ok = p != std::string::npos;
	if( m_ok )
	{
	  if( p > 0 )
	  {
	    m_range.first = me.substr( 0, p );
	    m_range.second = m_range.first + "{"; // '{' is 'z'+1
	  }
	  me = "^" + me + "$";
	  regcomp( &m_regex, me.c_str(), 0 );
	}
      } 

    ~Regex( void ) { if( m_ok ) regfree( &m_regex ); }

    bool empty( void ) const { return me.empty(); }

    bool notRegex( void ) const { return !m_ok; }

    const std::string & value( void ) const { return me;}

    bool match( const std::string & s ) const {
      if( m_ok )
	return !regexec( &m_regex, s.c_str(), 0, nullptr, 0 );
      else
	return me == s;
    }

    const std::pair< std::string, std::string> & range( void ) const { return m_range; }
  private:
    bool m_ok;
    regex_t m_regex;
    std::string me;
    // range of me in a collating sequence
    std::pair<std::string, std::string> m_range;
  };
}

std::pair<bool, std::string>
DDIsValid( const std::string & ns, const std::string & nm, std::vector<DDLogicalPart> & result, bool doRegex )
{
  if( !doRegex )
  {
    DDName ddnm( nm, ns );
    result.emplace_back( DDLogicalPart( ddnm ));
    return std::make_pair( true, "" );
  }
  std::string status;
  Regex aRegex( nm );
  Regex aNsRegex( ns );
  bool emptyNs = aNsRegex.empty();
  
  // THIS IS THE SLOW PART: I have to compare every namespace & name of every
  // logical part with a regex-comparison .... a linear search always through the
  // full range of logical parts!!!!
  /*
    Algorithm description:
    x. empty nm and ns argument of method means: use all matching regex ^.*$
    a. iterate over all logical part names, match against regex for names
    b. iterate over all namespaces of names found in a & match against regex for namespaces   
  */
  LPNAMES::value_type::const_iterator bn(LPNAMES::instance().begin()),
                                      ed(LPNAMES::instance().end());
  typedef std::vector< LPNAMES::value_type::const_iterator> Candidates;
  Candidates candidates;
  if ( aRegex.notRegex() ) {
    LPNAMES::value_type::const_iterator it = LPNAMES::instance().find(aRegex.value());
    if (it!=ed) candidates.emplace_back(it);
  }
  else {
    if ( !aRegex.range().first.empty()) {
      bn =  LPNAMES::instance().lower_bound(aRegex.range().first);
      ed =  LPNAMES::instance().upper_bound(aRegex.range().second);
    }
    for (LPNAMES::value_type::const_iterator it=bn; it != ed; ++it)
      if(aRegex.match(it->first)) candidates.emplace_back(it);
  }
  for (const auto & it : candidates) {
    //if (doit)  edm::LogInfo("DDLogicalPart") << "rgx: " << aName << ' ' << it->first << ' ' << doit << std::endl;
    std::vector<DDName>::size_type sz = it->second.size(); // no of 'compatible' namespaces
    if ( emptyNs && (sz==1) ) { // accept all logical parts in all the namespaces
      result.emplace_back(it->second[0]);
      //std::vector<DDName>::const_iterator nsIt(it->second.begin()), nsEd(it->second.end());
      //for(; nsIt != nsEd; ++nsIt) {
      //   result.emplace_back(DDLogicalPart(*nsIt));
      //   edm::LogInfo("DDLogicalPart") << "DDD-WARNING: multiple namespaces match (in SpecPars PartSelector): " << *nsIt << std::endl;
      //}
    }
    else if ( !emptyNs ) { // only accept matching namespaces
      std::vector<DDName>::const_iterator nsit(it->second.begin()), nsed(it->second.end());
      for (; nsit !=nsed; ++nsit) {
	//edm::LogInfo("DDLogicalPart") << "comparing " << aNs << " with " << *nsit << std::endl;
	bool another_doit = aNsRegex.match(nsit->ns());
	if ( another_doit ) {
	  //temp.emplace_back(std::make_pair(it->first,*nsit));
	  result.emplace_back(DDLogicalPart(*nsit));
	}
      }
    }
    else { // emtpyNs and sz>1 -> error, too ambigous
      std::string message = "DDLogicalPart-name \"" + it->first +"\" matching regex \""
	+ nm + "\" has been found at least in following namespaces:\n";
      std::vector<DDName>::const_iterator vit = it->second.begin();
      for(; vit != it->second.end(); ++vit) {
	message += vit->ns();
	message += " "; 
      } 
	message += "\nQualify the name with a regexp for the namespace, i.e \".*:name-regexp\" !";
	return std::make_pair(false,message);        
    }
  }
  bool flag=true;    
  std::string message;
  
  // check whether the found logical-parts are also defined (i.e. have material, solid ...)
  if (!result.empty()) {
    std::vector<DDLogicalPart>::const_iterator lpit(result.begin()), lped(result.end());
    for (; lpit != lped; ++lpit) { 
      // std::cout << "VI- " << std::string(lpit->name()) << std::endl;
      if (!lpit->isDefined().second) {
         message = message + "LogicalPart " + lpit->name().fullname() + " not (yet) defined!\n";
	 flag = false;
      }
    }
  }
  else {
    flag = false;
    message = "No regex-match for namespace=" + ns + "  name=" + nm + "\n";
  }

  return std::make_pair(flag,message);
}

const std::vector< std::pair<const DDPartSelection*, const DDsvalues_type*> > & 
DDLogicalPart::attachedSpecifics( void ) const
{
  return rep().attachedSpecifics();
}
