#include <ostream>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/src/Material.h"
#include "DetectorDescription/Core/interface/DDUnits.h"
#include "FWCore/Utilities/interface/Exception.h"

using DDI::Material;
using namespace dd::operators;

DDMaterial::DDMaterial() : DDBase< DDName, std::unique_ptr<Material>>() { }

/**
   If a DDMaterial with \a name was already defined, this constructor creates a
   reference object to the defined material. Otherwise it creates a (default)
   initialized reference-object to a material with DDName \a name. 
      
   For further details concerning the usage of reference-objects refere
   to the documentation of DDLogicalPart.
*/
DDMaterial::DDMaterial( const DDName & name )
  : DDBase< DDName, std::unique_ptr<Material>>()
{ 
  create( name );
}

/** 
   \arg \c z atomic number
   \arg \c a atomic mass
   \arg \c density density
      
   Example:
   \code
     DDMaterial hydrogen("Hydrogen", 
                          double z=1, 
	                  double a=1.1*g/mole, 
                          density=2*g/cm3);
   \endcode  
*/
DDMaterial::DDMaterial( const DDName & name, double z, double a, double d )
  : DDBase< DDName, std::unique_ptr<Material>>()
{ 
  create( name, std::make_unique<Material>( z, a, d ));
}

/** 
   For a mixture material it is sufficient to specify the \a density of the
   mixture (in addition to the \a name). The compounds are added by
   using the addMaterial(..) method. 
     
   The result of this constructor is a reference-object. One can use the
   mixture material immidiately (without specifying the compund materials).
   Compound materials can be added at any later stage.
      
   For further details concerning the usage of reference-objects refere
   to the documentation of DDLogicalPart.      
*/
DDMaterial::DDMaterial( const DDName & name, double density )
  : DDBase< DDName, std::unique_ptr<Material>>()
{ 
  create( name, std::make_unique<Material>( 0, 0, density ));
}

/** 
  The fraction-masses of all compounds must sum up to 1
*/
int
DDMaterial::addMaterial( const DDMaterial & m, double fm )
{  
  if( m.ddname() == ddname()) {
    throw cms::Exception("DDException") << "DDMaterial::addMaterial(..): name-clash\n        trying to add material " << m << " to itself! ";
  }  
  rep().addMaterial( m, fm );
  return rep().noOfConstituents();
}

int
DDMaterial::noOfConstituents() const
{
   return rep().noOfConstituents();
}

DDMaterial::FractionV::value_type DDMaterial::constituent( int i ) const 
{ 
  return rep().constituent( i );
}

double
DDMaterial::a() const
{
  return rep().a(); 
}

double
DDMaterial::z() const
{
  return rep().z(); 
}

double DDMaterial::density() const
{
  return rep().density(); 
}

namespace {
  std::ostream &doStream(std::ostream & os, const DDMaterial & mat, int level)
  {
    ++level; 
    if (mat) {
      os << '[' << mat.name() <<']' << " z=" << mat.z()
	 << " a=" << CONVERT_TO( mat.a(), g_per_mole ) << "*g/mole"
	 << " d=" << CONVERT_TO( mat.density(), g_per_cm3 ) << "*g/cm3";
      std::string s(2*level,' ');
      for (int i=0; i<mat.noOfConstituents(); ++i) {
         DDMaterial::FractionV::value_type f = mat.constituent(i);
         os << std::endl << s << i+1 << " : fm=" << f.second
                    << " : ";
         doStream(os, f.first, level);
      }
    }
    else
      os << "* material not declared * ";
    --level;
    return os;
  }
}

std::ostream & operator<<(std::ostream & os, const DDMaterial & mat)
{ 
  return doStream( os, mat, 0 );
}
