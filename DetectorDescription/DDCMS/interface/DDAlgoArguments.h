#ifndef DETECTOR_DESCRIPTION_DD_ALGO_ARGUMENTS_H
#define DETECTOR_DESCRIPTION_DD_ALGO_ARGUMENTS_H

#include "XML/XML.h"
#include "DD4hep/DetElement.h"
#include "DetectorDescription/DDCMS/interface/DDXMLTags.h"
#include "DetectorDescription/DDCMS/interface/DDNamespace.h"
#include "DetectorDescription/DDCMS/interface/DDParsingContext.h"

#include <map>
#include <sstream>

namespace cms
{
  using DD3Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;

  constexpr unsigned int hash( const char* str, int h = 0 )
  {
    return !str[h] ? 5381 : ( hash( str, h+1 )*33 ) ^ str[h];
  }

  inline unsigned int hash( const std::string& str )
  {
    return hash( str.c_str());
  }

  dd4hep::Rotation3D makeRotation3D( double thetaX, double phiX,
				     double thetaY, double phiY,
				     double thetaZ, double phiZ );

  dd4hep::Rotation3D makeRotReflect( double thetaX, double phiX,
				     double thetaY, double phiY,
				     double thetaZ, double phiZ );

  dd4hep::Rotation3D makeRotation3D( dd4hep::Rotation3D rotation,
				     const std::string& axis, double angle );

  class DDAlgoArguments
  {
  public:
    DDAlgoArguments( cms::DDParsingContext&, xml_h algorithm );
    
    DDAlgoArguments() = delete;
    DDAlgoArguments( const DDAlgoArguments& copy ) = delete;
    DDAlgoArguments& operator=( const DDAlgoArguments& copy ) = delete;
    ~DDAlgoArguments() = default;

    std::string name;
    cms::DDParsingContext& context;
    xml_h element;
    
    std::string parentName() const;
    std::string childName() const;
    bool find( const std::string& name ) const;
    template<typename T> T value( const std::string& name ) const;
    std::string str( const std::string& nam ) const;
    double dble( const std::string& nam ) const;
    int integer( const std::string& nam ) const;
    std::vector<double> vecDble( const std::string& nam ) const;
    std::vector<int> vecInt( const std::string& nam ) const;
    std::vector<std::string> vecStr( const std::string& nam ) const;

  private:

    xml_h rawArgument( const std::string& name ) const;
    std::string resolved_scalar_arg( const std::string& name ) const;
  };
}

#endif
