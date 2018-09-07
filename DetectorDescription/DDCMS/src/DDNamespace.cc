#include "DetectorDescription/DDCMS/interface/DDNamespace.h"
#include "DetectorDescription/DDCMS/interface/DDParsingContext.h"
#include "DD4hep/Path.h"
#include "DD4hep/Printout.h"
#include "XML/XML.h"

#include <TClass.h>

using namespace std;
using namespace cms;

DDNamespace::DDNamespace( DDParsingContext* context, xml_h element )
  : context( context )
{
  xml_dim_t elt( element );
  bool has_label = elt.hasAttr(_U(label));
  m_name = has_label ? elt.labelStr() : "";
  if( !has_label ) {
    if( !context->namespaces.empty()) {
      m_name = context->namespaces.back();
    }
    dd4hep::printout( context->debug_namespaces ? dd4hep::ALWAYS : dd4hep::DEBUG,
	      "MyDDCMS", "+++ Current namespace is now: %s", m_name.c_str());
    return;
  }
  if( has_label ) {
    size_t idx = m_name.find('.');
    m_name = m_name.substr( 0, idx );
  }
  else {
    dd4hep::Path path( xml_handler_t::system_path( element ));
    m_name = path.filename().substr( 0, path.filename().rfind('.'));
  }
  if ( !m_name.empty()) m_name += NAMESPACE_SEP;
  context->namespaces.push_back( m_name );
  m_pop = true;
  dd4hep::printout( context->debug_namespaces ? dd4hep::ALWAYS : dd4hep::DEBUG,
	    "MyDDCMS","+++ Current namespace is now: %s", m_name.c_str());
  return;
}

DDNamespace::DDNamespace( DDParsingContext& ctx, xml_h element, bool )
  : context(&ctx)
{
  xml_dim_t elt(element);
  bool has_label = elt.hasAttr(_U(label));
  m_name = has_label ? elt.labelStr() : "";
  if( has_label ) {
    size_t idx = m_name.find('.');
    m_name = m_name.substr(0,idx);
  }
  else {
    dd4hep::Path path( xml_handler_t::system_path( element ));
    m_name = path.filename().substr( 0, path.filename().rfind('.'));
  }
  if( !m_name.empty()) m_name += NAMESPACE_SEP;
  context->namespaces.push_back( m_name );
  m_pop = true;
  dd4hep::printout( context->debug_namespaces ? dd4hep::ALWAYS : dd4hep::DEBUG,
	    "MyDDCMS","+++ Current namespace is now: %s", m_name.c_str());
  return;
}

DDNamespace::DDNamespace( DDParsingContext* ctx )
  : context( ctx )
{
  m_name = context->namespaces.back();
}

DDNamespace::DDNamespace( DDParsingContext& ctx )
  : context( &ctx )
{
  m_name = context->namespaces.back();
}

DDNamespace::~DDNamespace() {
  if( m_pop ) {
    context->namespaces.pop_back();
    dd4hep::printout( context->debug_namespaces ? dd4hep::ALWAYS : dd4hep::DEBUG,
	      "MyDDCMS","+++ Current namespace is now: %s", context->ns().c_str());
  }
}

string
DDNamespace::prepend( const string& n ) const
{
  return m_name + n;
}

string
DDNamespace::realName( const string& v ) const
{
  size_t idx, idq, idp;
  string val = v;
  while(( idx = val.find('[')) != string::npos )
  {
    val.erase( idx, 1 );
    idp = val.find( NAMESPACE_SEP, idx );
    idq = val.find( ']', idx );
    val.erase( idq, 1 );
    if( idp == string::npos || idp > idq )
      val.insert( idx, m_name );
    else if ( idp != string::npos && idp < idq )
      val[idp] = NAMESPACE_SEP;
  }
  return val;
}

string
DDNamespace::nsName( const string& nam )
{
  size_t idx;
  if(( idx = nam.find( NAMESPACE_SEP )) != string::npos )
    return nam.substr( 0, idx );
  return "";
}

string
DDNamespace::objName( const string& nam )
{
  size_t idx;
  if(( idx = nam.find( NAMESPACE_SEP )) != string::npos )
    return nam.substr( idx + 1 );
  return "";
}

void
DDNamespace::addConstant( const string& nam, const string& val, const string& typ ) const
{
  addConstantNS( prepend( nam ), val, typ );
}

void
DDNamespace::addConstantNS( const string& nam, const string& val, const string& typ ) const
{
  const string& v = val;
  const string& n = nam;
  dd4hep::printout( context->debug_constants ? dd4hep::ALWAYS : dd4hep::DEBUG,
	    "MyDDCMS","+++ Add constant object: %-40s = %s [type:%s]",
	    n.c_str(), v.c_str(), typ.c_str());
  dd4hep::_toDictionary( n, v, typ );
  dd4hep::Constant c( n, v, typ );
  context->description->addConstant( c );
}

void
DDNamespace::addVector( const string& name, const vector<double>& value ) const
{
  const vector<double>& v = value;
  const string& n = name;
  dd4hep::printout( context->debug_constants ? dd4hep::ALWAYS : dd4hep::DEBUG,
		    "MyDDCMS","+++ Add constant object: %-40s = %s ",
		    n.c_str(), "vector<double>");
  context->addVector( n, v );
}

dd4hep::Material
DDNamespace::material( const string& name ) const
{
  return context->description->material( realName( name ));
}

void
DDNamespace::addRotation( const string& name, const dd4hep::Rotation3D& rot ) const
{
  string n = prepend( name );
  context->rotations[n] = rot;
}

const dd4hep::Rotation3D&
DDNamespace::rotation( const string& nam ) const
{
  static dd4hep::Rotation3D s_null;
  size_t idx;
  auto i = context->rotations.find( nam );
  if( i != context->rotations.end())
    return (*i).second;
  else if( nam == "NULL" )
    return s_null;
  else if( nam.find(":NULL") == nam.length() - 5 )
    return s_null;
  string n = nam;
  if(( idx = nam.find( NAMESPACE_SEP )) != string::npos )
  {
    n[idx] = NAMESPACE_SEP;
    i = context->rotations.find(n);
    if( i != context->rotations.end() )
      return (*i).second;
  }
  for( const auto& r : context->rotations )  {
    cout << r.first << endl;
  }
  throw runtime_error("Unknown rotation identifier:"+nam);
}

dd4hep::Volume
DDNamespace::addVolumeNS( dd4hep::Volume vol ) const
{
  string   n = vol.name();
  dd4hep::Solid    s = vol.solid();
  dd4hep::Material m = vol.material();
  vol->SetName(n.c_str());
  context->volumes[n] = vol;
  dd4hep::printout(context->debug_volumes ? dd4hep::ALWAYS : dd4hep::DEBUG, "MyDDCMS",
           "+++ Add volume:%-38s Solid:%-26s[%-16s] Material:%s",
           vol.name(), s.name(), s.type(), m.name());
  return vol;
}

/// Add rotation matrix to current namespace
dd4hep::Volume
DDNamespace::addVolume( dd4hep::Volume vol ) const
{
  string   n = prepend(vol.name());
  dd4hep::Solid    s = vol.solid();
  dd4hep::Material m = vol.material();
  vol->SetName(n.c_str());
  context->volumes[n] = vol;
  dd4hep::printout(context->debug_volumes ? dd4hep::ALWAYS : dd4hep::DEBUG, "MyDDCMS",
           "+++ Add volume:%-38s Solid:%-26s[%-16s] Material:%s",
           vol.name(), s.name(), s.type(), m.name());
  return vol;
}

dd4hep::Volume
DDNamespace::volume( const string& nam, bool exc ) const
{
  size_t idx;
  auto i = context->volumes.find(nam);
  if ( i != context->volumes.end() )  {
    return (*i).second;
  }
  if(( idx = nam.find( NAMESPACE_SEP )) != string::npos )  {
    string n = nam;
    n[idx] = NAMESPACE_SEP;
    i = context->volumes.find( n );
    if( i != context->volumes.end())
      return (*i).second;
  }
  if( exc )  {
    throw runtime_error("Unknown volume identifier:"+nam);
  }
  return nullptr;
}

dd4hep::Solid
DDNamespace::addSolidNS( const string& nam, dd4hep::Solid sol ) const
{
  dd4hep::printout(context->debug_shapes ? dd4hep::ALWAYS : dd4hep::DEBUG, "MyDDCMS",
           "+++ Add shape of type %s : %s",sol->IsA()->GetName(), nam.c_str());
  context->shapes[nam] = sol.setName(nam);

  return sol;
}

dd4hep::Solid
DDNamespace::addSolid( const string& nam, dd4hep::Solid sol ) const
{
  return addSolidNS( prepend(nam), sol );
}

dd4hep::Solid
DDNamespace::solid( const string& nam ) const
{
  size_t idx;
  string n = context->namespaces.back() + nam;
  auto i = context->shapes.find( n );
  if( i != context->shapes.end())
    return (*i).second;
  if(( idx = nam.find( NAMESPACE_SEP )) != string::npos ) {
    n = realName( nam );
    n[idx] = NAMESPACE_SEP;
    i = context->shapes.find( n );
    if ( i != context->shapes.end() )
      return (*i).second;
  }  
  i = context->shapes.find(nam);
  if( i != context->shapes.end()) return (*i).second;
  throw runtime_error( "Unknown shape identifier:" + nam );
}
