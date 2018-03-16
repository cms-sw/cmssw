#include "DD4hep/Path.h"
#include "DD4hep/Printout.h"
#include "DD4hep/Detector.h"
#include "DD4hep/BasicGrammar.h"
#include "DetectorDescription/DDCMS/interface/DDCMS.h"

#include <TClass.h>

#include <stdexcept>

using namespace std;
using namespace dd4hep;
using namespace dd4hep::cms;

#define NAMESPACE_SEP '_'

/// Create 3D rotation matrix from angles.
Rotation3D dd4hep::cms::make_rotation3D(double thetaX, double phiX,
                                        double thetaY, double phiY,
                                        double thetaZ, double phiZ)   {
  Position  posX(sin(thetaX) * cos(phiX), sin(thetaX) * sin(phiX), cos(thetaX));
  Position  posY(sin(thetaY) * cos(phiY), sin(thetaY) * sin(phiY), cos(thetaY));
  Position  posZ(sin(thetaZ) * cos(phiZ), sin(thetaZ) * sin(phiZ), cos(thetaZ));
  Rotation3D rot(posX,posY,posZ);
  return rot;
}

/// Helper: Convert the name of a placed volume into a subdetector name
string dd4hep::cms::detElementName(PlacedVolume pv)   {
  if ( pv.isValid() )  {
    string nam = pv.name();
    string nnam = nam.substr(nam.find('_')+1);
    return nnam;
    //size_t idx = nnam.rfind('_');
    //return idx == string::npos ? nnam : nnam.substr(0,idx);
  }
  except("DDCMS","++ Cannot deduce name from invalid PlacedVolume handle!");
  return string();
}

/// Initializing constructor
Namespace::Namespace(ParsingContext* ctx, xml_h element) : context(ctx)  {
  xml_dim_t elt(element);
  bool has_label = elt.hasAttr(_U(label));
  name = has_label ? elt.labelStr() : "";
  if ( !has_label )  {
    if ( !context->namespaces.empty() )  {
      name = context->namespaces.back();
    }
    printout(context->debug_namespaces ? ALWAYS : DEBUG,
             "DDCMS","+++ Current namespace is now: %s",name.c_str());
    return;
  }
  if ( has_label )   {
    size_t idx = name.find('.');
    name = name.substr(0,idx);
  }
  else  {
    Path   path(xml::DocumentHandler::system_path(element));
    name = path.filename().substr(0,path.filename().rfind('.'));
  }
  if ( !name.empty() ) name += NAMESPACE_SEP;
  context->namespaces.push_back(name);
  pop = true;
  printout(context->debug_namespaces ? ALWAYS : DEBUG,
           "DDCMS","+++ Current namespace is now: %s",name.c_str());
  return;
}

/// Initializing constructor
Namespace::Namespace(ParsingContext& ctx, xml_h element, bool ) : context(&ctx)  {
  xml_dim_t elt(element);
  bool has_label = elt.hasAttr(_U(label));
  name = has_label ? elt.labelStr() : "";
  if ( has_label )   {
    size_t idx = name.find('.');
    name = name.substr(0,idx);
  }
  else  {
    Path   path(xml::DocumentHandler::system_path(element));
    name = path.filename().substr(0,path.filename().rfind('.'));
  }
  if ( !name.empty() ) name += NAMESPACE_SEP;
  context->namespaces.push_back(name);
  pop = true;
  printout(context->debug_namespaces ? ALWAYS : DEBUG,
           "DDCMS","+++ Current namespace is now: %s",name.c_str());
  return;
}

/// Initializing constructor
Namespace::Namespace(ParsingContext* ctx) : context(ctx)  {
  name = context->namespaces.back();
}

/// Initializing constructor
Namespace::Namespace(ParsingContext& ctx) : context(&ctx)  {
  name = context->namespaces.back();
}

/// Standard destructor (Non virtual!)
Namespace::~Namespace()   {
  if ( pop )  {
    context->namespaces.pop_back();
    printout(context->debug_namespaces ? ALWAYS : DEBUG,
             "DDCMS","+++ Current namespace is now: %s",context->ns().c_str());
  }
}

/// Prepend name with namespace
string Namespace::prepend(const string& n)  const   {
  return name + n;
}

/// Resolve namespace during XML parsing
string Namespace::real_name(const string& v)  const  {
  size_t idx, idq, idp;
  string val = v;
  while ( (idx=val.find('[')) != string::npos )  {
    val.erase(idx,1);
    idp = val.find(':');
    idq = val.find(']');
    val.erase(idq,1);
    if ( idp == string::npos || idp > idq )
      val.insert(idx,name);
    else if ( idp != string::npos && idp < idq )
      val[idp] = NAMESPACE_SEP;
  }
  while ( (idx=val.find(':')) != string::npos ) val[idx]=NAMESPACE_SEP;
  return val;
}

/// Return the namespace name of a component
string Namespace::ns_name(const string& nam)    {
  size_t idx;
  if ( (idx=nam.find(':')) != string::npos )
    return nam.substr(0,idx);
  else if ( (idx=nam.find('_')) != string::npos )
    return nam.substr(0,idx);
  return "";
}

/// Strip off the namespace part of a given name
string Namespace::obj_name(const string& nam)   {
  size_t idx;
  if ( (idx=nam.find(':')) != string::npos )
    return nam.substr(idx+1);
  else if ( (idx=nam.find('_')) != string::npos )
    return nam.substr(idx+1);
  return "";
}

/// Add a new constant to the namespace
void Namespace::addConstant(const string& nam, const string& val, const string& typ)  const  {
  addConstantNS(prepend(nam), val, typ);
}

/// Add a new constant to the namespace indicated by the name
void Namespace::addConstantNS(const string& nam, const string& val, const string& typ)  const {
  const string& v = val;
  const string& n = nam;
  printout(context->debug_constants ? ALWAYS : DEBUG,
           "DDCMS","+++ Add constant object: %-40s = %s [type:%s]",
           n.c_str(), v.c_str(), typ.c_str());
  _toDictionary(n, v, typ);
  Constant c(n, v, typ);
  context->description->addConstant(c);
}

/// Access material by its namespace dressed name
Material Namespace::material(const string& nam)  const   {
  return context->description->material(real_name(nam));
}

/// Add rotation matrix to current namespace
void Namespace::addRotation(const string& nam,const Rotation3D& rot)  const  {
  string n = prepend(nam);
  context->rotations[n] = rot;
}

const Rotation3D& Namespace::rotation(const string& nam)  const   {
  static Rotation3D s_null;
  size_t idx;
  auto i = context->rotations.find(nam);
  if ( i != context->rotations.end() )
    return (*i).second;
  else if ( nam == "NULL" )
    return s_null;
  else if ( nam.find("_NULL") == nam.length()-5 )
    return s_null;
  string n = nam;
  if ( (idx=nam.find(':')) != string::npos )  {
    n[idx] = NAMESPACE_SEP;
    i = context->rotations.find(n);
    if ( i != context->rotations.end() )
      return (*i).second;
  }
  for (const auto& r : context->rotations )  {
    cout << r.first << endl;
  }
  throw runtime_error("Unknown rotation identifier:"+nam);
}

/// Add rotation matrix to current namespace
Volume Namespace::addVolumeNS(Volume vol)  const  {
  string   n = vol.name();
  Solid    s = vol.solid();
  Material m = vol.material();
  vol->SetName(n.c_str());
  context->volumes[n] = vol;
  printout(context->debug_volumes ? ALWAYS : DEBUG, "DDCMS",
           "+++ Add volume:%-38s Solid:%-26s[%-16s] Material:%s",
           vol.name(), s.name(), s.type(), m.name());
  return vol;
}

/// Add rotation matrix to current namespace
Volume Namespace::addVolume(Volume vol)  const  {
  string   n = prepend(vol.name());
  Solid    s = vol.solid();
  Material m = vol.material();
  vol->SetName(n.c_str());
  context->volumes[n] = vol;
  printout(context->debug_volumes ? ALWAYS : DEBUG, "DDCMS",
           "+++ Add volume:%-38s Solid:%-26s[%-16s] Material:%s",
           vol.name(), s.name(), s.type(), m.name());
  return vol;
}

Volume Namespace::volume(const string& nam, bool exc)  const   {
  size_t idx;
  auto i = context->volumes.find(nam);
  if ( i != context->volumes.end() )  {
    return (*i).second;
  }
  if ( (idx=nam.find(':')) != string::npos )  {
    string n = nam;
    n[idx] = NAMESPACE_SEP;
    i = context->volumes.find(n);
    if ( i != context->volumes.end() )
      return (*i).second;
  }
  if ( exc )  {
    throw runtime_error("Unknown volume identifier:"+nam);
  }
  return 0;
}

/// Add solid to current namespace as fully indicated by the name
Solid Namespace::addSolidNS(const string& nam,Solid sol)  const   {
  printout(context->debug_shapes ? ALWAYS : DEBUG, "DDCMS",
           "+++ Add shape of type %s : %s",sol->IsA()->GetName(), nam.c_str());
  context->shapes[nam] = sol.setName(nam);
  return sol;
}

/// Add solid to current namespace
Solid Namespace::addSolid(const string& nam, Solid sol)  const  {
  return addSolidNS(prepend(nam), sol);
}

Solid Namespace::solid(const string& nam)  const   {
  size_t idx;
  string n = context->namespaces.back() + nam;
  auto i = context->shapes.find(n);
  if ( i != context->shapes.end() )
    return (*i).second;
  if ( (idx=nam.find(':')) != string::npos )  {
    n = real_name(nam);
    n[idx] = NAMESPACE_SEP;
    i = context->shapes.find(n);
    if ( i != context->shapes.end() )
      return (*i).second;
  }  
  i = context->shapes.find(nam);
  if ( i != context->shapes.end() ) return (*i).second;
  throw runtime_error("Unknown shape identifier:"+nam);
}

AlgoArguments::AlgoArguments(ParsingContext& ctxt, xml_h elt)
  : context(ctxt), element(elt)
{
  name = xml_dim_t(element).nameStr();
}

/// Access value of rParent child node
string AlgoArguments::parentName()  const    {
  Namespace n(context);
  xml_dim_t e(element);
  string val = n.real_name(xml_dim_t(e.child(_CMU(rParent))).nameStr());
  return val;
}

/// Access value of child'name from the xml element
string AlgoArguments::childName()  const   {
  Namespace n(context);
  return n.real_name(value<string>("ChildName"));
}

/// Check the existence of an argument by name
bool AlgoArguments::find(const string& nam)  const   {
  for(xml_coll_t p(element,_U(star)); p; ++p)  {
    string n = p.attr<string>(_U(name));
    if ( n == nam )  {
      return true;
    }
  }
  return false;
}

/// Access raw argument as a string by name
xml_h AlgoArguments::raw_arg(const string& nam)  const   {
  for(xml_coll_t p(element,_U(star)); p; ++p)  {
    string n = p.attr<string>(_U(name));
    if ( n == nam )  {
      return p;
    }
  }
  except("DDCMS","+++ Attempt to access non-existing algorithm option %s[%s]",name.c_str(),nam.c_str());
  throw runtime_error("DDCMS: Attempt to access non-existing algorithm option.");
}

/// Access namespace resolved argument as a string by name
string AlgoArguments::resolved_scalar_arg(const string& nam)  const   {
  Namespace ns(context);
  xml_h  arg = raw_arg(nam);
  string val = arg.attr<string>(_U(value));
  return ns.real_name(val);
}

namespace {

  /// Access of raw strings as vector by argument name
  vector<string> raw_vector(const AlgoArguments* a, xml_h arg)   {
    xml_dim_t xp(arg);
    vector<string> data;
    Namespace ns(a->context);
    string val = xp.text();
    string nam = xp.nameStr();
    string typ = xp.typeStr();
    int    num = xp.attr<int>(_CMU(nEntries));
    const BasicGrammar& gr = BasicGrammar::instance<vector<string> >();

    val = '['+ns.real_name(val)+']';
    val = remove_whitespace(val);
    int res = gr.fromString(&data,val);
    if ( !res )  {
      except("DDCMS","+++ VectorParam<%s>: %s -> %s [Invalid conversion:%d]",
             typ.c_str(), nam.c_str(), val.c_str(), res);
    }
    else if ( num != (int)data.size() )  {
      except("DDCMS","+++ VectorParam<%s>: %s -> %s [Invalid entry count: %d <> %ld]",
             typ.c_str(), nam.c_str(), val.c_str(), num, data.size());
    }
    printout(DEBUG,"DDCMS","+++ VectorParam<%s>: ret=%d %s -> %s",
             typ.c_str(), res, nam.c_str(), gr.str(&data).c_str());
    return data;
  }


  template <typename T> T __cnv(const string&)       { return 0;}
  template <> double __cnv<double>(const string& arg)   { return _toDouble(arg); }
  template <> float  __cnv<float> (const string& arg)   { return _toFloat(arg); }
  template <> long   __cnv<long>  (const string& arg)   { return _toLong(arg); }
  template <> int    __cnv<int>   (const string& arg)   { return _toInt(arg);  }
  template <> string __cnv<string>(const string& arg)   { return arg;  }

  template <typename T> vector<T> __cnvVect(const AlgoArguments* a, const char* req_typ, xml_h xp)   {
    Namespace ns(a->context);
    string piece;
    string nam = xp.attr<string>(_U(name));
    string typ = xp.attr<string>(_U(type));
    string val = xp.text();
    int    num = xp.attr<int>(_CMU(nEntries));
    if ( typ != req_typ )   {
      except("DDCMS",
             "+++ VectorParam<%s | %s>: %s -> <%s> %s [Incompatible vector-type]",
             req_typ, typ.c_str(), nam.c_str(), typeName(typeid(T)).c_str(),
             val.c_str());
    }
    vector<T> data;
    val = remove_whitespace(val);
    if ( !val.empty() ) val += ',';
    for(size_t idx=0, idq=val.find(',',idx);
        idx != string::npos && idq != string::npos;
        idx=++idq, idq=val.find(',',idx))
    {
      piece = ns.real_name(val.substr(idx,idq-idx));
      T d = __cnv<T>(piece);
      data.push_back(d);
    }
    printout(DEBUG,"DDCMS","+++ VectorParam<%s>: %s[%d] -> %s",
             typ.c_str(), nam.c_str(), num, val.c_str());
    return data;
  }
}

/// Namespace for the AIDA detector description toolkit
namespace dd4hep {

  /// Namespace of DDCMS conversion namespace
  namespace cms  {

    /// Access typed argument by name
    template<typename T> T AlgoArguments::value(const string& nam)  const   {
      return __cnv<T>(resolved_scalar_arg(nam));
    }

    template double AlgoArguments::value<double>(const string& nam)  const;
    template float  AlgoArguments::value<float>(const string& nam)  const;
    template long   AlgoArguments::value<long>(const string& nam)  const;
    template int    AlgoArguments::value<int>(const string& nam)  const;
    template string AlgoArguments::value<string>(const string& nam)  const;

    /// Access typed vector<string> argument by name
    template<> vector<string> AlgoArguments::value<vector<string> >(const string& nam)  const
    {      return raw_vector(this,raw_arg(nam));                     }

    /// Access typed vector<double> argument by name
    template<> vector<double> AlgoArguments::value<vector<double> >(const string& nam)  const
    {      return __cnvVect<double>(this,"numeric",raw_arg(nam));    }

    /// Access typed vector<float> argument by name
    template<> vector<float> AlgoArguments::value<vector<float> >(const string& nam)  const
    {      return __cnvVect<float>(this,"numeric",raw_arg(nam));     }

    /// Access typed vector<long> argument by name
    template<> vector<long> AlgoArguments::value<vector<long> >(const string& nam)  const
    {      return __cnvVect<long>(this,"numeric",raw_arg(nam));      }

    /// Access typed vector<int> argument by name
    template<> vector<int> AlgoArguments::value<vector<int> >(const string& nam)  const
    {      return __cnvVect<int>(this,"numeric",raw_arg(nam));       }
  }
}

/// Shortcut to access string arguments
string AlgoArguments::str(const string& nam)  const
{  return this->value<string>(nam);                }

/// Shortcut to access double arguments
double AlgoArguments::dble(const string& nam)  const
{  return this->value<double>(nam);                }

/// Shortcut to access integer arguments
int AlgoArguments::integer(const string& nam)  const
{  return this->value<int>(nam);                   }

/// Shortcut to access vector<double> arguments
vector<double> AlgoArguments::vecDble(const string& nam)  const
{  return this->value<vector<double> >(nam);       }

/// Shortcut to access vector<int> arguments
vector<int> AlgoArguments::vecInt(const string& nam)  const
{  return this->value<vector<int> >(nam);          }

/// Shortcut to access vector<string> arguments
vector<string> AlgoArguments::vecStr(const string& nam)  const
{  return this->value<vector<string> >(nam);       }

namespace {
  bool s_debug_algorithms = false;
  vector<string> s_algorithms;
  const std::string currentAlg()  {
    static std::string s_none = "??????";
    if ( !s_algorithms.empty() ) return s_algorithms.back();
    return s_none;
  }
}

LogDebug::LogDebug(const std::string& tag_value, bool /* set_context */)  {
  level = s_debug_algorithms ? ALWAYS : DEBUG;
  s_algorithms.push_back(tag_value);
  pop = true;
}

LogDebug::LogDebug(const std::string& t) : stringstream(), tag(t)  {
  level = s_debug_algorithms ? ALWAYS : DEBUG;
}

LogDebug::~LogDebug()   {
  if ( pop )   {
    s_algorithms.pop_back();
    return;
  }
  if ( this->str().empty() ) return;
  printout(PrintLevel(level),
           currentAlg(),"%s: %s",
           tag.c_str(),this->str().c_str());
}

void LogDebug::setDebugAlgorithms(bool value)   {
  s_debug_algorithms = value;
}

LogWarn::LogWarn(const std::string& t) : LogDebug(t)  {
  level = WARNING;
}
