#include "DetectorDescription/DDCMS/interface/DDAlgoArguments.h"
#include <DD4hep/Path.h>
#include <DD4hep/Printout.h>
#include <DD4hep/Detector.h>
#include <DD4hep/Filter.h>
#include <DD4hep/Grammar.h>

#include <TClass.h>

#include <algorithm>
#include <stdexcept>

using namespace std;
using namespace cms;
using namespace dd4hep;

//
// Helpers to create 3D rotation matrix from angles
//
dd4hep::Rotation3D cms::makeRotation3D(
    double thetaX, double phiX, double thetaY, double phiY, double thetaZ, double phiZ) {
  dd4hep::Position posX(sin(thetaX) * cos(phiX), sin(thetaX) * sin(phiX), cos(thetaX));
  dd4hep::Position posY(sin(thetaY) * cos(phiY), sin(thetaY) * sin(phiY), cos(thetaY));
  dd4hep::Position posZ(sin(thetaZ) * cos(phiZ), sin(thetaZ) * sin(phiZ), cos(thetaZ));
  dd4hep::Rotation3D rotation(posX, posY, posZ);

  return rotation;
}

// makes sure that the RotationMatrix built is
// LEFT-handed coordinate system (i.e. reflected)
dd4hep::Rotation3D cms::makeRotReflect(
    double thetaX, double phiX, double thetaY, double phiY, double thetaZ, double phiZ) {
  // define 3 unit std::vectors forming the new left-handed axes
  DD3Vector x(cos(phiX) * sin(thetaX), sin(phiX) * sin(thetaX), cos(thetaX));
  DD3Vector y(cos(phiY) * sin(thetaY), sin(phiY) * sin(thetaY), cos(thetaY));
  DD3Vector z(cos(phiZ) * sin(thetaZ), sin(phiZ) * sin(thetaZ), cos(thetaZ));

  constexpr double tol = 1.0e-3;       // Geant4 compatible
  double check = (x.Cross(y)).Dot(z);  // in case of a LEFT-handed orthogonal system this must be -1
  if (abs(1. + check) > tol) {
    except("DD4CMS", "+++ FAILED to construct Rotation is not LEFT-handed!");
  }

  dd4hep::Rotation3D rotation(x.x(), y.x(), z.x(), x.y(), y.y(), z.y(), x.z(), y.z(), z.z());

  return rotation;
}

dd4hep::Rotation3D cms::makeRotation3D(dd4hep::Rotation3D rotation, const std::string& axis, double angle) {
  switch (hash(axis)) {
    case hash("x"):
      rotation = dd4hep::RotationX(angle);
      break;
    case hash("y"):
      rotation = dd4hep::RotationY(angle);
      break;
    case hash("z"):
      rotation = dd4hep::RotationZ(angle);
      break;
    default:
      throw std::runtime_error("Axis is not valid.");
  }
  return rotation;
}

namespace {

  std::string& ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r ") {
    str.erase(0, str.find_first_not_of(chars));
    return str;
  }

  std::string& rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r ") {
    str.erase(str.find_last_not_of(chars) + 1);
    return str;
  }

  std::string& trimWhitespace(std::string& str, const std::string& chars = "\t\n\v\f\r ") {
    return ltrim(rtrim(str, chars), chars);
  }

  std::string trimSVecWhitespace(std::string& str) {
    std::string res;

    vector<string_view> v = dd::split(str, ",");
    for (auto& n : v) {
      std::string ts{n.data(), n.size()};
      trimWhitespace(ts);
      res.append(ts).append(",");
    };

    res.erase(res.find_last_of(','));

    return res;
  }
}  // namespace

DDAlgoArguments::DDAlgoArguments(cms::DDParsingContext& ctxt, xml_h elt) : context(ctxt), element(elt) {
  name = xml_dim_t(element).nameStr();
}

/// Access value of rParent child node
string DDAlgoArguments::parentName() const {
  cms::DDNamespace n(context);
  xml_dim_t e(element);
  string val = n.realName(xml_dim_t(e.child(DD_CMU(rParent))).nameStr());
  return val;
}

/// Access value of child'name from the xml element
string DDAlgoArguments::childName() const {
  cms::DDNamespace n(context);
  return n.realName(value<string>("ChildName"));
}

/// Check the existence of an argument by name
bool DDAlgoArguments::find(const string& nam) const {
  for (xml_coll_t p(element, _U(star)); p; ++p) {
    string n = p.attr<string>(_U(name));
    if (n == nam) {
      return true;
    }
  }
  return false;
}

/// Access raw argument as a string by name
xml_h DDAlgoArguments::rawArgument(const string& nam) const {
  for (xml_coll_t p(element, _U(star)); p; ++p) {
    string n = p.attr<string>(_U(name));
    if (n == nam) {
      return p;
    }
  }
  except("DD4CMS", "+++ Attempt to access non-existing algorithm option %s[%s]", name.c_str(), nam.c_str());
  throw runtime_error("DDCMS: Attempt to access non-existing algorithm option.");
}

/// Access namespace resolved argument as a string by name
string DDAlgoArguments::resolved_scalar_arg(const string& nam) const {
  cms::DDNamespace ns(context);
  xml_h arg = rawArgument(nam);
  string val = arg.attr<string>(_U(value));
  return ns.realName(val);
}

string DDAlgoArguments::resolveValue(const std::string& aValue) const {
  cms::DDNamespace ns(context);
  string value(aValue);
  size_t idx = value.find('[');
  if (idx == string::npos) {
    return value;
  }

  while (idx != string::npos) {
    ++idx;
    size_t idp = value.find(':', idx);
    size_t idq = value.find(']', idx);
    if (idp == string::npos || idp > idq)
      value.insert(idx, ns.name());
    else if (idp != string::npos && idp < idq)
      value[idp] = NAMESPACE_SEP;
    idx = value.find('[', idx);
  }

  string rep;
  string& v = value;
  size_t idq;
  for (idx = v.find('[', 0); idx != string::npos; idx = v.find('[', idx + 1)) {
    idq = v.find(']', idx + 1);
    rep = v.substr(idx + 1, idq - idx - 1);
    auto r = ns.context()->description.constants().find(rep);
    if (r != ns.context()->description.constants().end()) {
      rep = "(" + r->second->type + ")";
      v.replace(idx, idq - idx + 1, rep);
    }
  }
  return value;
}

namespace {

  /// Access of raw strings as vector by argument name
  vector<string> raw_vector(const DDAlgoArguments* a, xml_h arg) {
    xml_dim_t xp(arg);
    vector<string> data;
    cms::DDNamespace ns(a->context);
    string val = xp.text();
    string nam = xp.nameStr();
    string typ = xp.typeStr();
    string numValue = xp.attr<string>(DD_CMU(nEntries));
    int num = _toDouble(numValue);
    const BasicGrammar& gr = BasicGrammar::instance<vector<string> >();

    val = trimSVecWhitespace(val);
    val = '[' + ns.realName(val) + ']';
    int res = gr.fromString(&data, val);
    if (!res) {
      except(
          "DD4CMS", "+++ VectorParam<%s>: %s -> %s [Invalid conversion:%d]", typ.c_str(), nam.c_str(), val.c_str(), res);
    } else if (num != (int)data.size()) {
      except("DD4CMS",
             "+++ VectorParam<%s>: %s -> %s [Invalid entry count: %d <> %ld]",
             typ.c_str(),
             nam.c_str(),
             val.c_str(),
             num,
             data.size());
    }
    printout(a->context.debug_algorithms ? ALWAYS : DEBUG,
             "DD4CMS",
             "+++ VectorParam<%s>: ret=%d %s -> %s",
             typ.c_str(),
             res,
             nam.c_str(),
             gr.str(&data).c_str());
    return data;
  }

  template <typename T>
  T __cnv(const string&) {
    return 0;
  }
  template <>
  double __cnv<double>(const string& arg) {
    return _toDouble(arg);
  }
  template <>
  float __cnv<float>(const string& arg) {
    return _toFloat(arg);
  }
  template <>
  long __cnv<long>(const string& arg) {
    return _toLong(arg);
  }
  template <>
  int __cnv<int>(const string& arg) {
    return _toInt(arg);
  }
  template <>
  string __cnv<string>(const string& arg) {
    return arg;
  }

  template <typename T>
  vector<T> __cnvVect(const DDAlgoArguments* a, const char* req_typ, xml_h xp) {
    cms::DDNamespace ns(a->context);
    string piece;
    string nam = xp.attr<string>(_U(name));
    string typ = xp.hasAttr(_U(type)) ? xp.attr<string>(_U(type)) : "numeric";
    string val = xp.text();
    string nValues = a->resolveValue(xp.attr<string>(DD_CMU(nEntries)));
    int num = _toInt(nValues);
    if (typ != req_typ) {
      except("DD4CMS",
             "+++ VectorParam<%s | %s>: %s -> <%s> %s [Incompatible vector-type]",
             req_typ,
             typ.c_str(),
             nam.c_str(),
             typeName(typeid(T)).c_str(),
             val.c_str());
    }
    vector<T> data;
    val = remove_whitespace(val);

    if (!val.empty())
      val += ',';
    for (size_t idx = 0, idq = val.find(',', idx); idx != string::npos && idq != string::npos;
         idx = ++idq, idq = val.find(',', idx)) {
      piece = ns.realName(val.substr(idx, idq - idx));
      T d = __cnv<T>(piece);
      data.push_back(d);
    }
    printout(a->context.debug_algorithms ? ALWAYS : DEBUG,
             "DD4CMS",
             "+++ VectorParam<%s>: %s[%d] -> %s",
             typ.c_str(),
             nam.c_str(),
             num,
             val.c_str());
    return data;
  }
}  // namespace

/// Namespace of DDCMS conversion namespace
namespace cms {

  /// Access typed argument by name
  template <typename T>
  T DDAlgoArguments::value(const string& nam) const {
    return __cnv<T>(resolved_scalar_arg(nam));
  }

  template double DDAlgoArguments::value<double>(const string& nam) const;
  template float DDAlgoArguments::value<float>(const string& nam) const;
  template long DDAlgoArguments::value<long>(const string& nam) const;
  template int DDAlgoArguments::value<int>(const string& nam) const;
  template string DDAlgoArguments::value<string>(const string& nam) const;

  /// Access typed vector<string> argument by name
  template <>
  vector<string> DDAlgoArguments::value<vector<string> >(const string& nam) const {
    return raw_vector(this, rawArgument(nam));
  }

  /// Access typed vector<double> argument by name
  template <>
  vector<double> DDAlgoArguments::value<vector<double> >(const string& nam) const {
    return __cnvVect<double>(this, "numeric", rawArgument(nam));
  }

  /// Access typed vector<float> argument by name
  template <>
  vector<float> DDAlgoArguments::value<vector<float> >(const string& nam) const {
    return __cnvVect<float>(this, "numeric", rawArgument(nam));
  }

  /// Access typed vector<long> argument by name
  template <>
  vector<long> DDAlgoArguments::value<vector<long> >(const string& nam) const {
    return __cnvVect<long>(this, "numeric", rawArgument(nam));
  }

  /// Access typed vector<int> argument by name
  template <>
  vector<int> DDAlgoArguments::value<vector<int> >(const string& nam) const {
    return __cnvVect<int>(this, "numeric", rawArgument(nam));
  }
}  // namespace cms

/// Shortcut to access string arguments
string DDAlgoArguments::str(const string& nam) const { return this->value<string>(nam); }

/// Shortcut to access double arguments
double DDAlgoArguments::dble(const string& nam) const { return this->value<double>(resolveValue(nam)); }

/// Shortcut to access integer arguments
int DDAlgoArguments::integer(const string& nam) const { return this->value<int>(nam); }

/// Shortcut to access vector<double> arguments
vector<double> DDAlgoArguments::vecDble(const string& nam) const { return this->value<vector<double> >(nam); }

/// Shortcut to access vector<float> arguments
vector<float> DDAlgoArguments::vecFloat(const string& nam) const { return this->value<vector<float> >(nam); }

/// Shortcut to access vector<int> arguments
vector<int> DDAlgoArguments::vecInt(const string& nam) const { return this->value<vector<int> >(nam); }

/// Shortcut to access vector<string> arguments
vector<string> DDAlgoArguments::vecStr(const string& nam) const { return this->value<vector<string> >(nam); }
