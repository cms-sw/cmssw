#include "DetectorDescription/DDCMS/interface/DDNamespace.h"
#include "DetectorDescription/DDCMS/interface/DDParsingContext.h"
#include "DD4hep/Path.h"
#include "DD4hep/Printout.h"
#include "XML/XML.h"

#include <TClass.h>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace cms;

DDNamespace::DDNamespace(DDParsingContext* context, xml_h element) : m_context(context) {
  dd4hep::Path path(xml_handler_t::system_path(element));
  m_name = path.filename().substr(0, path.filename().rfind('.'));
  if (!m_name.empty())
    m_name += NAMESPACE_SEP;
  m_context->namespaces.emplace_back(m_name);
  m_pop = true;
  dd4hep::printout(m_context->debug_namespaces ? dd4hep::ALWAYS : dd4hep::DEBUG,
                   "DD4CMS",
                   "+++ Current namespace is now: %s",
                   m_name.c_str());
}

DDNamespace::DDNamespace(DDParsingContext& ctx, xml_h element, bool) : m_context(&ctx) {
  dd4hep::Path path(xml_handler_t::system_path(element));
  m_name = path.filename().substr(0, path.filename().rfind('.'));
  if (!m_name.empty())
    m_name += NAMESPACE_SEP;
  m_context->namespaces.emplace_back(m_name);
  m_pop = true;
  dd4hep::printout(m_context->debug_namespaces ? dd4hep::ALWAYS : dd4hep::DEBUG,
                   "DD4CMS",
                   "+++ Current namespace is now: %s",
                   m_name.c_str());
}

DDNamespace::DDNamespace(DDParsingContext* ctx) : m_context(ctx) {
  if (!m_context->namespaces.empty())
    m_name = m_context->namespaces.back();
}

DDNamespace::DDNamespace(DDParsingContext& ctx) : m_context(&ctx) {
  if (!m_context->namespaces.empty())
    m_name = m_context->namespaces.back();
}

DDNamespace::~DDNamespace() {
  if (m_pop) {
    m_context->namespaces.pop_back();
    dd4hep::printout(m_context->debug_namespaces ? dd4hep::ALWAYS : dd4hep::DEBUG,
                     "DD4CMS",
                     "+++ Current namespace is now: %s",
                     m_context->ns().c_str());
  }
}

string DDNamespace::prepend(const string& n) const {
  if (strchr(n.c_str(), NAMESPACE_SEP) == nullptr)
    return m_name + n;
  else
    return n;
}

string DDNamespace::realName(const string& v) const {
  size_t idx, idq, idp;
  string val = v;
  while ((idx = val.find('[')) != string::npos) {
    val.erase(idx, 1);
    idp = val.find(NAMESPACE_SEP, idx);
    idq = val.find(']', idx);
    val.erase(idq, 1);
    if (idp == string::npos || idp > idq)
      val.insert(idx, m_name);
    else if (idp != string::npos && idp < idq)
      val[idp] = NAMESPACE_SEP;
  }
  return val;
}

string DDNamespace::nsName(const string& name) {
  size_t idx;
  if ((idx = name.find(NAMESPACE_SEP)) != string::npos)
    return name.substr(0, idx);
  return "";
}

string DDNamespace::objName(const string& name) {
  size_t idx;
  if ((idx = name.find(NAMESPACE_SEP)) != string::npos)
    return name.substr(idx + 1);
  return "";
}

void DDNamespace::addConstant(const string& name, const string& val, const string& type) const {
  addConstantNS(prepend(name), val, type);
}

void DDNamespace::addConstantNS(const string& name, const string& val, const string& type) const {
  const string& v = val;
  const string& n = name;
  dd4hep::printout(m_context->debug_constants ? dd4hep::ALWAYS : dd4hep::DEBUG,
                   "DD4CMS",
                   "+++ Add constant object: %-40s = %s [type:%s]",
                   n.c_str(),
                   v.c_str(),
                   type.c_str());
  dd4hep::_toDictionary(n, v, type);
  dd4hep::Constant c(n, v, type);

  m_context->description.addConstant(c);
}

dd4hep::Material DDNamespace::material(const string& name) const {
  return m_context->description.material(realName(name));
}

void DDNamespace::addRotation(const string& name, const dd4hep::Rotation3D& rot) const {
  string n = prepend(name);
  m_context->rotations[n] = rot;
}

const dd4hep::Rotation3D& DDNamespace::rotation(const string& name) const {
  static const dd4hep::Rotation3D s_null;
  size_t idx;
  auto i = m_context->rotations.find(name);
  if (i != m_context->rotations.end())
    return (*i).second;
  else if (name == "NULL")
    return s_null;
  else if (name.find(":NULL") == name.length() - 5)
    return s_null;
  string n = name;
  if ((idx = name.find(NAMESPACE_SEP)) != string::npos) {
    n[idx] = NAMESPACE_SEP;
    i = m_context->rotations.find(n);
    if (i != m_context->rotations.end())
      return (*i).second;
  }
  throw runtime_error("Unknown rotation identifier:" + name);
}

dd4hep::Volume DDNamespace::addVolumeNS(dd4hep::Volume vol) const {
  string n = prepend(vol.name());
  dd4hep::Solid s = vol.solid();
  dd4hep::Material m = vol.material();
  vol->SetName(n.c_str());
  m_context->volumes[n] = vol;
  const char* solidName = "Invalid solid";
  if (s.isValid())         // Protect against seg fault
    solidName = s.name();  // If Solid is not valid, s.name() will seg fault.
  dd4hep::printout(m_context->debug_volumes ? dd4hep::ALWAYS : dd4hep::DEBUG,
                   "DD4CMS",
                   "+++ Add volumeNS:%-38s Solid:%-26s[%-16s] Material:%s",
                   vol.name(),
                   solidName,
                   s.type(),
                   m.name());
  return vol;
}

/// Add rotation matrix to current namespace
dd4hep::Volume DDNamespace::addVolume(dd4hep::Volume vol) const {
  string n = prepend(vol.name());
  dd4hep::Solid s = vol.solid();
  dd4hep::Material m = vol.material();
  m_context->volumes[n] = vol;
  const char* solidName = "Invalid solid";
  if (s.isValid())         // Protect against seg fault
    solidName = s.name();  // If Solid is not valid, s.name() will seg fault.
  dd4hep::printout(m_context->debug_volumes ? dd4hep::ALWAYS : dd4hep::DEBUG,
                   "DD4CMS",
                   "+++ Add volume:%-38s as [%s] Solid:%-26s[%-16s] Material:%s",
                   vol.name(),
                   n.c_str(),
                   solidName,
                   s.type(),
                   m.name());
  return vol;
}

dd4hep::Assembly DDNamespace::addAssembly(dd4hep::Assembly assembly) const {
  string n = assembly.name();
  m_context->assemblies[n] = assembly;
  dd4hep::printout(
      m_context->debug_volumes ? dd4hep::ALWAYS : dd4hep::DEBUG, "DD4CMS", "+++ Add assembly:%-38s", assembly.name());
  return assembly;
}

dd4hep::Assembly DDNamespace::assembly(const std::string& name) const {
  auto i = m_context->assemblies.find(name);
  if (i != m_context->assemblies.end()) {
    return (*i).second;
  }
  if (name.front() == NAMESPACE_SEP) {
    i = m_context->assemblies.find(name.substr(1, name.size()));
    if (i != m_context->assemblies.end())
      return (*i).second;
  }
  throw runtime_error("Unknown assembly identifier:" + name);
}

dd4hep::Volume DDNamespace::volume(const string& name, bool exc) const {
  auto i = m_context->volumes.find(name);
  if (i != m_context->volumes.end()) {
    return (*i).second;
  }
  if (name.front() == NAMESPACE_SEP) {
    i = m_context->volumes.find(name.substr(1, name.size()));
    if (i != m_context->volumes.end())
      return (*i).second;
  }
  if (exc) {
    throw runtime_error("Unknown volume identifier:" + name);
  }
  return nullptr;
}

dd4hep::Solid DDNamespace::addSolidNS(const string& name, dd4hep::Solid solid) const {
  dd4hep::printout(m_context->debug_shapes ? dd4hep::ALWAYS : dd4hep::DEBUG,
                   "DD4CMS",
                   "+++ Add shape of type %s : %s",
                   solid->IsA()->GetName(),
                   name.c_str());

  auto shape = m_context->shapes.emplace(name, solid.setName(name));
  if (!shape.second) {
    m_context->shapes[name] = solid.setName(name);
  }

  return solid;
}

dd4hep::Solid DDNamespace::addSolid(const string& name, dd4hep::Solid sol) const {
  return addSolidNS(prepend(name), sol);
}

dd4hep::Solid DDNamespace::solid(const string& nam) const {
  size_t idx;
  string n = m_name + nam;
  auto i = m_context->shapes.find(n);
  if (i != m_context->shapes.end())
    return (*i).second;
  if ((idx = nam.find(NAMESPACE_SEP)) != string::npos) {
    n = realName(nam);
    n[idx] = NAMESPACE_SEP;
    i = m_context->shapes.find(n);
    if (i != m_context->shapes.end())
      return (*i).second;
  }
  i = m_context->shapes.find(nam);
  if (i != m_context->shapes.end())
    return (*i).second;
  // Register a temporary shape
  auto tmpShape = m_context->shapes.emplace(nam, dd4hep::Solid(nullptr));
  return (*tmpShape.first).second;
}

std::vector<double> DDNamespace::vecDbl(const std::string& name) const {
  cms::DDVectorsMap* registry = m_context->description.extension<cms::DDVectorsMap>();
  auto it = registry->find(name);
  if (it != registry->end()) {
    return {begin(it->second), end(it->second)};
  } else
    return std::vector<double>();
}

std::vector<float> DDNamespace::vecFloat(const std::string& name) const {
  cms::DDVectorsMap* registry = m_context->description.extension<cms::DDVectorsMap>();
  auto it = registry->find(name);
  if (it != registry->end()) {
    std::vector<float> result;
    std::transform(
        begin(it->second), end(it->second), std::back_inserter(result), [](double i) -> float { return (float)i; });
    return result;
  } else
    return std::vector<float>();
}

std::string DDNamespace::noNamespace(const std::string& fullName) const {
  std::string result(fullName);
  auto n = result.find(':');
  if (n == std::string::npos) {
    return result;
  } else {
    return result.substr(n + 1);
  }
}
