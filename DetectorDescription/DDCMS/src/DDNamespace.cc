#include "DetectorDescription/DDCMS/interface/DDNamespace.h"
#include "DetectorDescription/DDCMS/interface/DDParsingContext.h"
#include "DD4hep/Path.h"
#include "DD4hep/Printout.h"
#include "XML/XML.h"

#include <TClass.h>
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_vector.h"

using namespace std;
using namespace cms;

DDNamespace::DDNamespace(DDParsingContext* context, xml_h element) : m_context(context) {
  dd4hep::Path path(xml_handler_t::system_path(element));
  m_name = path.filename().substr(0, path.filename().rfind('.'));
  if (!m_name.empty())
    m_name += NAMESPACE_SEP;
  m_context->namespaces.emplace(m_name);
  m_pop = true;
  dd4hep::printout(m_context->debug_namespaces ? dd4hep::ALWAYS : dd4hep::DEBUG,
                   "DD4CMS",
                   "+++ Current namespace is now: %s",
                   m_name.c_str());
  return;
}

DDNamespace::DDNamespace(DDParsingContext& ctx, xml_h element, bool) : m_context(&ctx) {
  dd4hep::Path path(xml_handler_t::system_path(element));
  m_name = path.filename().substr(0, path.filename().rfind('.'));
  if (!m_name.empty())
    m_name += NAMESPACE_SEP;
  m_context->namespaces.emplace(m_name);
  m_pop = true;
  dd4hep::printout(m_context->debug_namespaces ? dd4hep::ALWAYS : dd4hep::DEBUG,
                   "DD4CMS",
                   "+++ Current namespace is now: %s",
                   m_name.c_str());
  return;
}

DDNamespace::DDNamespace(DDParsingContext* ctx) : m_context(ctx) {
  if (!m_context->ns(m_name))
    m_name.clear();
}

DDNamespace::DDNamespace(DDParsingContext& ctx) : m_context(&ctx) {
  if (!m_context->ns(m_name))
    m_name.clear();
}

DDNamespace::~DDNamespace() {
  if (m_pop) {
    string result("");
    if (m_context->namespaces.try_pop(result))
      m_name = result;
    else
      m_name.clear();
    dd4hep::printout(m_context->debug_namespaces ? dd4hep::ALWAYS : dd4hep::DEBUG,
                     "DD4CMS",
                     "+++ Current namespace is now: %s",
                     m_name.c_str());
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

string DDNamespace::nsName(const string& nam) {
  size_t idx;
  if ((idx = nam.find(NAMESPACE_SEP)) != string::npos)
    return nam.substr(0, idx);
  return "";
}

string DDNamespace::objName(const string& nam) {
  size_t idx;
  if ((idx = nam.find(NAMESPACE_SEP)) != string::npos)
    return nam.substr(idx + 1);
  return "";
}

void DDNamespace::addConstant(const string& nam, const string& val, const string& typ) const {
  addConstantNS(prepend(nam), val, typ);
}

void DDNamespace::addConstantNS(const string& nam, const string& val, const string& typ) const {
  const string& v = val;
  const string& n = nam;
  dd4hep::printout(m_context->debug_constants ? dd4hep::ALWAYS : dd4hep::DEBUG,
                   "DD4CMS",
                   "+++ Add constant object: %-40s = %s [type:%s]",
                   n.c_str(),
                   v.c_str(),
                   typ.c_str());
  dd4hep::_toDictionary(n, v, typ);
  dd4hep::Constant c(n, v, typ);
  m_context->description.load()->addConstant(c);
}

dd4hep::Material DDNamespace::material(const string& name) const {
  return m_context->description.load()->material(realName(name));
}

void DDNamespace::addRotation(const string& name, const dd4hep::Rotation3D& rot) const {
  string n = prepend(name);
  m_context->rotations[n] = rot;
}

const dd4hep::Rotation3D& DDNamespace::rotation(const string& nam) const {
  static const dd4hep::Rotation3D s_null;
  size_t idx;
  auto i = m_context->rotations.find(nam);
  if (i != m_context->rotations.end())
    return (*i).second;
  else if (nam == "NULL")
    return s_null;
  else if (nam.find(":NULL") == nam.length() - 5)
    return s_null;
  string n = nam;
  if ((idx = nam.find(NAMESPACE_SEP)) != string::npos) {
    n[idx] = NAMESPACE_SEP;
    i = m_context->rotations.find(n);
    if (i != m_context->rotations.end())
      return (*i).second;
  }
  throw runtime_error("Unknown rotation identifier:" + nam);
}

dd4hep::Volume DDNamespace::addVolumeNS(dd4hep::Volume vol) const {
  string n = vol.name();
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
  //vol->SetName(n.c_str());
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
  cms::DDVectorsMap* registry = m_context->description.load()->extension<cms::DDVectorsMap>();
  auto it = registry->find(name);
  if (it != registry->end()) {
    std::vector<double> result;
    for (auto in : it->second)
      result.emplace_back(in);
    return result;
  } else
    return std::vector<double>();
}
