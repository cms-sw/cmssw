#ifndef DETECTOR_DESCRIPTION_DD_NAMESPACE_H
#define DETECTOR_DESCRIPTION_DD_NAMESPACE_H

#include "XML/XML.h"
#include "DD4hep/Objects.h"
#include "DD4hep/Shapes.h"
#include "DD4hep/Volumes.h"
#include <unordered_map>
#include <vector>

namespace cms {

  class DDParsingContext;
  using DDVectorsMap = std::unordered_map<std::string, std::vector<double>>;

  class DDNamespace {
  public:
    DDNamespace(DDParsingContext*, xml_h);
    DDNamespace(DDParsingContext&, xml_h, bool);
    DDNamespace(DDParsingContext*);
    DDNamespace(DDParsingContext&);
    ~DDNamespace();

    DDNamespace() = delete;
    DDNamespace(const DDNamespace&) = delete;
    DDNamespace& operator=(const DDNamespace&) = delete;

    std::string prepend(const std::string&) const;
    std::string realName(const std::string&) const;
    static std::string objName(const std::string&);
    static std::string nsName(const std::string&);

    template <typename T>
    T attr(xml_elt_t element, const xml_tag_t& name) const {
      std::string val = realName(element.attr<std::string>(name));
      element.setAttr(name, val);
      return element.attr<T>(name);
    }

    template <typename T>
    T attr(xml_elt_t element, const xml_tag_t& name, T defaultValue) const {
      if (element.hasAttr(name)) {
        std::string val = realName(element.attr<std::string>(name));
        element.setAttr(name, val);
        return element.attr<T>(name);
      }
      return defaultValue;
    }

    void addConstant(const std::string& name, const std::string& value, const std::string& type) const;
    void addConstantNS(const std::string& name, const std::string& value, const std::string& type) const;

    dd4hep::Material material(const std::string& name) const;

    dd4hep::Solid solid(const std::string& name) const;
    dd4hep::Solid addSolid(const std::string& name, dd4hep::Solid solid) const;
    dd4hep::Solid addSolidNS(const std::string& name, dd4hep::Solid solid) const;

    dd4hep::Assembly assembly(const std::string& name) const;
    dd4hep::Assembly addAssembly(dd4hep::Assembly asmb) const;

    dd4hep::Volume volume(const std::string& name, bool exc = true) const;
    dd4hep::Volume addVolume(dd4hep::Volume vol) const;
    dd4hep::Volume addVolumeNS(dd4hep::Volume vol) const;

    const dd4hep::Rotation3D& rotation(const std::string& name) const;
    void addRotation(const std::string& name, const dd4hep::Rotation3D& rot) const;

    DDParsingContext* const context() const { return m_context; }
    DDParsingContext* setContext() { return m_context; }

    std::string_view name() const { return m_name; }

    std::vector<double> vecDbl(const std::string& name) const;
    std::vector<float> vecFloat(const std::string& name) const;

  private:
    DDParsingContext* m_context = nullptr;
    std::string m_name;
    bool m_pop = false;
  };
}  // namespace cms

#define NAMESPACE_SEP ':'

#endif
