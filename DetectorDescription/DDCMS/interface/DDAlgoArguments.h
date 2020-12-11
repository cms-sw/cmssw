#ifndef DetectorDescription_DDCMS_DDAlgoArguments_h
#define DetectorDescription_DDCMS_DDAlgoArguments_h

#include "XML/XML.h"
#include "DetectorDescription/DDCMS/interface/DDXMLTags.h"
#include "DetectorDescription/DDCMS/interface/DDNamespace.h"
#include "DetectorDescription/DDCMS/interface/DDParsingContext.h"
#include "DetectorDescription/DDCMS/interface/DDRotationMatrix.h"
#include "DetectorDescription/DDCMS/interface/DDTranslation.h"

#include <map>
#include <sstream>

namespace cms {

  static constexpr long s_executed = 1l;

  constexpr unsigned int hash(const char* str, int h = 0) { return !str[h] ? 5381 : (hash(str, h + 1) * 33) ^ str[h]; }

  inline unsigned int hash(const std::string& str) { return hash(str.c_str()); }

  DDRotationMatrix makeRotation3D(double thetaX, double phiX, double thetaY, double phiY, double thetaZ, double phiZ);

  DDRotationMatrix makeRotReflect(double thetaX, double phiX, double thetaY, double phiY, double thetaZ, double phiZ);

  DDRotationMatrix makeRotation3D(DDRotationMatrix rotation, const std::string& axis, double angle);

  class DDAlgoArguments {
  public:
    DDAlgoArguments(cms::DDParsingContext&, xml_h algorithm);

    DDAlgoArguments() = delete;
    DDAlgoArguments(const DDAlgoArguments& copy) = delete;
    DDAlgoArguments& operator=(const DDAlgoArguments& copy) = delete;
    ~DDAlgoArguments() = default;

    std::string name;
    cms::DDParsingContext& context;
    xml_h element;

    std::string parentName() const;
    std::string childName() const;
    bool find(const std::string& name) const;
    template <typename T>
    T value(const std::string& name) const;
    std::string str(const std::string& nam) const;
    double dble(const std::string& nam) const;
    int integer(const std::string& nam) const;
    std::vector<double> vecDble(const std::string& nam) const;
    std::vector<float> vecFloat(const std::string& nam) const;
    std::vector<int> vecInt(const std::string& nam) const;
    std::vector<std::string> vecStr(const std::string& nam) const;
    std::string resolveValue(const std::string& value) const;

  private:
    xml_h rawArgument(const std::string& name) const;
    std::string resolved_scalar_arg(const std::string& name) const;
  };
}  // namespace cms

#endif
