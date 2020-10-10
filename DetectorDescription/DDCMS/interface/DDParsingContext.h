#ifndef DETECTOR_DESCRIPTION_DD_PARSING_CONTEXT_H
#define DETECTOR_DESCRIPTION_DD_PARSING_CONTEXT_H

#include "DD4hep/Detector.h"

#include <string>
#include <variant>
#include <unordered_map>
#include <vector>

namespace cms {

  class DDParsingContext {
  public:
    DDParsingContext(dd4hep::Detector& det) : description(det) {
      assemblies.reserve(100);
      rotations.reserve(3000);
      shapes.reserve(4000);
      volumes.reserve(3000);
      unresolvedMaterials.reserve(300);
      unresolvedVectors.reserve(300);
      unresolvedShapes.reserve(1000);

      namespaces.emplace_back("");
    }

    DDParsingContext() = delete;
    DDParsingContext(const DDParsingContext&) = delete;
    DDParsingContext& operator=(const DDParsingContext&) = delete;

    ~DDParsingContext() = default;

    const std::string& ns() const { return namespaces.back(); }

    template <class TYPE>
    struct BooleanShape {
      BooleanShape(const std::string& aName, const std::string& bName, dd4hep::Transform3D t)
          : firstSolidName(aName), secondSolidName(bName), transform(t) {}

      const std::string firstSolidName;
      const std::string secondSolidName;
      dd4hep::Transform3D transform;

      dd4hep::Solid make(dd4hep::Solid firstSolid, dd4hep::Solid secondSolid) {
        return TYPE(firstSolid, secondSolid, transform);
      }
    };

    struct CompositeMaterial {
      CompositeMaterial(const std::string& n, double f) : name(n), fraction(f) {}

      const std::string name;
      double fraction;
    };

    // Debug flags
    bool debug_includes = false;
    bool debug_constants = false;
    bool debug_materials = false;
    bool debug_rotations = false;
    bool debug_shapes = false;
    bool debug_volumes = false;
    bool debug_placements = false;
    bool debug_namespaces = false;
    bool debug_algorithms = false;
    bool debug_specpars = false;

    dd4hep::Detector& description;

    std::unordered_map<std::string, dd4hep::Assembly> assemblies;
    std::unordered_map<std::string, dd4hep::Rotation3D> rotations;
    std::unordered_map<std::string, dd4hep::Solid> shapes;
    std::unordered_map<std::string, dd4hep::Volume> volumes;
    std::vector<std::string> namespaces;

    std::unordered_map<std::string, std::vector<CompositeMaterial>> unresolvedMaterials;
    std::unordered_map<std::string, std::vector<std::string>> unresolvedVectors;
    std::unordered_map<std::string,
                       std::variant<BooleanShape<dd4hep::UnionSolid>,
                                    BooleanShape<dd4hep::SubtractionSolid>,
                                    BooleanShape<dd4hep::IntersectionSolid>>>
        unresolvedShapes;
  };
}  // namespace cms

#endif
