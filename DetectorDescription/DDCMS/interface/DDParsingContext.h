#ifndef DETECTOR_DESCRIPTION_DD_PARSING_CONTEXT_H
#define DETECTOR_DESCRIPTION_DD_PARSING_CONTEXT_H

#include "DD4hep/Detector.h"

#include <string>
#include <variant>
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_vector.h"
#include "tbb/concurrent_queue.h"

namespace cms {

  class DDParsingContext {
  public:
    DDParsingContext(dd4hep::Detector* det) : description(det) {}

    ~DDParsingContext() {
      rotations.clear();
      shapes.clear();
      volumes.clear();
      disabledAlgs.clear();
      namespaces.clear();
    };

    bool const ns(std::string& result) {
      std::string res;
      if (namespaces.try_pop(res)) {
        result = res;
        namespaces.emplace(res);
        return true;
      }
      return false;
    }

    std::atomic<dd4hep::Detector*> description;
    tbb::concurrent_unordered_map<std::string, dd4hep::Rotation3D> rotations;
    tbb::concurrent_unordered_map<std::string, dd4hep::Solid> shapes;
    tbb::concurrent_unordered_map<std::string, dd4hep::Volume> volumes;
    tbb::concurrent_vector<std::string> disabledAlgs;
    tbb::concurrent_queue<std::string> namespaces;

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

    std::map<std::string,
             std::variant<BooleanShape<dd4hep::UnionSolid>,
                          BooleanShape<dd4hep::SubtractionSolid>,
                          BooleanShape<dd4hep::IntersectionSolid>>>
        unresolvedShapes;

    struct CompositeMaterial {
      CompositeMaterial(const std::string& n, double f) : name(n), fraction(f) {}

      const std::string name;
      double fraction;
    };

    std::map<std::string, std::vector<CompositeMaterial>> unresolvedMaterials;

    bool geo_inited = false;

    // Debug flags
    bool debug_includes = false;
    bool debug_constants = false;
    bool debug_materials = false;
    bool debug_rotations = false;
    bool debug_shapes = false;
    bool debug_volumes = false;
    bool debug_placements = false;
    bool debug_namespaces = false;
    bool debug_visattr = false;
    bool debug_algorithms = false;
    bool debug_specpars = false;
  };
}  // namespace cms

#endif
