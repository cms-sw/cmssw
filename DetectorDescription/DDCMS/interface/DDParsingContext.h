#ifndef DETECTOR_DESCRIPTION_DD_PARSING_CONTEXT_H
#define DETECTOR_DESCRIPTION_DD_PARSING_CONTEXT_H

#include "DD4hep/Detector.h"

namespace cms  {

  class DDParsingContext {

  public:
    using VecDouble = std::vector<double>;
    
    DDParsingContext( dd4hep::Detector* det )
      : description( det ) {
      namespaces.push_back("");
    }

    ~DDParsingContext() = default;
    const std::string& ns() const { return namespaces.back(); }
    void addVector( const std::string& name, const VecDouble& value );
    
    dd4hep::Detector* description;
    std::map<std::string, dd4hep::Rotation3D> rotations;
    std::map<std::string, dd4hep::Solid> shapes;
    std::map<std::string, dd4hep::Volume> volumes;
    std::map<std::string, VecDouble > numVectors;
    std::set<std::string> disabledAlgs;
    std::vector<std::string> namespaces;

    bool geo_inited = false;
    
    // Debug flags
    bool debug_includes     = false;
    bool debug_constants    = false;
    bool debug_materials    = false;
    bool debug_rotations    = false;
    bool debug_shapes       = false;
    bool debug_volumes      = false;
    bool debug_placements   = false;
    bool debug_namespaces   = false;
    bool debug_visattr      = false;
    bool debug_algorithms   = false;
  };
}

#endif
