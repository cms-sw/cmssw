#ifndef DETECTOR_DESCRIPTION_DD_DETECTOR_H
#define DETECTOR_DESCRIPTION_DD_DETECTOR_H

#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include <string>

namespace dd4hep {
  class Detector;
}

namespace cms  {
  class DDDetector {
  public:
    using Detector = dd4hep::Detector;

    explicit DDDetector(const std::string&, const std::string&);
    DDDetector() = delete;
    
    ~DDDetector();

    // FIXME: remove the need for it
    Detector const* description() const {
      return m_description;
    }
    
    DDVectorsMap const& vectors() const {
      return m_vectors;
    }

    DDPartSelectionMap const& partsels() const {
      return m_partsels;
    }
    
    DDSpecParRegistry const& specpars() const {
      return m_specpars;
    }

  private:

    void process(const std::string&);
    
    Detector* m_description = nullptr;
    DDVectorsMap m_vectors;
    DDPartSelectionMap m_partsels;
    DDSpecParRegistry m_specpars;
    const std::string m_tag;
  };
}

#endif
