#ifndef DETECTOR_DESCRIPTION_DD_DETECTOR_H
#define DETECTOR_DESCRIPTION_DD_DETECTOR_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace dd4hep {
  class Detector;
}

namespace cms  {
  struct DDDetector {
    
    using DDVectorsMap = std::unordered_map< std::string, std::vector<double>>;

    DDDetector();

    void process(const std::string& file);
    dd4hep::Detector& description() const { return *m_description; };

  private:
    dd4hep::Detector* m_description;
    DDVectorsMap m_vectors;
  };
}

#endif
