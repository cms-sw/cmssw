#ifndef DETECTOR_DESCRIPTION_DD_DETECTOR_H
#define DETECTOR_DESCRIPTION_DD_DETECTOR_H

#include <string>

namespace dd4hep {
  class Detector;
}

namespace cms  {
  struct DDDetector {
    DDDetector();
    void process(const std::string& file);
    dd4hep::Detector& description() const { return *m_description; };
  private:
    dd4hep::Detector* m_description;
  };
}

#endif
