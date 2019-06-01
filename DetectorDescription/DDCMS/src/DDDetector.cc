#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include <DD4hep/Detector.h>
#include <DD4hep/Volumes.h>

#include <iostream>

using namespace cms;
using namespace std;

DDDetector::DDDetector(const string& tag, const string& fileName) : m_tag(tag) {
  m_description = &Detector::getInstance(tag);
  m_description->addExtension<DDVectorsMap>(&m_vectors);
  m_description->addExtension<DDPartSelectionMap>(&m_partsels);
  m_description->addExtension<DDSpecParRegistry>(&m_specpars);
  process(fileName);
}

DDDetector::~DDDetector() { Detector::destroyInstance(m_tag); }

void DDDetector::process(const string& fileName) {
  std::string name("DD4hep_CompactLoader");
  const char* files[] = {fileName.c_str(), nullptr};
  m_description->apply(name.c_str(), 2, (char**)files);
}

dd4hep::Volume DDDetector::worldVolume() const {
  assert(m_description);
  return m_description->worldVolume();
}
