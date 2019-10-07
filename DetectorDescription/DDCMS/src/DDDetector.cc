#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include <DD4hep/Detector.h>
#include <DD4hep/DetectorTools.h>
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

void DDDetector::process(const string& fileName) {
  std::string name("DD4hep_CompactLoader");
  const char* files[] = {fileName.c_str(), nullptr};
  m_description->apply(name.c_str(), 2, (char**)files);
}

dd4hep::Volume DDDetector::worldVolume() const {
  assert(m_description);
  return m_description->worldVolume();
}

dd4hep::PlacedVolume DDDetector::worldPlacement() const { return world().placement(); }

dd4hep::DetElement DDDetector::world() const {
  assert(m_description);
  return m_description->world();
}

const dd4hep::Detector::HandleMap& DDDetector::detectors() const {
  assert(m_description);
  return m_description->detectors();
}

TGeoManager& DDDetector::manager() const {
  assert(m_description);
  return m_description->manager();
}

dd4hep::DetElement DDDetector::findElement(const std::string& path) const {
  assert(m_description);
  return dd4hep::detail::tools::findElement(*m_description, path);
}
