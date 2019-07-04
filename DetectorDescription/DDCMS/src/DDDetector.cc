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
//   ClhepEvaluator::ClhepEvaluator() {
// 0012   // enable standard mathematical funtions
// 0013   evaluator_.setStdMath();
// 0014 
// 0015   // set Geant4 compatible units
// 0016   evaluator_.setSystemOfUnits(1.e+3, 1. / 1.60217733e-25, 1.e+9, 1. / 1.60217733e-10, 1.0, 1.0, 1.0);
// 0017 
// 0018   // set some global vars, which are in fact known by Clhep::SystemOfUnits
// 0019   // but are NOT set in CLHEP::Evaluator ...
// 0020   evaluator_.setVariable("mum", "1.e-3*mm");
// 0021   evaluator_.setVariable("fm", "1.e-15*meter");
// 0022 }
  const char* files[] = {fileName.c_str(), nullptr};
  m_description->apply(name.c_str(), 2, (char**)files);
}

dd4hep::Volume DDDetector::worldVolume() const {
  assert(m_description);
  return m_description->worldVolume();
}
