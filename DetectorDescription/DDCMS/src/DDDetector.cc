#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DD4hep/Detector.h"

#include <iostream>

using namespace cms;

DDDetector::DDDetector()
  : m_description(nullptr)
{
}

void
DDDetector::process(const std::string& fileName)
{
  m_description = &dd4hep::Detector::getInstance();

  std::string name("DD4hep_CompactLoader");
  const char* files[] = { fileName.c_str(), nullptr };
  m_description->apply( name.c_str(), 2, (char**)files );
}

