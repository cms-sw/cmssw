#ifndef DETECTOR_DESCRIPTION_DD_CURRENT_NAMESPACE_H
#define DETECTOR_DESCRIPTION_DD_CURRENT_NAMESPACE_H

#include "DetectorDescription/Core/interface/DDSingleton.h"
#include <iostream>
#include <string>

struct DDCurrentNamespace;

std::ostream & operator<<( std::ostream &, const DDCurrentNamespace & );

struct DDCurrentNamespace : public dd::DDSingleton<std::string, DDCurrentNamespace>
{
  DDCurrentNamespace() : DDSingleton(1) {}
  static std::unique_ptr<std::string> init() {
    return std::make_unique<std::string>( "GLOBAL" ); }
};

#endif
