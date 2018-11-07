#include "DetectorDescription/DDCMS/interface/DDSingleton.h"

#include <map>
#include <string>
#include <vector>
#include <unordered_map>

struct DDVectorRegistry : public cms::DDSingleton< std::unordered_map< std::string, std::vector<double> >, DDVectorRegistry>{};
