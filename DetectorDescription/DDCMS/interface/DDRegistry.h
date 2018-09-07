#include "DetectorDescription/DDCMS/interface/DDSingleton.h"

#include <map>
#include <string>
#include <vector>
#include <unordered_map>

using namespace std;
using namespace cms;

struct DDVectorRegistry : public DDSingleton< unordered_map< string, vector<double> >, DDVectorRegistry>{};
