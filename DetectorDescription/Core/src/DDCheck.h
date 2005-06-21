#ifndef DDD_DDCheck_h
#define DDD_DDCheck_h

#include <iostream>
#include <vector>
#include "DetectorDescription/DDCore/interface/DDName.h"
#include <map>
// some self-consistency checks

bool DDCheck(std::ostream&);
bool DDCheckMaterials(std::ostream&, std::vector<std::pair<std::string,DDName> > * = 0);

#endif
