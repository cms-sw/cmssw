#ifndef DDD_DDCheck_h
#define DDD_DDCheck_h

#include <iostream>
#include <vector>
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

// some self-consistency checks
bool DDCheck(std::ostream&);
bool DDCheck(const DDCompactView& cpv, std::ostream&);
bool DDCheckMaterials(std::ostream&, std::vector<std::pair<std::string,DDName> > * = 0);

#endif
