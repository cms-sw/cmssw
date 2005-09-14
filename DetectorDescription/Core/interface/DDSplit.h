#ifndef DD_DDSPLIT_H
#define DD_DDSPLIT_H

#include <string>
#include<utility>

//! split into (name,namespace), separator = ':'
std::pair<std::string,std::string> DDSplit(const std::string & n);

#endif
