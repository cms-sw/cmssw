#ifndef DD_DDSPLIT_H
#define DD_DDSPLIT_H

#include<string>
#include<utility>

//! split into (name,namespace), separator = ':'
std::pair<string,string> DDSplit(const std::string & n);

#endif
