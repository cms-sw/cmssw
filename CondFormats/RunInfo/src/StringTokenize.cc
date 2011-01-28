/*
 * StringTokenize.cc
 *
 *  Created on: Mar 25, 2010
 *      Author: diguida
 */

#include "CondFormats/RunInfo/interface/StringTokenize.h"

void stringTokenize(const std::string& str
                   ,std::vector<std::string>& tokens
                   ,char delimiter) {
   // Skip delimiters at beginning.
   std::string::size_type lastPos = str.find_first_not_of(delimiter, 0);
   // Find first "non-delimiter".
   std::string::size_type pos = str.find_first_of(delimiter, lastPos);
   while (std::string::npos != pos || std::string::npos != lastPos) {
     // Found a token, add it to the vector.
     tokens.push_back(str.substr(lastPos, pos - lastPos));
     // Skip delimiters.  Note the "not_of"
     lastPos = str.find_first_not_of(delimiter, pos);
     // Find next "non-delimiter"
     pos = str.find_first_of(delimiter, lastPos);
   }
}
