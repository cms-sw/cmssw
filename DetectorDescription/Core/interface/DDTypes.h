#ifndef DD_DDTYPES_H
#define DD_DDTYPES_H

#include <string>
#include <vector>
#include <map>
#include "DetectorDescription/Core/interface/DDReadMapType.h"
#include <iosfwd>

//! corresponds to a collection of named doubles
/** in XML: a set of <Numeric name="n" value="1"/> */
typedef ReadMapType<double> DDNumericArguments;

//! corresponds to a collection of named strings
/** in XML: a set of <String name="n" value="val"/> */
typedef ReadMapType<std::string> DDStringArguments;

//! corresponds to a collection of named std::vectors of doubles */
/** in XML: a set of <Vector name="n"> 1,2,3,4 </Vector> */
typedef ReadMapType<std::vector<double> > DDVectorArguments;

//! corresponds to a collection of named std::maps of strings to double */
/** in XML: a set of <Map name="n"> val:1, val_2:2 </Map> */
typedef ReadMapType<std::map<std::string, double> > DDMapArguments;

typedef ReadMapType<std::vector<std::string> > DDStringVectorArguments;

std::ostream& operator<<(std::ostream& os, const DDNumericArguments& t);
std::ostream& operator<<(std::ostream& os, const DDStringArguments& t);
std::ostream& operator<<(std::ostream& os, const DDVectorArguments& t);
std::ostream& operator<<(std::ostream& os, const DDMapArguments& t);
std::ostream& operator<<(std::ostream& os, const DDStringVectorArguments& t);

// Formats an angle in radians as a 0-padded string in degrees; e.g. "0001.293900" for 1.2939 degrees.
std::string formatAsDegrees(double radianVal);

// Formats an angle in radians as a 0-padded string in degrees expressed as integer between 0 and 360; e.g. "090" for -270.001 degrees.
std::string formatAsDegreesInInteger(double radianVal);

#endif  // DDTYPES
