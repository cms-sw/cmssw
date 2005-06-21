#ifndef DD_DDTYPES_H
#define DD_DDTYPES_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "DetectorDescription/DDBase/interface/DDReadMapType.h"

//! corresponds to a collection of named doubles 
/** in XML: a set of <Numeric name="n" value="1"/> */
typedef ReadMapType<double> DDNumericArguments;

//! corresponds to a collection of named strings
/** in XML: a set of <String name="n" value="val"/> */
typedef ReadMapType<std::string> DDStringArguments;

//! corresponds to a collection of named vectors of doubles */
/** in XML: a set of <Vector name="n"> 1,2,3,4 </Vector> */
typedef ReadMapType<std::vector<double> > DDVectorArguments;

//! corresponds to a collection of named maps of strings to double */
/** in XML: a set of <Map name="n"> val:1, val_2:2 </Map> */
typedef ReadMapType<std::map<std::string,double> > DDMapArguments;

typedef ReadMapType<std::vector<std::string> > DDStringVectorArguments;


std::ostream & operator<<(std::ostream & os, const DDNumericArguments & t);
std::ostream & operator<<(std::ostream & os, const DDStringArguments & t);
std::ostream & operator<<(std::ostream & os, const DDVectorArguments & t);
std::ostream & operator<<(std::ostream & os, const DDMapArguments & t);
std::ostream & operator<<(std::ostream & os, const DDStringVectorArguments & t);
#endif // DDTYPES
