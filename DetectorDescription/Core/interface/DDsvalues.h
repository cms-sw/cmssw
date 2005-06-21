#ifndef DD_DDsvalues_type_h
#define DD_DDsvalues_type_h

//#include "DetectorDescription/DDCore/interface/DDAlgoPar.h"
#include <iostream>
#include <map>
//#include <vector>
#include "DetectorDescription/DDCore/interface/DDValue.h"

//typedef parS_type svalues_type;

//! maps an index to a DDValue. The index corresponds to the index assigned to the name of the mapped DDValue.
typedef std::map< unsigned int, DDValue> DDsvalues_type;
//typedef std::map< unsigned int, DDValue*> DDsvalues_type;

//! helper for retrieving DDValues from DDsvalues_type *.
bool DDfetch(const DDsvalues_type *,  DDValue &);

//! helper for retrieving DDValues from a vector of (DDsvalues_type *).
unsigned int DDfetch(const vector<const DDsvalues_type *> & sp, 
                                  DDValue & toFetch, 
				      vector<DDValue> & result);
/*
class DDsvalues_type : public std::map< unsigned int, DDValue>
{
public:
  DDValue operator[](const unsigned int& i) const;
};
*/

std::ostream & operator<<(std::ostream & , const DDsvalues_type &);
std::ostream & operator<<(std::ostream & , const std::vector<const DDsvalues_type*> &);

#endif
