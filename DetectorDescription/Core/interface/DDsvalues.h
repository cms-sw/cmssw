#ifndef DD_DDsvalues_type_h
#define DD_DDsvalues_type_h

//#include "DetectorDescription/Core/interface/DDAlgoPar.h"
#include <ostream>
#include <map>
#include <vector>
#include <algorithm>

#include "DetectorDescription/Core/interface/DDValue.h"

//typedef parS_type svalues_type;

//! std::maps an index to a DDValue. The index corresponds to the index assigned to the name of the std::mapped DDValue.

// typedef std::map< unsigned int, DDValue> DDsvalues_type;

// vin test
typedef std::vector< std::pair<unsigned int, DDValue> > DDsvalues_type;   
typedef DDsvalues_type::value_type  DDsvalues_Content_type;

//typedef std::map< unsigned int, DDValue*> DDsvalues_type;

typedef DDsvalues_type::value_type  DDsvalues_Content_type;

inline bool operator<(const DDsvalues_Content_type & lh, const DDsvalues_Content_type & rh){
   return lh.first < rh.first;
}

inline DDsvalues_type::const_iterator find( DDsvalues_type::const_iterator begin, DDsvalues_type::const_iterator end, unsigned int id) {
   static const DDValue dummy;
   DDsvalues_Content_type v(id,dummy);
   DDsvalues_type::const_iterator it = std::lower_bound(begin,end,v);
   if (it!=end && (*it).first==id) return it;
   return end; 
}

inline DDsvalues_type::const_iterator find(DDsvalues_type const & sv, unsigned int id) {
  return find(sv.begin(),sv.end(),id);
}



void merge(DDsvalues_type & target, DDsvalues_type const & sv, bool sortit=true );


//! helper for retrieving DDValues from DDsvalues_type *.
bool DDfetch(const DDsvalues_type *,  DDValue &);

//! helper for retrieving DDValues from a std::vector of (DDsvalues_type *).
unsigned int DDfetch(const std::vector<const DDsvalues_type *> & sp, 
                                  DDValue & toFetch, 
				      std::vector<DDValue> & result);
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
