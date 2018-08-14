#ifndef DETECTOR_DESCRIPTION_CORE_DDSVALUES_H
#define DETECTOR_DESCRIPTION_CORE_DDSVALUES_H

#include <algorithm>
#include <map>
#include <ostream>
#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/DDValue.h"

using DDsvalues_type = std::vector< std::pair<unsigned int, DDValue> >;   
using DDsvalues_Content_type = DDsvalues_type::value_type;

inline bool operator<( const DDsvalues_Content_type & lh, const DDsvalues_Content_type & rh ) {
   return lh.first < rh.first;
}

inline DDsvalues_type::const_iterator find( DDsvalues_type::const_iterator begin, DDsvalues_type::const_iterator end, unsigned int id ) {
   static const DDValue dummy;
   DDsvalues_Content_type v( id, dummy );
   DDsvalues_type::const_iterator it = std::lower_bound( begin, end, v );
   if( it != end && (*it).first == id ) return it;
   return end;
}

inline DDsvalues_type::const_iterator find( DDsvalues_type const & sv, unsigned int id ) {
  return find( sv.begin(), sv.end(), id );
}

void merge( DDsvalues_type & target, DDsvalues_type const & sv, bool sortit = true );

//! helper for retrieving DDValues from DDsvalues_type *.
bool DDfetch( const DDsvalues_type *,  DDValue & );

//! helper for retrieving DDValues from a std::vector of (DDsvalues_type *).
unsigned int DDfetch( const std::vector<const DDsvalues_type *> & sp, 
		      DDValue & toFetch, 
		      std::vector<DDValue> & result );

std::ostream & operator<<( std::ostream & , const DDsvalues_type & );
std::ostream & operator<<( std::ostream & , const std::vector<const DDsvalues_type*> & );

#endif
