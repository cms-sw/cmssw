#include "CondFormats/GeometryObjects/interface/PCaloGeometry.h"
#include <iostream>

PCaloGeometry::PCaloGeometry(){}

PCaloGeometry::PCaloGeometry( std::vector<double>   const & tra ,
			      std::vector<double>   const & dim , 
			      std::vector<uint32_t> const & ind   ) :
  m_translation(tra),
  m_dimension(dim),
  m_indexes(ind){}

