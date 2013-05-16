#include "CondFormats/GeometryObjects/interface/PCaloGeometry.h"
#include <iostream>

PCaloGeometry::PCaloGeometry(){}

PCaloGeometry::PCaloGeometry( std::vector<float>    const & tra ,
			      std::vector<float>    const & dim , 
			      std::vector<uint32_t> const & ind , 
			      std::vector<uint32_t> const & din  ) :
  m_translation(tra),
  m_dimension(dim),
  m_indexes(ind),
  m_dins(din) {}

