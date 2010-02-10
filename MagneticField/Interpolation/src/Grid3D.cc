#include "MagneticField/Interpolation/src/Grid3D.h"
#include <iostream>

void Grid3D::dump() const
{
  for (int j=0; j<gridb().nodes(); ++j) {
    for (int k=0; k<gridc().nodes(); ++k) {
      for (int i=0; i<grida().nodes(); ++i) {
        std::cout << grida().node(i) << " " << gridb().node(j) << " " << gridc().node(k) << " " 
		  << operator()(i,j,k) << std::endl;
      }
    }
  }
}


void  Grid3D::fillSub() {
#ifdef SUBGRID

  int sa =  grida().nodes()/subSize + (0==grida().nodes()%subSize) ? 0 : 1;
  int sb =  gridb().nodes()/subSize + (0==gridb().nodes()%subSize) ? 0 : 1;
  int sc =  gridc().nodes()/subSize + (0==gridc().nodes()%subSize) ? 0 : 1;
  m_subStride1 = sc;
  m_subStride2 = sc*sb;
  m_newdata.resize(sa*sb*sc);

  for (int i=0; i<grida().nodes(); ++i) 
    for (int j=0; j<gridb().nodes(); ++j) 
      for (int k=0; k<gridc().nodes(); ++k) {
	m_newdata[newIndex(i,j,k)] =  data_[index(i,j,k)];
      }

#endif
}
