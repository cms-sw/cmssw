#include "Grid3D.h"
#include <iostream>


/*
Grid3D::Grid3D( const Grid1D& ga, const Grid1D& gb, const Grid1D& gc,
		std::vector<ValueType> const & data) : 
  grida_(ga), gridb_(gb), gridc_(gc) {
  data_.reserve(data.size());
  //FIXME use a std algo
  for (size_t i=0; i<=data.size(); ++i)
    data_.push_back(ValueType(data[i].x(),data[i].y(),data[i].z()));
  stride1_ = gridb_.nodes() * gridc_.nodes();
  stride2_ = gridc_.nodes();
}
*/

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


