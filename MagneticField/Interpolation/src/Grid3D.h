#ifndef Grid3D_h
#define Grid3D_h

/** \class Grid3D
 *
 *  Implementation of a 3D regular grid.
 *
 *  $Date: $
 *  $Revision: $
 *  \author T. Todorov
 */


#include <vector>

template <class Value, class T>
class Grid3D {
public:

  typedef Value    ValueType;
  typedef T        Scalar;

  Grid3D() {}

  Grid3D( const Grid1D<T>& ga, const Grid1D<T>& gb, const Grid1D<T>& gc,
	  const std::vector<Value>& data) : 
    grida_(ga), gridb_(gb), gridc_(gc), data_(data) {
     stride1_ = gridb_.nodes() * gridc_.nodes();
     stride2_ = gridc_.nodes();
  }

  const Value& operator()( int i, int j, int k) const {
    return data_[index(i,j,k)];
  }

  const Grid1D<T>& grida() const {return grida_;}
  const Grid1D<T>& gridb() const {return gridb_;}
  const Grid1D<T>& gridc() const {return gridc_;}

  const std::vector<Value>& data() const {return data_;}

  void dump() const;

private:

  Grid1D<T> grida_;
  Grid1D<T> gridb_;
  Grid1D<T> gridc_;

  std::vector<Value> data_;

  int stride1_;
  int stride2_;

  int index(int i, int j, int k) const {return i*stride1_ + j*stride2_ + k;}

};

#include <iostream>
template <class Value, class T>
void Grid3D<Value,T>::dump() const 
{
  for (int j=0; j<gridb().nodes(); ++j) {
    for (int k=0; k<gridc().nodes(); ++k) {
      for (int i=0; i<grida().nodes(); ++i) {
	std::cout << grida().node(i) << " " << gridb().node(j) << " " << gridc().node(k) << " " << operator()(i,j,k) << std::endl;
      }
    }
  }
}

#endif
