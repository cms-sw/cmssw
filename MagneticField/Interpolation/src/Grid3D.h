#ifndef Grid3D_h
#define Grid3D_h

/** \class Grid3D
 *
 *  Implementation of a 3D regular grid.
 *
*  \author T. Todorov
 */

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "MagneticField/Interpolation/src/Grid1D.h"
#include <vector>

class Grid3D {
public:

  typedef Basic3DVector<float>   ValueType;
  typedef float   Scalar;
  // typedef double   Scalar;

  Grid3D() {}

  Grid3D( const Grid1D& ga, const Grid1D& gb, const Grid1D& gc,
	  std::vector<ValueType>& data) : 
    grida_(ga), gridb_(gb), gridc_(gc) {
     data_.swap(data);
     stride1_ = gridb_.nodes() * gridc_.nodes();
     stride2_ = gridc_.nodes();

     fillSub();
  }

#ifdef SUBGRID
  const ValueType& operator()( int i, int j, int k) const {
    return m_newdata[newIndex(i,j,k)];
  }
#else
  const ValueType& operator()( int i, int j, int k) const {
    return data_[index(i,j,k)];
  }
#endif

  int index(int i, int j, int k) const {return i*stride1_ + j*stride2_ + k;}
  int stride1() const { return stride1_;}
  int stride2() const { return stride2_;}
  int stride3() const { return 1;}
  const ValueType& operator()(int i) {
    return data_[i];
  }

  const Grid1D& grida() const {return grida_;}
  const Grid1D& gridb() const {return gridb_;}
  const Grid1D& gridc() const {return gridc_;}

  const std::vector<ValueType>& data() const {return data_;}

  void dump() const;

private:

  Grid1D grida_;
  Grid1D gridb_;
  Grid1D gridc_;

  std::vector<ValueType> data_;

  int stride1_;
  int stride2_;


  void fillSub();

#ifdef SUBGRID
  const int subSize = 4;
  std::vector<ValueType> m_newdata;
  int m_subStride1;
  int m_subStride2;
  void fillSub();

  int newIndex(int i, int j, int k) const {
    // find submatrix
    int si = i/subSize;
    int sj = j/subSize;
    int sk = k/subSize;
    // location in submatrix
    int l =  (k -sk*subSize)  + subSize*( (j -sj*subSize) + subSize*(i -si*subSize) );
    return l + si*m_subStride1+sj*sunStride2+sk;
  }

#endif

};

#endif
