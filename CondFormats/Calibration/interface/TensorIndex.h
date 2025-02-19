#ifndef TensorIndex_h
#define TensorIndex_h
///
///Credit: 
///Utility class from 
///
///http://www.sitmo.com/doc/A_Simple_and_Extremely_Fast_CPP_Template_for_Matrices_and_Tensors
///
///Usage:
///
///The template below offers a simple and efficient solution for handling matrices and tensors in C++. The idea is to store the matrix (or tensor) in a standard vector by translating the multidimensional index to a one dimensional index.
///
///The only thing we need to do is to convert two dimensional indices (r,c) into a one dimensional index. Using template we can do this very efficiently compile time, minimizing the runtime overhead.
///
template <int d1,int d2=1,int d3=1,int d4=1>
class TensorIndex {
  public:
	enum {SIZE = d1*d2*d3*d4 };
	enum {LEN1 = d1 };
	enum {LEN2 = d2 };
	enum {LEN3 = d3 };
	enum {LEN4 = d4 };

    static int indexOf(const int i) {
      return i;
    }
    static int indexOf(const int i,const int j) {
      return j*d1 + i;
    }
    static int indexOf(const int i,const int j, const int k) {
      return (k*d2 + j)*d1 + i;
    }
    static int indexOf(const int i,const int j, const int k,const int l) {
      return ((l*d3 + k)*d2 + j)*d1 + i;
    }
};

template <int d1,int d2=1,int d3=1,int d4=1>
class TensorIndex_base1 {
  public:
	enum {SIZE = d1*d2*d3*d4 };
	enum {LEN1 = d1 };
	enum {LEN2 = d2 };
	enum {LEN3 = d3 };
	enum {LEN4 = d4 };

    static int indexOf(const int i) {
      return i -1;
    }
    static int indexOf(const int i,const int j) {
      return j*d1 + i -1 -d1;
    }
    static int indexOf(const int i,const int j, const int k) {
      return (k*d2 + j)*d1 + i -1 -d1 -d1*d2;
    }
    static int indexOf(const int i,const int j, const int k,const int l) {
      return ((l*d3 + k)*d2 + j)*d1 + i -1 -d1 -d1*d2 - d1*d2*d3;
    }
};
#endif
