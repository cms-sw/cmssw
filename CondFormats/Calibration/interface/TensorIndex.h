#ifndef TensorIndex_h
#define TensorIndex_h
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
