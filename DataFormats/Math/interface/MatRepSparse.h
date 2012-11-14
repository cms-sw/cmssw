#ifndef MatRepSparse_H
#define MatRepSparse_H
#include<type_traits>
#include<algorithm>

/** a sparse matrix
 *  just for storage purposes
 *  can be assigned to other matrices
 */
template <typename T, unsigned int D1, unsigned int D2, unsigned int S, typename F=int(*)(int)>
class MatRepSparse {
  
public: 
  MatRepSparse(){}
  template<typename FI> 
  explicit MatRepSparse(FI fi) : f(fi) { } 
  
  typedef T  value_type;
  
  static T & sink() {
    static T t=0;  // this should throw...
    return t;
  }

  static T const & csink() {
    static const T t=0;
    return t;
  }


  inline T const & operator()(unsigned int i, unsigned int j) const {
    int k =  f(i*D2+j);
    return k<0 ? csink() : fArray[k];
  }

  inline T& operator()(unsigned int i, unsigned int j) {
    int k =  f(i*D2+j);
    return k<0 ? sink() : fArray[k];
  }
  
  inline T& operator[](unsigned int i) { 
    int k =  f(i);
    return k<0 ? sink() : fArray[k];
  }
  
  inline T const & operator[](unsigned int i) const {
    int k =  f(i);
    return k<0 ? csink() : fArray[k];
  }

  inline T apply(unsigned int i) const {
    int k =  f(i);
    return k<0 ? 0 : fArray[k];
  }

  inline T* Array() { return fArray; }  
  
  inline const T* Array() const { return fArray; }  
  
  /**
     assignment : only sparse to sparse allowed
  */
  template <class R>
  inline MatRepSparse & operator=(const R&) {
    static_assert(std::is_same<R,MatRepSparse<T,D1,D2,S,F>>::value,
		  "Cannot_assign_general_to_sparse_matrix_representation");
    return *this;
  }

  inline MatRepSparse & operator=(const MatRepSparse& rhs) {
    for(unsigned int i=0; i<kSize; ++i) fArray[i] = rhs.Array()[i];
    return *this;
  }
  
  /**
     self addition : only sparse to sparse allowed
  */
  template <class R>
  inline MatRepSparse & operator+=(const R&) {
    static_assert(std::is_same<R,MatRepSparse<T,D1,D2,S,F>>::value,
		  "Cannot_add_general_to_sparse_matrix_representation");
         return *this;
  }
  inline MatRepSparse & operator+=(const MatRepSparse& rhs) {
    for(unsigned int i=0; i<kSize; ++i) fArray[i] += rhs.Array()[i];
    return *this;
  }

  /**
     self subtraction : only sparse to sparse allowed
  */
  template <class R>
  inline MatRepSparse & operator-=(const R&) {
    static_assert(std::is_same<R,MatRepSparse<T,D1,D2,S,F>>::value,
		  "Cannot_substract_general_to_sparse_matrix_representation");
    return *this;
  }

  inline MatRepSparse & operator-=(const MatRepSparse& rhs) {
    for(unsigned int i=0; i<kSize; ++i) fArray[i] -= rhs.Array()[i];
    return *this;
  }
  template <class R> 
  inline bool operator==(const R& rhs) const {
    bool rc = true;
    for(unsigned int i=0; i<D1*D2; ++i) {
      rc = rc && (operator[](i) == rhs[i]);
    }
    return rc;
  }
  
  enum {
    /// return no. of matrix rows
    kRows = D1,
    /// return no. of matrix columns
    kCols = D1,
    /// return no of elements: rows*columns
    kSize = S
  };
  
  
public:
  T fArray[kSize]={0};
  F f;
};

#endif //
