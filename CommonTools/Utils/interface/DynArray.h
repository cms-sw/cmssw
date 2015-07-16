#ifndef CommonTools_Utils_DynArray_H
#define CommonTools_Utils_DynArray_H

template<typename T>
class DynArray {
public:
   T * a=nullptr;
   unsigned int s=0;
public :
  DynArray(unsigned char * storage, unsigned int isize) : a((T*)(storage)), s(isize){
    for (auto i=0U; i<s; ++i) new((begin()+i)) T();
  }
  DynArray(unsigned char * storage, unsigned int isize, T const& it) : a((T*)(storage)), s(isize){
    for (auto i=0U; i<s; ++i) new((begin()+i)) T(it);
  }

  ~DynArray() { for (auto i=0U; i<s; ++i) a[i].~T(); }

   T & operator[](unsigned int i) { return a[i];}
   T * begin() { return a;}
   T * end() { return a+s;}
   T const & operator[](unsigned int i) const { return a[i];}
   T const * begin() const { return a;}
   T const * end() const { return a+s;}
   unsigned int size() const { return s;}
};

#define declareDynArray(T,n,x)  alignas(alignof(T)) unsigned char x ## _storage[sizeof(T)*n]; DynArray<T> x(x ## _storage,n)
#define initDynArray(T,n,x,i)  alignas(alignof(T)) unsigned char x ## _storage[sizeof(T)*n]; DynArray<T> x(x ## _storage,n,i)


#endif // CommonTools_Utils_DynArray_H
