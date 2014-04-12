#ifndef DataFormats_Common_EDCollection_h
#define DataFormats_Common_EDCollection_h

/*----------------------------------------------------------------------
  
EDCollection: A collection of homogeneous objects that can be used for an EDProduct,
or as a base class for an EDProduct.


----------------------------------------------------------------------*/

#include <vector>

namespace edm {
  template <class T>
  class EDCollection {
  public:
    typedef T value_type;
    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef typename std::vector<T>::size_type size_type;
    EDCollection();
    explicit EDCollection(size_type n);
    explicit EDCollection(std::vector<T> const& vec);
    EDCollection(EDCollection<T> const& h);
    virtual ~EDCollection();
    void push_back(T const& t);
    void swap(EDCollection<T>& other);
    EDCollection<T>& operator=(EDCollection<T> const& rhs);
    bool empty() const;
    size_type size() const;
    size_type capacity() const;
    void reserve(size_type n);
    T& operator[](size_type i);
    T const& operator[](size_type i) const;
    T& at(size_type i);
    T const& at(size_type i) const;
    const_iterator begin() const;
    const_iterator end() const;
    

  private:
    std::vector<T> obj;    
  };

  template <class T>
  inline
  EDCollection<T>::EDCollection() : obj() {}

  template <class T>
  inline
  EDCollection<T>::EDCollection(size_type n) : obj(n) {}

  template <class T>
  inline
  EDCollection<T>::EDCollection(std::vector<T> const& vec) : obj(vec) {}

  template <class T>
  inline
  EDCollection<T>::EDCollection(EDCollection<T> const& h) : obj(h.obj) {}

  template <class T>
  EDCollection<T>::~EDCollection() {}

  template <class T>
  inline
  void
  EDCollection<T>::push_back(T const& t) {
    obj.push_back(t);
  }

  template <class T>
  inline
  void
  EDCollection<T>::swap(EDCollection<T>& other) {
    obj.swap(other.obj);
  }

  template <class T>
  inline
  EDCollection<T>&
  EDCollection<T>::operator=(EDCollection<T> const& rhs) {
    EDCollection<T> temp(rhs);
    this->swap(temp);
    return *this;
  }

  template <class T>
  inline
  bool
  EDCollection<T>::empty() const {
    return obj.empty();
  }

  template <class T>
  inline
  typename std::vector<T>::size_type
  EDCollection<T>::size() const {
    return obj.size();
  }

  template <class T>
  inline
  typename std::vector<T>::size_type
  EDCollection<T>::capacity() const {
    return obj.capacity();
  }

  template <class T>
  inline
  void
  EDCollection<T>::reserve(typename std::vector<T>::size_type n) {
    obj.reserve(n);
  }

  template <class T>
  inline
  T& 
  EDCollection<T>::operator[](size_type i) {
    return obj[i];
  }

  template <class T>
  inline
  T const& 
  EDCollection<T>::operator[](size_type i) const {
    return obj[i];
  }

  template <class T>
  inline
  T& 
  EDCollection<T>::at(size_type i) {
    return obj.at(i);
  }

  template <class T>
  inline
  T const& 
  EDCollection<T>::at(size_type i) const {
    return obj.at(i);
  }

  template <class T>
  inline
  typename std::vector<T>::const_iterator
  EDCollection<T>::begin() const {
    return obj.begin();
  }

  template <class T>
  inline
  typename std::vector<T>::const_iterator
  EDCollection<T>::end() const {
    return obj.end();
  }

  // Free swap function
  template <class T>
  inline
  void
  swap(EDCollection<T>& a, EDCollection<T>& b) 
  {
    a.swap(b);
  }

}

#endif
