#ifndef FWCore_Utilities_ConstRespectingPtr_h
#define FWCore_Utilities_ConstRespectingPtr_h

// Package:     FWCore/Utilities
// Class  :     ConstRespectingPtr
//
/**\class edm::ConstRespectingPtr

 Description: When this is a member of a class, const functions of the class
can only call const functions of the object it points at. This aids in using
the compiler to help maintain thread safety of const functions.

Usage: WARNING: member data which uses this class must be made transient in
the classes_def.xml file if it is a member of a persistent class!

*/
// Original Author:  W. David Dagenhart
//         Created:  20 March 2014

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <memory>
#endif

namespace edm {

  template <typename T>
  class ConstRespectingPtr {

  public:

    ConstRespectingPtr();
    explicit ConstRespectingPtr(T*);
    ~ConstRespectingPtr();

    T const* operator->() const { return m_data; }
    T const& operator*() const { return *m_data; }
    T const* get() const { return m_data; }

    T* operator->() { return m_data; }
    T& operator*() { return *m_data; }
    T* get() { return m_data; }

    bool isSet() const;

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    void set(std::unique_ptr<T> iNewValue);
#endif

    T* release();
    void reset();

  private:

    ConstRespectingPtr(ConstRespectingPtr<T> const&);
    ConstRespectingPtr& operator=(ConstRespectingPtr<T> const&);

    T* m_data;
  };

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)

  template<typename T>
  ConstRespectingPtr<T>::ConstRespectingPtr() : m_data(nullptr) {}

  template<typename T>
  ConstRespectingPtr<T>::ConstRespectingPtr(T* v) : m_data(v) {}

  template<typename T>
  ConstRespectingPtr<T>::~ConstRespectingPtr() {
    delete m_data;
  }

  template<typename T>
  bool ConstRespectingPtr<T>::isSet() const { return nullptr != m_data; }

  template<typename T>
  void ConstRespectingPtr<T>::set(std::unique_ptr<T> iNewValue) {
    delete m_data;
    m_data = iNewValue.release();
  }

  template<typename T>
  T* ConstRespectingPtr<T>::release() {
    T* tmp = m_data;
    m_data = nullptr;
    return tmp;
  }

  template<typename T>
  void ConstRespectingPtr<T>::reset() {
    delete m_data;
    m_data = nullptr;
  }
#endif
}
#endif
