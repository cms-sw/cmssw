#ifndef DataFormatsCommonSmartVector_H
#define DataFormatsCommonSmartVector_H

#include<vector>
#include<variant>
#include<array>
#include<cstdint>


// a mimimal smart vector that can be either an array or a vector
// T must be an integer type (even if it may work with float as well)
template<typename T>
class SmartVector {
public :
  using Vector = std::vector<T>;
  static constexpr uint32_t maxSize = sizeof(Vector)/sizeof(T)-1;
  using Array = std::array<T,sizeof(Vector)/sizeof(T)>;
  using Variant = std::variant<Vector,Array>;

  SmartVector(){}

  template<typename Iter>
  SmartVector(Iter b, Iter e) {
     initialize(b,e);
  }

  template<typename Iter>
  void initialize(Iter b, Iter e) {
     if (e-b<=maxSize) {
       m_container = Array();
       auto & a = std::get<Array>(m_container);
       std::copy(b,e,a.begin());
       a.back()=e-b;
     } else
       m_container. template emplace<Vector>(b,e);
  }

  template<typename Iter>
  void extend(Iter b, Iter e) {
    if(auto pval = std::get_if<Array>(&m_container)) {
      auto cs = pval->back();         
      uint32_t ns = (e-b)+cs;
      if (ns<=maxSize) {
        std::copy(b,e,&(*pval)[cs]);
        pval->back()=ns;
      } else {
        Vector v; v.reserve(ns);
        v.insert(v.end(),pval->begin(),pval->begin()+cs);
        v.insert(v.end(),b,e);
        m_container = std::move(v);
      }
    }else if(auto pval = std::get_if<Vector>(&m_container)) {
      pval->insert(pval->end(),b,e);
    }
    else {
     initialize(b,e);
    }
  }


  T const * begin() const { 
    if(auto pval = std::get_if<Array>(&m_container))
       return pval->data();
    else
       return std::get<Vector>(m_container).data();
  }

  T const * end() const {
    if(auto pval = std::get_if<Array>(&m_container))
       return pval->data()+pval->back();
    else
       return std::get<Vector>(m_container).data()+std::get<Vector>(m_container).size();
  }

  T const & operator[](uint32_t i) const {
    return *(begin()+i);
  }

  uint32_t size() const {
    if(auto pval = std::get_if<Array>(&m_container))
       return pval->back();
    else
       return std::get<Vector>(m_container).size();
  }


  Variant const & container() const { return m_container;}
private:
  Variant m_container;
};

#endif
