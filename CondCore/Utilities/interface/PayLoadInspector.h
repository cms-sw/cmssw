#ifndef  PayLoadInspector_H
#define  PayLoadInspector_H
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/IOVService/interface/IOVProxy.h"
#include <string>
#include <vector>

namespace cond {

  template<typename T>
  class BaseValueExtractor {
  public:
    typedef T Class;

    BaseValueExtractor(){}
    
    virtual ~BaseValueExtractor(){}
    void computeW(Class const &o){
      reset();
      compute(o);
    }
    std::vector<float> const & values() const { return m_values;}
  protected:
    void add(float v) { m_values.push_back(v); }
  private:
    void reset() { m_values.clear();}
    virtual void compute(Class const &){}
    
    
  private:
    std::vector<float> m_values;
  };


  template<typename T>
  class ValueExtractor : public  BaseValueExtractor<T> {
  public:
    typedef T Class;
    ValueExtractor(){}
    ValueExtractor(std::string const &, std::vector<int> const&){}
  private:
    void compute(Class const &){}
  private:
  };

  template<typename T>
  class PayLoadInspector {
  public:
    typedef T Class;
    typedef ValueExtractor<T> Extractor;
        
    PayLoadInspector() {}
    PayLoadInspector(const cond::IOVElement & elem) : 
      object(*elem.db(),elem.payloadToken()){}

    std::string print() const { return ""; }

    std::string summary() const {return ""; }

    void extract(Extractor & extractor) const {extractor.computeW(*object); }

  private:
    cond::TypedRef<Class> object;    

  };

}

#endif //   PayLoadInspector_H
