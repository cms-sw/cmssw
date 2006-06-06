#ifndef ParameterSet_ProductTag_h
#define ParameterSet_ProductTag_h

#include <string>

namespace edm {

  class ProductTag
  {
  public:
    ProductTag();
    /// the input string is of the form:
    /// label
    /// label:instance
    /// label;alias    <-- note semicolon
    /// label:instance;alias
    void decode(const std::string & s);
    std::string encode() const;

    std::string label()    const {return label_;} 
    std::string instance() const {return instance_;}
    std::string alias()    const {return alias_;}

  private:
    std::string label_;
    std::string instance_;
    std::string alias_;
  };

}

#endif

