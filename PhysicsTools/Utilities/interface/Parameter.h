#ifndef PhysicsTools_Parameter_h
#define PhysicsTools_Parameter_h
#include <string>
#include <boost/shared_ptr.hpp>

namespace function {
  class Parameter {
  public:
    explicit Parameter(const std::string & name ="undefined" , double value = 0) :
      name_(name), value_(new double(value)) {
    }
    const std::string & name() const { return name_; }
    double value() const { return *value_; }
    boost::shared_ptr<double> ptr() const { return value_; }
    operator double() const { return value(); }
    operator boost::shared_ptr<double>() const { return value_; }
    Parameter & operator=(double value) { *value_ = value; return * this; }
  private:
    std::string name_;
    boost::shared_ptr<double> value_;
  };
}

#include <iostream>

inline std::ostream & operator<<(std::ostream&cout, const function::Parameter & p) {
  return cout << p.name() <<" = " << p.value();
}

#endif
