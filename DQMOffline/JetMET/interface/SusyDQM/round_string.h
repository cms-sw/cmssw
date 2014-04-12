#ifndef ROUND_STRING_H
#define ROUND_STRING_H

#include <iomanip>
#include <sstream>
#include <string>
#include <sstream>
#include <cmath>

struct round_string {
  template<class T>
  std::string operator()(const std::pair<T,T> x) const {
    int val_digit(0), err_digit(0);
    
    if(x.first != 0) {
      while( fabs(x.first) / pow(10, val_digit) < 1 ) val_digit--;
      while( fabs(x.first) / pow(10, val_digit) > 10 )val_digit++;
    }
    if(x.second != 0 ) {
      while( x.second / pow(10,err_digit) < 0.95 ) err_digit--;
      while( x.second / pow(10,err_digit) > 9.50 ) err_digit++;
    }
    
    if(val_digit<err_digit) val_digit=err_digit;
    const bool scinot = (val_digit<-1 || err_digit>0);
    
    std::stringstream s;
    s << std::fixed << std::setprecision( scinot? val_digit-err_digit : -err_digit)
      << ( scinot? x.first/pow(10,val_digit) : x.first )
      << "("
      << unsigned(x.second / pow(10,err_digit) + 0.5) << ")";
    if(scinot) s<< "e" << (val_digit>0 ? "+" : "") << val_digit;
    
    return s.str();
  }
};
  
#endif
