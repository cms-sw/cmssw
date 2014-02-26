///
/// \class l1t::CaloParams
///
/// Description: Placeholder for calorimeter trigger parameters
///
/// Implementation:
///    
///
/// \author: Jim Brooke
///

#ifndef CaloParams_h
#define CaloParams_h

#include <iostream>

namespace l1t {
  
  class CaloParams {
    
  public:

    CaloParams() {}

    // getters (placeholder)
    double param() const {return p_;}
    
    // setters (placeholder)
    void setParam(double p) {p_ = p;}

    // print parameters to stream:
    void print(std::ostream&) const;
    friend std::ostream& operator<<(std::ostream& o, const CaloParams & p)  { p.print(o); return o; } 
    
  private:

    double p_;

  };

}// namespace
#endif
