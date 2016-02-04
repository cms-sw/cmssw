#ifndef RecoFP420_h
#define RecoFP420_h

#include <vector>

class RecoFP420 {
public:

  RecoFP420() : e0_(0), x0_(0), y0_(0), tx0_(0), ty0_(0), q20_(0), direction_(0)  {}

    RecoFP420( double e0, double x0, double y0, double tx0, double ty0, double q20, int direction) : e0_(e0), x0_(x0), y0_(y0), tx0_(tx0), ty0_(ty0), q20_(q20), direction_(direction) {}

  // Access to track information
  double e0() const   {return e0_;}
  double x0() const   {return x0_;}
  double y0() const   {return y0_;}
  double tx0() const {return tx0_;}
  double ty0() const {return ty0_;}
  double q20() const {return q20_;}
  int direction() const {return direction_;}

private:
  double e0_;
  double x0_;
  double y0_;
  double tx0_;
  double ty0_;
  double q20_;
  int direction_;
};

// Comparison operators
inline bool operator<( const RecoFP420& one, const RecoFP420& other) {
  return ( one.e0() ) < ( other.e0() );
//   return ( one.x0() + one.y0() ) < ( other.x0() + other.y0()  );
} 
#endif 
