#ifndef TrackFP420_h
#define TrackFP420_h

#include <vector>

class TrackFP420 {
public:

  TrackFP420() : ax_(0), bx_(0), chi2x_(0), nclusterx_(0), ay_(0), by_(0), chi2y_(0), nclustery_(0)  {}

    TrackFP420( double ax, double bx, double chi2x, int nclusterx, double ay, double by, double chi2y, int nclustery) : ax_(ax), bx_(bx), chi2x_(chi2x), nclusterx_(nclusterx), ay_(ay), by_(by), chi2y_(chi2y), nclustery_(nclustery) {}

  // Access to track information
  double ax() const   {return ax_;}
  double bx() const     {return bx_;}
  double chi2x() const {return chi2x_;}
  int nclusterx() const {return nclusterx_;}
  double ay() const   {return ay_;}
  double by() const     {return by_;}
  double chi2y() const {return chi2y_;}
  int nclustery() const {return nclustery_;}

private:
  double ax_;
  double bx_;
  double chi2x_;
  int nclusterx_;
  double ay_;
  double by_;
  double chi2y_;
  int nclustery_;
};

// Comparison operators
inline bool operator<( const TrackFP420& one, const TrackFP420& other) {
  return ( one.nclusterx() + one.nclustery() ) < ( other.nclusterx() + other.nclustery()  );
} 
#endif 
