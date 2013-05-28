#ifndef CSCRecHitD_CSCStripData_h
#define CSCRecHitD_CSCStripData_h

/** \class CSCStripData
 *
 * Hold strip data while building strip hits in CSCHitFromStripOnly.
 *
 */
	
#include <algorithm>
#include <functional>
#include <vector>
#include <iosfwd>

class CSCStripData
{	
 public:
		
  /** The default ctor initializes all elements of thePulseHeightMap for
   * which explicit digis do not exist.
   * Use sentinel value for istrip and tmax. 
   *
   * Note that _raw_ pulseheights are int.
   */
  CSCStripData() : istrip_(-1), phmax_(0.), tmax_(-1), phRaw_( ntbins_ ), ph_( ntbins_ ) {};
  CSCStripData( int istrip,  float phmax,  int tmax, const std::vector<int>& phRaw, const std::vector<float>& ph ) :
      istrip_(istrip), phmax_(phmax), tmax_(tmax), phRaw_(phRaw), ph_(ph) {};
   
  /// strip to which these data belong (counts from 1)
  int   strip() const {return istrip_;}
  /// maximum pulseheight in one SCA time bin
  float phmax() const {return phmax_;}
  /// the time bin in which the maximum pulseheight occurs (counts from 0)
  int   tmax()  const {return tmax_;}

  /**
   * pulseheights in the 8 SCA time bins, after pedestal subtraction and (possibly) gain-correction
   */
  const std::vector<float>& ph() const {return ph_;}
	
  /**
   * pulseheights in the 8 SCA time bins, after pedestal subtraction but without gain-correction
   */
  const std::vector<int>& phRaw() const {return phRaw_;}

  /**
   * scale pulseheights by argument, but leave raw pulseheights unchanged.
   */
  void operator*=( float factor) {
    // scale all elements of ph by 'factor'. Leaves phRaw_ unchanged.
    std::transform( ph_.begin(), ph_.end(), ph_.begin(), 
          	   std::bind2nd( std::multiplies<float>(), factor ) );
    phmax_ *= factor;
  }

  bool operator<( const CSCStripData & data ) const { return phmax_ < data.phmax_; }

  /// for debugging purposes
  friend std::ostream & operator<<(std::ostream &, const CSCStripData &);

 private:

  static const int ntbins_ = 8; //@@ Number of time bins & hence length of ph vectors
  int istrip_;
  float phmax_;
  int tmax_;
  std::vector<int> phRaw_;
  std::vector<float> ph_;

};

#endif

