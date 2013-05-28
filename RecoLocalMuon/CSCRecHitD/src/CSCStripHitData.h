#ifndef CSCRecHitD_CSCStripHitData_h
#define CSCRecHitD_CSCStripHitData_h

/** \class CSCStripHitData
 *
 * Hold strip hit data while building strip hit clusters.
 * Only 4 timebins of the SCA adc pulseheights are stored.
 *
 * tmax() returns the bin with max pulseheight, and we store
 * bins tmax-1 to tmax+2. Therefore the peak bin is the 2nd.
 *
 */	

#include <vector>

class CSCStripHitData
{	
 public:	

  CSCStripHitData() : istrip_(-1), tmax_(-1), phRaw_(nbins_), ph_(nbins_){};
  
  CSCStripHitData( int istrip, int tmax, const std::vector<float>& phRaw,  const std::vector<float>& ph ) :
           istrip_(istrip), tmax_(tmax), phRaw_(phRaw),  ph_(ph){};
  
  int strip() const {return istrip_;}
  int tmax() const {return  tmax_;}
  std::vector<float> ph() const { return ph_;}
  std::vector<float> phRaw() const { return phRaw_;}

  /// Order by 2nd ph bin
  bool operator<( const CSCStripHitData & data ) const { return ph_[1] < data.ph_[1]; }

 private:
  static const int nbins_ = 4; //@ Number of ph bins saved
  int istrip_;
  int tmax_;
  std::vector<float> phRaw_;
  std::vector<float> ph_;  
};

#endif

