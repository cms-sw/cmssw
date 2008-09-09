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

	CSCStripHitData() : istrip_(-1), tmax_(-1), ph_(std::vector<float>()), phRaw_(ph_){};
  
  CSCStripHitData( int istrip, int tmax, std::vector<float> ph,  std::vector<float> phRaw ) :
           istrip_(istrip), tmax_(tmax), ph_(ph),  phRaw_(phRaw){};
  
  int strip() const {return istrip_;}
  int tmax() const {return  tmax_;}
	std::vector<float> ph() const { return ph_;}
	std::vector<float> phRaw() const { return phRaw_;}

  /// Order by 2nd ph bin
  bool operator<( const CSCStripHitData & data ) const { return ph_[1] < data.ph_[1]; }

 private:
  
  int istrip_;
  int tmax_;
	std::vector<float> ph_;
	std::vector<float> phRaw_;
  
};

#endif

