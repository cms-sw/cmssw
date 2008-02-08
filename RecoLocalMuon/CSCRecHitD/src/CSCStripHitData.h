#ifndef CSCRecHitD_CSCStripHitData_h
#define CSCRecHitD_CSCStripHitData_h

/** \class CSCStripHitData
 *
 * Hold strip hit data while building strip hit clusters.
 *
 * Note that the peak is set to occur at y1
 *
 * \author Dominique Fortin
 */	

class CSCStripHitData
{	
 public:	

  CSCStripHitData() : x_(-1.),  y0_(0.),  y1_(0.),  y2_(0.),  y3_(0.), t_(-1) {};
  CSCStripHitData(    float x, float y0, float y1, float y2, float y3, int t)   : 
                        x_(x),  y0_(y0),  y1_(y1),  y2_(y2),  y3_(y3),  t_(t) {};
  
  float  x()   const {return  x_;}
  float y0()   const {return y0_;}
  float y1()   const {return y1_;}
  float y2()   const {return y2_;}
  float y3()   const {return y3_;}
  int    t()   const {return  t_;}

  bool operator<( const CSCStripHitData & data ) const { return y1_ < data.y1_; }

 
 private:
  
  float   x_;
  float  y0_;
  float  y1_;
  float  y2_;
  float  y3_;
  int     t_;
  
};

#endif

