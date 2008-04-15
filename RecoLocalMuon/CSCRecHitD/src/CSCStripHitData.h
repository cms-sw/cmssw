#ifndef CSCRecHitD_CSCStripHitData_h
#define CSCRecHitD_CSCStripHitData_h

/** \class CSCStripHitData
 *
 * Hold strip hit data while building strip hit clusters.
 *
 * Note that the peak is set to occur at y1
 *
 * \author Dominique Fortin
 * modified - Stoyan Stoynev
 */	

class CSCStripHitData
{	
 public:	

  CSCStripHitData() : x_(-1.),  y0_(0.),  y1_(0.),  y2_(0.),  y3_(0.), 
    y0Raw_(0.),  y1Raw_(0.),  y2Raw_(0.),  y3Raw_(0.), t_(-1) {};
  CSCStripHitData(    float x, float y0, float y1, float y2, float y3, 
		      float y0Raw, float y1Raw, float y2Raw, float y3Raw, int t)   : 
                        x_(x),  y0_(y0),  y1_(y1),  y2_(y2),  y3_(y3),  
    y0Raw_(y0Raw),  y1Raw_(y1Raw),  y2Raw_(y2Raw),  y3Raw_(y3Raw), t_(t) {};
  
  float  x()   const {return  x_;}
  float y0()   const {return y0_;}
  float y1()   const {return y1_;}
  float y2()   const {return y2_;}
  float y3()   const {return y3_;}
  float y0Raw()   const {return y0Raw_;}
  float y1Raw()   const {return y1Raw_;}
  float y2Raw()   const {return y2Raw_;}
  float y3Raw()   const {return y3Raw_;}
  int    t()   const {return  t_;}

  bool operator<( const CSCStripHitData & data ) const { return y1_ < data.y1_; }

 
 private:
  
  float   x_;
  float  y0_;
  float  y1_;
  float  y2_;
  float  y3_;
  float  y0Raw_;
  float  y1Raw_;
  float  y2Raw_;
  float  y3Raw_;
  int     t_;
  
};

#endif

