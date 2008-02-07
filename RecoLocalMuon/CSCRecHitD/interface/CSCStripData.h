#ifndef CSCRecHitD_CSCStripData_h
#define CSCRecHitD_CSCStripData_h

/** \class CSCStripData
 *
 * Hold strip data while building strip hits.
 *
 * \author Dominique Fortin
 */	

class CSCStripData
{	
 public:	
  /* The default ctor initializes all elements of thePulseHeightMap for
   * which explicit digis do not exist.  Thus the y's must be 0.
   * Use sentinel value for x and t. 
   */
  CSCStripData() : x_(-1.),   ymax_(0.), t_(-1),  y2_(0.),  y3_(0.),  y4_(0.),  y5_(0.),  y6_(0.),  y7_(0.) {};
  CSCStripData(    float x,  float ymax,  int t, float y2, float y3, float y4, float y5, float y6, float y7)  : 
                     x_(x), ymax_(ymax),  t_(t),  y2_(y2),  y3_(y3),  y4_(y4),  y5_(y5),  y6_(y6),  y7_(y7) {};
  
  float  x()   const {return  x_;}
  float ymax() const {return ymax_;}
  int    t()   const {return  t_;}
  float y2()   const {return y2_;}
  float y3()   const {return y3_;}
  float y4()   const {return y4_;}
  float y5()   const {return y5_;}
  float y6()   const {return y6_;}
  float y7()   const {return y7_;}

  bool operator<( const CSCStripData & data ) const { return ymax_ < data.ymax_; }
  
  void operator*=( float addend) {
    ymax_ *= addend;
    y2_   *= addend;
    y3_   *= addend;
    y4_   *= addend;
    y5_   *= addend;
    y6_   *= addend;
    y7_   *= addend;
  }

  
 private:
  
  float  x_;
  float  ymax_;
  int    t_;
  float  y2_;
  float  y3_;
  float  y4_;
  float  y5_;
  float  y6_;
  float  y7_;
  
};

#endif

