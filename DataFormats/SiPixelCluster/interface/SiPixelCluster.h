#ifndef DataFormats_SiPixel_Cluster_SiPixelCluster_h
#define DataFormats_SiPixel_Cluster_SiPixelCluster_h

//---------------------------------------------------------------------------
//!  \class SiPixelCluster
//!  \brief Pixel cluster -- collection of pixels with ADC counts + misc info.
//!
//!  Class to contain and store all the topological information of pixel clusters:
//!  charge, global size, size and the barycenter in x and y
//!  local directions. It builds a vector of SiPixel (which is
//!  an inner class) and a container of channels. 
//!
//!  Mostly ported from ORCA's class PixelReco::Cluster.
//!
//!  \author Petar Maksimovic, JHU
//---------------------------------------------------------------------------

#include <vector>
class PixelDigi;

class SiPixelCluster {
 public:
  
  class Pixel {
  public:
    Pixel(){} // for root
    Pixel( float pix_x, float pix_y, float pix_adc) :
      x(pix_x), y(pix_y), adc(pix_adc) {}
    float x;
    float y;
    float adc;
  };
  
  //--- Integer shift in x and y directions.
  class Shift {
  public:
    Shift( int dx, int dy) : dx_(dx), dy_(dy) {}
    Shift() : dx_(0), dy_(0) {}
    int dx() const { return dx_;}
    int dy() const { return dy_;}
  private:
    int dx_;
    int dy_;
  };
  
  //--- Position of a SiPixel
  class PixelPos {
  public:
    PixelPos() : row_(0), col_(0) {}
    PixelPos(int row, int col) : row_(row) , col_(col) {}
    int row() const { return row_;}
    int col() const { return col_;}
    PixelPos operator+( const Shift& shift) {
      return PixelPos( row() + shift.dx(), col() + shift.dy());
    }
  private:
    int row_;
    int col_;
  };
  
  


  typedef std::vector<PixelDigi>::const_iterator   PixelDigiIter;
  typedef std::pair<PixelDigiIter,PixelDigiIter>   PixelDigiRange;
  
  /** Construct from a range of digis that form a cluster and from 
   *  a DetID. The range is assumed to be non-empty.
   */
  
  // &&& Decide the fate of the two strip-like constructors below:
  SiPixelCluster() : detId_(0) {}  // needed by vector::push_back()!
  // SiPixelCluster( unsigned int detid, const PixelDigiRange& range)
    
  SiPixelCluster( const PixelPos& pix, float adc);
  
  void add( const PixelPos& pix, float adc);
  
  // Analog linear average position (barycenter) 
  float x() const { return theSumX / theCharge;}
  float y() const { return theSumY / theCharge;}

  // Return number of pixels.
  int size() const { return thePixels.size();}

  // Return cluster dimension in the x direction.
  int sizeX() const {return theMaxPixelRow - theMinPixelRow +1;}

  // Return cluster dimension in the y direction.
  int sizeY() const {return theMaxPixelCol - theMinPixelCol +1;}

  // Detect clusters at the edge of the detector
  bool edgeHitX() const;
  bool edgeHitY() const;

  inline float charge() const { return theCharge;} // Return total cluster charge.
  inline int minPixelRow() const { return theMinPixelRow;} // The min x index.
  inline int maxPixelRow() const { return theMaxPixelRow;} // The max x index.
  inline int minPixelCol() const { return theMinPixelCol;} // The min y index.
  inline int maxPixelCol() const { return theMaxPixelCol;} // The max y index.
  
  const std::vector<Pixel> & pixels() const { return thePixels;}

  //--- Cloned fom Strips:
  
  /** The geographical ID of the corresponding DetUnit, 
   *  to be used for transformations to local and to global reference 
   *  frames etc.   */
  unsigned int geographicalId() const {return detId_;}
  
  // &&& Decide if we still need these two:
  // typedef vector<Digi::ChannelType>    ChannelContainer;
  // ChannelContainer  channels() const;
  
 private:
  unsigned int         detId_;
  
  std::vector<Pixel> thePixels;
  
  float theSumX;  // Sum of charge weighted pixel positions.
  float theSumY;
  float theCharge;  // Total charge
  int theMinPixelRow; // Minimum pixel index in the x direction (low edge).
  int theMaxPixelRow; // Maximum pixel index in the x direction (top edge).
  int theMinPixelCol; // Minimum pixel index in the y direction (left edge).
  int theMaxPixelCol; // Maximum pixel index in the y direction (right edge).
  
};



// Comparison operators
inline bool operator<( const SiPixelCluster& one, const SiPixelCluster& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else if ( one.geographicalId() > other.geographicalId() ) {
    return false;
  } else if ( one.minPixelRow() < other.minPixelRow() ) {
    return true;
  } else if ( one.minPixelRow() > other.minPixelRow() ) {
    return false;
  } else if ( one.minPixelCol() <= other.minPixelCol() ) {
    return true;
  } else {
    return false;
  }
}



#endif // DATAFORMATS_SISTRIPCLUSTER_H
