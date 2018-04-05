#ifndef DataFormats_Phase2ITPixelCluster_Phase2ITPixelCluster_h
#define DataFormats_Phase2ITPixelCluster_Phase2ITPixelCluster_h

//---------------------------------------------------------------------------
//!  \class Phase2ITPixelCluster
//!  \brief Pixel cluster -- collection of neighboring pixels above threshold
//!
//!  Class to contain and store all the topological information of pixel clusters:
//!  charge, global size, size and the barycenter in x and y
//!  local directions. It builds a vector of Pixel (which is
//!  an inner class) and a container of channels. 
//---------------------------------------------------------------------------

#include <vector>
#include "FWCore/Utilities/interface/GCC11Compatibility.h"


#include <cstdint>
#include <cassert>

class PixelDigi;

class Phase2ITPixelCluster {
public:
  
  class Pixel {
  public:
    constexpr Pixel() : x(0), y(0), adc(0){} // for root
    constexpr Pixel(uint32_t pix_x, uint32_t pix_y, uint32_t pix_adc) :
      x(pix_x), y(pix_y), adc(pix_adc) {}
    uint32_t x;
    uint32_t y;
    uint32_t adc;
  };
  
  //--- Integer shift in x and y directions.
  class Shift {
  public:
    constexpr Shift( int dx, int dy) : dx_(dx), dy_(dy) {}
    constexpr Shift() : dx_(0), dy_(0) {}
    constexpr int dx() const { return dx_;}
    constexpr int dy() const { return dy_;}
  private:
    int dx_;
    int dy_;
  };
  
  //--- Position of a SiPixel
  class PixelPos {
  public:
    constexpr PixelPos() : row_(0), col_(0) {}
    constexpr PixelPos(int row, int col) : row_(row) , col_(col) {}
    constexpr uint32_t row() const { return row_;}
    constexpr uint32_t col() const { return col_;}
    constexpr PixelPos operator+( const Shift& shift) const {
      return PixelPos( row() + shift.dx(), col() + shift.dy());
    }
  private:
    uint32_t row_;
    uint32_t col_;
  };
  
  typedef std::vector<PixelDigi>::const_iterator   PixelDigiIter;
  typedef std::pair<PixelDigiIter,PixelDigiIter>   PixelDigiRange;
    
  static constexpr unsigned int POSBITS=20;
  static constexpr unsigned int SPANBITS=12;
  static constexpr unsigned int MAXSPAN=255;
  static constexpr unsigned int MAXPOS=2047;
  
  /** Construct from a range of digis that form a cluster and from 
   *  a DetID. The range is assumed to be non-empty.
   */
  
  Phase2ITPixelCluster() : thePixelRow(MAXPOS), thePixelCol(MAXPOS), err_x(-99999.9), err_y(-99999.9) {}  // needed by many....
  
  Phase2ITPixelCluster(unsigned int isize, uint32_t const * adcs,
		 uint32_t const * xpos,  uint32_t const * ypos, 
		 uint32_t const  xmin,  uint32_t const  ymin) :   
    thePixelOffset(2*isize), thePixelADC(adcs,adcs+isize), err_x(-99999.9), err_y(-99999.9) {
    uint32_t maxCol = 0;
    uint32_t maxRow = 0;
    for (unsigned int i=0; i!=isize; ++i) {
      uint32_t xoffset = xpos[i]-xmin;
      uint32_t yoffset = ypos[i]-ymin;
      thePixelOffset[i*2] = std::min(uint32_t(MAXSPAN),xoffset);
      thePixelOffset[i*2+1] = std::min(uint32_t(MAXSPAN),yoffset);
      if (xoffset > maxRow) maxRow = xoffset; 
      if (yoffset > maxCol) maxCol = yoffset; 
    }
    packRow(xmin,maxRow);
    packCol(ymin,maxCol);
  }
  
  
  // obsolete (only for regression tests)
  Phase2ITPixelCluster( const PixelPos& pix, uint32_t adc);
  void add( const PixelPos& pix, uint32_t adc);
  
  // Analog linear average position (barycenter) 
  float x() const {
    float qm = 0.0;
    int isize = thePixelADC.size();
    for (int i=0; i<isize; ++i)
      qm += float(thePixelADC[i]) * (thePixelOffset[i*2] + minPixelRow() + 0.5f);
    return qm/charge();
  }
  
  float y() const {
    float qm = 0.0;
    int isize = thePixelADC.size();
    for (int i=0; i<isize; ++i)
      qm += float(thePixelADC[i]) * (thePixelOffset[i*2+1]  + minPixelCol() + 0.5f);
    return qm/charge();
  }
  
  // Return number of pixels.
  int size() const { return thePixelADC.size();}
  
  // Return cluster dimension in the x direction.
  int sizeX() const {verifyVersion(); return rowSpan() +1;}
  
  // Return cluster dimension in the y direction.
  int sizeY() const {verifyVersion(); return colSpan() +1;}
  
  
  inline float charge() const {
    float qm = 0.0;
    int isize = thePixelADC.size();
    for (int i=0; i<isize; ++i) 
      qm += float(thePixelADC[i]);
    return qm;
  } // Return total cluster charge.
  
  inline uint32_t minPixelRow() const { return thePixelRow&MAXPOS;} // The min x index.
  inline uint32_t maxPixelRow() const { verifyVersion(); return minPixelRow() + rowSpan();} // The max x index.
  inline uint32_t minPixelCol() const { return thePixelCol&MAXPOS;} // The min y index.
  inline uint32_t maxPixelCol() const { verifyVersion(); return minPixelCol() + colSpan();} // The max y index.
  
  
  const std::vector<uint16_t> & pixelOffset() const { return thePixelOffset;}
  const std::vector<uint32_t> & pixelADC() const { return thePixelADC;}
  
  // obsolete, use single pixel access below
  const std::vector<Pixel> pixels() const {
    std::vector<Pixel> oldPixVector;
    int isize = thePixelADC.size();
    oldPixVector.reserve(isize); 
    for(int i=0; i<isize; ++i) {
      oldPixVector.push_back(pixel(i));
    }
    return oldPixVector;
  }
  
  // infinite faster than above...
  Pixel pixel(int i) const {
    return Pixel(minPixelRow() + thePixelOffset[i*2],
		 minPixelCol() + thePixelOffset[i*2+1],
		 thePixelADC[i]
		 );
  }
  
private:
  
  static int span_(uint32_t packed) { return packed >> POSBITS;}
  static int overflow_(uint32_t packed) { return span_(packed)==uint32_t(MAXSPAN);}
  static uint32_t pack_(uint32_t zmin, unsigned  int zspan) {
    zspan = std::min(zspan, uint32_t(MAXSPAN));
    return (zspan<<POSBITS) | zmin;
  }
public:
  
  int colSpan() const {return span_(thePixelCol); }
  
  int rowSpan() const { return span_(thePixelRow); }
  
  
  bool overflowCol() const { return overflow_(thePixelCol); }
  
  bool overflowRow() const { return overflow_(thePixelRow); }
  
  bool overflow() const { return  overflowCol() || overflowRow(); }
  
  void packCol(uint32_t ymin, uint32_t yspan) {
    thePixelCol = pack_(ymin,yspan);
  }
  void packRow(uint32_t xmin, uint32_t xspan) {
    thePixelRow = pack_(xmin,xspan);
  }
  
  
  
  /// mostly to be compatible for <610 
  void verifyVersion() const {
    if unlikely( thePixelRow<MAXPOS && thePixelCol<MAXPOS)
		 const_cast<Phase2ITPixelCluster*>(this)->computeMax();
  }
  
  /// moslty to be compatible for <610 
  void computeMax()  {
    int maxRow = 0;
    int maxCol = 0;
    int isize  = thePixelADC.size();
    for (int i=0; i!=isize; ++i) {
      int xsize  = thePixelOffset[i*2];
      if (xsize > maxRow) maxRow = xsize;
      int ysize = thePixelOffset[i*2+1] ;
      if (ysize > maxCol) maxCol = ysize;
    }
    // assume minimum is correct
    uint32_t minCol= minPixelCol();
    packCol(minCol,maxCol);
    uint32_t minRow= minPixelRow();
    packRow(minRow,maxRow);
  }
  
  // Getters and setters for the newly added data members (err_x and err_y). See below. 
  void setSplitClusterErrorX( float errx ) { err_x = errx; }
  void setSplitClusterErrorY( float erry ) { err_y = erry; }
  float getSplitClusterErrorX() const { return err_x; }
  float getSplitClusterErrorY() const { return err_y; }
  
  
private:
  
  std::vector<uint16_t> thePixelOffset;
  std::vector<uint32_t> thePixelADC;
  
  
  uint32_t thePixelRow; // Minimum and span pixel index in the x direction (low edge).
  uint32_t thePixelCol; // Minimum and span pixel index in the y direction (left edge).
  // Need 10 bits for Position information. the other 6 used for span
  
  // A rechit from a split cluster should have larger errors than rechits from normal clusters. 
  // However, when presented with a cluster, the CPE does not know if the cluster comes 
  // from a splitting procedure or not. That's why we have to instruct the CPE to use 
  // appropriate errors for split clusters.
  float err_x;
  float err_y;
  
};

// Comparison operators  (no clue...)
inline bool operator<( const Phase2ITPixelCluster& one, const Phase2ITPixelCluster& other) {
  if ( one.minPixelRow() < other.minPixelRow() ) {
    return true;
  } else if ( one.minPixelRow() > other.minPixelRow() ) {
    return false;
  } else if ( one.minPixelCol() < other.minPixelCol() ) {
    return true;
  } else {
    return false;
  }
}

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSetRefVector.h"

typedef edm::DetSetVector<Phase2ITPixelCluster> Phase2ITPixelClusterCollection;
typedef edm::Ref<Phase2ITPixelClusterCollection, Phase2ITPixelCluster> Phase2ITPixelClusterRef;
typedef edm::DetSetRefVector<Phase2ITPixelCluster> Phase2ITPixelClusterRefVector;
typedef edm::RefProd<Phase2ITPixelClusterCollection> Phase2ITPixelClusterRefProd;

typedef edmNew::DetSetVector<Phase2ITPixelCluster> Phase2ITPixelClusterCollectionNew;
typedef edm::Ref<Phase2ITPixelClusterCollectionNew, Phase2ITPixelCluster> Phase2ITPixelClusterRefNew;
#endif 
