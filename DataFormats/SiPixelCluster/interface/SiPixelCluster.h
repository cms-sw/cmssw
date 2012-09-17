#ifndef DataFormats_SiPixel_Cluster_SiPixelCluster_h
#define DataFormats_SiPixel_Cluster_SiPixelCluster_h

//---------------------------------------------------------------------------
//!  \class SiPixelCluster
//!  \brief Pixel cluster -- collection of neighboring pixels above threshold
//!
//!  Class to contain and store all the topological information of pixel clusters:
//!  charge, global size, size and the barycenter in x and y
//!  local directions. It builds a vector of SiPixel (which is
//!  an inner class) and a container of channels. 
//!
//!  March 2007: Edge methods moved to RectangularPixelTopology class (V.Chiochia)
//!  Feb 2008: Modify the Pixel class from float to shorts
//!  May   2008: Offset based packing (D.Fehling / A. Rizzi)
//!  Sep 2012: added Max back, removed detId (V.I.)
//---------------------------------------------------------------------------

#include <vector>
#include "boost/cstdint.hpp"
#include <cassert>
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class PixelDigi;

class SiPixelCluster {
 public:
  
  class Pixel {
  public:
    constexpr Pixel() : x(0), y(0), adc(0){} // for root
    constexpr Pixel(int pix_x, int pix_y, int pix_adc) :
      x(pix_x), y(pix_y), adc(pix_adc) {}
    unsigned char  x;
    unsigned short y;
    unsigned short adc;
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
    constexpr int row() const { return row_;}
    constexpr int col() const { return col_;}
    constexpr PixelPos operator+( const Shift& shift) {
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
  
  SiPixelCluster() : theMinPixelRow(255), theMaxPixelRow(0), thePixelCol(511), err_x(-99999.9), err_y(-99999.9) {}  // needed by many....
    
  SiPixelCluster( const PixelPos& pix, int adc);
  
  void add( const PixelPos& pix, int adc);
  
  // Analog linear average position (barycenter) 
  float x() const {
    float qm = 0.0;
    int isize = thePixelADC.size();
    for (int i=0; i<isize; ++i)
      qm += float(thePixelADC[i]) * (thePixelOffset[i*2] + minPixelRow() + 0.5);
    return qm/charge();
  }
  float y() const {
    float qm = 0.0;
    int isize = thePixelADC.size();
    for (int i=0; i<isize; ++i)
      qm += float(thePixelADC[i]) * (thePixelOffset[i*2+1]  + minPixelCol() + 0.5);
    return qm/charge();
  }

  // Return number of pixels.
  int size() const { return thePixelADC.size();}

  // Return cluster dimension in the x direction.
  int sizeX() const {return maxPixelRow() - minPixelRow() +1;}

  // Return cluster dimension in the y direction.
  int sizeY() const {return int(thePixelCol >>9) +1;}


  inline float charge() const {
    float qm = 0.0;
    int isize = thePixelADC.size();
    for (int i=0; i<isize; ++i) 
      qm += float(thePixelADC[i]);
    return qm;
  } // Return total cluster charge.

  inline int minPixelRow() const { return theMinPixelRow;} // The min x index.
  inline int maxPixelRow() const { verifyVersion(); return theMaxPixelRow;} // The max x index.
  inline int minPixelCol() const { return thePixelCol & 511;} // The min y index.
  inline int maxPixelCol() const { verifyVersion(); return minPixelCol() + colSpan();} // The max y index.

  
  const std::vector<uint8_t> & pixelOffset() const { return thePixelOffset;}
  const std::vector<uint16_t> & pixelADC() const { return thePixelADC;}

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

  int colSpan() const {
    if unlikely(overflowCol()) return  computeColSpan();
    return (thePixelCol >>9);
  }

  bool overflowCol() const {
    return (thePixelCol >>9) == 127;
  }

  void packCol(unsigned int ymin, unsigned int yspan) {
    yspan = std::min(yspan,(unsigned int)(127));
    thePixelCol = (yspan<<9) | ymin;
  }

  int computeColSpan() const {
    int maxCol = 0;
    int isize  = thePixelADC.size();
    for (int i=0; i!=isize; ++i) {
      int ysize = thePixelOffset[i*2+1] ;
      if (ysize > maxCol) maxCol = ysize;
    }
    return maxCol;
  }

  /// mostly to be compatible for <610 
  void verifyVersion() const {
    if unlikely( theMaxPixelRow==0 && thePixelCol<511)
      const_cast<SiPixelCluster*>(this)->computeMax();
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
    theMaxPixelRow=maxRow+minPixelRow();
    int minCol= minPixelCol();
    packCol(minCol,maxCol);
  }
  
  // ggiurgiu@fnal.gov, 01/05/12 
  // Getters and setters for the newly added data members (err_x and err_y). See below. 
  void setSplitClusterErrorX( float errx ) { err_x = errx; }
  void setSplitClusterErrorY( float erry ) { err_y = erry; }
  float getSplitClusterErrorX() const { return err_x; }
  float getSplitClusterErrorY() const { return err_y; }
  

 private:
  
  std::vector<uint8_t>  thePixelOffset;
  std::vector<uint16_t> thePixelADC;
  

  uint8_t  theMinPixelRow; // Minimum pixel index in the x direction (low edge).
  uint8_t  theMaxPixelRow; // Maximum pixel index in the x direction (low edge).
  uint16_t thePixelCol; // Minimum and span pixel index in the y direction (left edge).
  // Need 9 bits for Col information the other 7 used for span

  // ggiurgiu@fnal.gov, 01/05/12
  // Add cluster errors to be used by rechits from split clusters. 
  // A rechit from a split cluster has larger errors than rechits from normal clusters. 
  // However, when presented with a cluster, the CPE does not know if the cluster comes 
  // from a splitting procedure or not. That's why we have to instruct the CPE to use 
  // appropriate errors for split clusters.
  // To avoid increase of data size on disk,these new data members are set as transient in: 
  // DataFormats/SiPixelCluster/src/classes_def.xml
  float err_x;
  float err_y;
  
};

// Comparison operators  (no clue...)
inline bool operator<( const SiPixelCluster& one, const SiPixelCluster& other) {
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

typedef edm::DetSetVector<SiPixelCluster> SiPixelClusterCollection;
typedef edm::Ref<SiPixelClusterCollection, SiPixelCluster> SiPixelClusterRef;
typedef edm::DetSetRefVector<SiPixelCluster> SiPixelClusterRefVector;
typedef edm::RefProd<SiPixelClusterCollection> SiPixelClusterRefProd;

typedef edmNew::DetSetVector<SiPixelCluster> SiPixelClusterCollectionNew;
typedef edm::Ref<SiPixelClusterCollectionNew, SiPixelCluster> SiPixelClusterRefNew;
#endif 
