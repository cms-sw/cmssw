#ifndef DataFormats_CTPPS_CTPPSPixelCluster_h
#define DataFormats_CTPPS_CTPPSPixelCluster_h

/*
  \class CTPPSPixelCluster 
  \brief CTPPSPixelCluster stores the information of CTPPS Tracker clusters of 3D pixels
 Author: F.Ferro - INFN Genova - 2016
*/


#include <vector>
#include <cstdint>
#include <cassert>


class CTPPSPixelCluster {

public:

  CTPPSPixelCluster() {}
  static constexpr uint8_t MAXSPAN=255;
  static constexpr uint8_t MAXCOL=155; 
  static constexpr uint8_t MAXROW=159; 

  
CTPPSPixelCluster(uint16_t isize, uint16_t * adcs,         
		  uint8_t const * rowpos,  uint8_t const * colpos, 
		  uint8_t const  rowmin,  uint8_t const  colmin) :
  thePixelOffset(2*isize), // the pixel offset is the pixel position inside the cluster wrt rowmin (even positions) and colmin (odd positions)
    thePixelADC(adcs, adcs+isize)   
    {

      uint8_t maxCol = 0;
      uint8_t maxRow = 0;
      for (unsigned int i=0; i!=isize; ++i) {
	uint8_t rowoffset = rowpos[i]-rowmin;
	uint8_t coloffset = colpos[i]-colmin;
	thePixelOffset[i*2] = std::min(MAXSPAN, rowoffset);
	thePixelOffset[i*2+1] = std::min(MAXSPAN, coloffset);
	if (rowoffset > maxRow) maxRow = rowoffset; 
	if (coloffset > maxCol) maxCol = coloffset; 
      }
      
      
      theMinPixelRow = rowmin;
      thePixelRowSpan = std::min(maxRow, MAXSPAN);
                
      theMinPixelCol = colmin;
      thePixelColSpan = std::min(maxCol, MAXSPAN);
    }

// barycenter

  float avg_row() const {
    float qm = 0.0;
    int isize = thePixelADC.size();
    for (int i=0; i<isize; ++i)
      qm += float(thePixelADC[i]) * (thePixelOffset[i*2] + theMinPixelRow + 0.5f);
    return qm/charge();
  }
   
  float avg_col() const {
    float qm = 0.0;
    int isize = thePixelADC.size();
    for (int i=0; i<isize; ++i)
      qm += float(thePixelADC[i]) * (thePixelOffset[i*2+1]  + theMinPixelCol + 0.5f);
    return qm/charge();
  }

//cluster charge

  inline float charge() const {
    float qm = 0.0;
    int isize = thePixelADC.size();
    for (int i=0; i<isize; ++i) 
      qm += float(thePixelADC[i]);
    return qm;
  }

  // Return number of pixels.
  int size() const { return thePixelADC.size();}
   
  // Return cluster dimension in rows
  int sizeRow() const { return thePixelRowSpan +1;}
   
  // Return cluster dimension in columns
  int sizeCol() const { return thePixelColSpan +1;}

  inline int minPixelRow() const { return theMinPixelRow;}
  inline int minPixelCol() const { return theMinPixelCol;}

  
  inline int colSpan() const {return thePixelColSpan; }
  inline int rowSpan() const { return thePixelRowSpan; }

  const std::vector<uint8_t> & pixelOffset() const { return thePixelOffset;}
  const std::vector<uint16_t> & pixelADC() const { return thePixelADC;}

  int pixelRow(int i) const {
    return theMinPixelRow + thePixelOffset[i*2];
  }
  int pixelCol(int i) const {
    return theMinPixelCol + thePixelOffset[i*2+1];
  }
  int pixelADC(int i) const {
    return thePixelADC[i];
  }
  
private:

  std::vector<uint8_t>  thePixelOffset;
  std::vector<uint16_t> thePixelADC;

  
  uint8_t theMinPixelRow=MAXROW;
  uint8_t theMinPixelCol=MAXCOL; 
  uint8_t thePixelRowSpan=0; 
  uint8_t thePixelColSpan=0; 
   
};

inline bool operator<( const CTPPSPixelCluster& one, const CTPPSPixelCluster& two) {
  if ( one.minPixelRow() < two.minPixelRow() ) {
    return true;
  } else if ( one.minPixelRow() > two.minPixelRow() ) {
    return false;
  } else if ( one.minPixelCol() < two.minPixelCol() ) {
    return true;
  } else {
    return false;
  }
}

#endif
