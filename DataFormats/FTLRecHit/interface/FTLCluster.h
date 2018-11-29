#ifndef DataFormats_FTL_Cluster_FTLCluster_h
#define DataFormats_FTL_Cluster_FTLCluster_h

/** \class FTLCluster
 *  
 * based on SiPixelCluster
 *
 * \author Paolo Meridiani
 */

#include <vector>
#include <iostream>
#include <cstdint>
#include <cassert>

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"

class FTLCluster {
public:

  typedef DetId key_type;

  class FTLHit {
  public:
    constexpr FTLHit() : x(0), y(0), energy(0),  time(0), time_error(0) {}; // for root
    constexpr FTLHit(int hit_x, int hit_y, float hit_energy, float hit_time, float hit_time_error) :
    x(hit_x), y(hit_y), energy(hit_energy), time(hit_time), time_error(hit_time_error) {};
    uint16_t x; //row
    uint16_t y; //col
    float energy;
    float time;
    float time_error;
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
  
  //--- Position of a FTL Hit
  class FTLHitPos {
  public:
    constexpr FTLHitPos() : row_(0), col_(0) {}
    constexpr FTLHitPos(int row, int col) : row_(row) , col_(col) {}
    constexpr int row() const { return row_;}
    constexpr int col() const { return col_;}
    constexpr FTLHitPos operator+( const Shift& shift) const {
      return FTLHitPos( row() + shift.dx(), col() + shift.dy());
    }
  private:
    int row_;
    int col_;
  };
  
  
  
  static constexpr unsigned int MAXSPAN=255;
  static constexpr unsigned int MAXPOS=2047;
  
  /** Construct from a range of digis that form a cluster and from 
   *  a DetID. The range is assumed to be non-empty.
   */
  FTLCluster() {}
  
 FTLCluster(DetId id, unsigned int isize, float const * energys, float const* times, float const* time_errors,
	    uint16_t const * xpos,  uint16_t const * ypos, 
	    uint16_t const  xmin,  uint16_t const  ymin) :   
  id_(id), theHitOffset(2*isize), theHitENERGY(energys,energys+isize), theHitTIME(times,times+isize), theHitTIME_ERROR(time_errors,time_errors+isize)  {
    uint16_t maxCol = 0;
    uint16_t maxRow = 0;
    int maxHit=-1;
    float maxEnergy=-99999;
    for (unsigned int i=0; i!=isize; ++i) {
      uint16_t xoffset = xpos[i]-xmin;
      uint16_t yoffset = ypos[i]-ymin;
      theHitOffset[i*2] = std::min(uint16_t(MAXSPAN),xoffset);
      theHitOffset[i*2+1] = std::min(uint16_t(MAXSPAN),yoffset);
      if (xoffset > maxRow) maxRow = xoffset; 
      if (yoffset > maxCol) maxCol = yoffset; 
      if (theHitENERGY[i]>maxEnergy)
	{
	  maxHit=i;
	  maxEnergy=theHitENERGY[i];
	}
    }
    packRow(xmin,maxRow);
    packCol(ymin,maxCol);

    if (maxHit>=0)
      seed_=(uint8_t)std::min(uint8_t(MAXSPAN),uint8_t(maxHit));
  }
  
  // linear average position (barycenter) 
  float x() const {
    float qm = 0.0;
    int isize = theHitENERGY.size();
    for (int i=0; i<isize; ++i)
      qm += float(theHitENERGY[i]) * (theHitOffset[i*2] + minHitRow() + 0.5f);
    return qm/energy();
  }
  
  float y() const {
    float qm = 0.0;
    int isize = theHitENERGY.size();
    for (int i=0; i<isize; ++i)
      qm += float(theHitENERGY[i]) * (theHitOffset[i*2+1]  + minHitCol() + 0.5f);
    return qm/energy();
  }

  float time() const {
    float qm = 0.0;
    int isize = theHitENERGY.size();
    for (int i=0; i<isize; ++i)
      qm += float(theHitENERGY[i]) * theHitTIME[i];
    return qm/energy();
  }
  
  // Return number of hits.
  int size() const { return theHitENERGY.size();}
  
  // Return cluster dimension in the x direction.
  int sizeX() const { return rowSpan() +1;}
  
  // Return cluster dimension in the y direction.
  int sizeY() const { return colSpan() +1;}
  
  
  inline float energy() const {
    float qm = 0;
    int isize = theHitENERGY.size();
    for (int i=0; i<isize; ++i) 
      qm += theHitENERGY[i];
    return qm;
  } // Return total cluster energy.
  
  inline int minHitRow() const { return theMinHitRow;} // The min x index.
  inline int maxHitRow() const { return minHitRow() + rowSpan();} // The max x index.
  inline int minHitCol() const { return theMinHitCol;} // The min y index.
  inline int maxHitCol() const { return minHitCol() + colSpan();} // The max y index.
  
  const std::vector<uint8_t> & hitOffset() const { return theHitOffset;}
  const std::vector<float> & hitENERGY() const { return theHitENERGY;}
  const std::vector<float> & hitTIME() const { return theHitTIME;}
  const std::vector<float> & hitTIME_ERROR() const { return theHitTIME_ERROR;}
  
  // obsolete, use single hit access below
  const std::vector<FTLHit> hits() const {
    std::vector<FTLHit> oldHitVector;
    int isize = theHitENERGY.size();
    oldHitVector.reserve(isize); 
    for(int i=0; i<isize; ++i) {
      oldHitVector.push_back(hit(i));
    }
    return oldHitVector;
  }
  
  // infinite faster than above...
  FTLHit hit(int i) const {
    return FTLHit(minHitRow() + theHitOffset[i*2],
		 minHitCol() + theHitOffset[i*2+1],
		  theHitENERGY[i],
		  theHitTIME[i],
		  theHitTIME_ERROR[i]
		 );

  }
  
  FTLHit seed() const {
    return hit(seed_);
  }

private:
  
  static int overflow_(uint16_t span) { return span==uint16_t(MAXSPAN);}

public:
  
  int colSpan() const {return theHitColSpan; }
  
  int rowSpan() const { return theHitRowSpan; }
  
  const DetId& id() const { return id_; }
  const DetId& detid() const { return id(); }

  bool overflowCol() const { return overflow_(theHitColSpan); }
  
  bool overflowRow() const { return overflow_(theHitRowSpan); }
  
  bool overflow() const { return  overflowCol() || overflowRow(); }
  
  void packCol(uint16_t ymin, uint16_t yspan) {
    theMinHitCol = ymin;
    theHitColSpan = std::min(yspan, uint16_t(MAXSPAN));
  }
  void packRow(uint16_t xmin, uint16_t xspan) {
    theMinHitRow = xmin;
    theHitRowSpan = std::min(xspan, uint16_t(MAXSPAN));
  }
  
   void setClusterErrorX( float errx ) { err_x = errx; }
   void setClusterErrorY( float erry ) { err_y = erry; }
   void setClusterErrorTime( float errtime ) { err_time = errtime; }
   float getClusterErrorX() const { return err_x; }
   float getClusterErrorY() const { return err_y; }
   float getClusterErrorTime() const { return err_time; }
  
private:

  DetId id_;

  std::vector<uint8_t>  theHitOffset;
  std::vector<float> theHitENERGY;
  std::vector<float> theHitTIME;
  std::vector<float> theHitTIME_ERROR;
    
  uint16_t theMinHitRow=MAXPOS; // Minimum hit index in the x direction (low edge).
  uint16_t theMinHitCol=MAXPOS; // Minimum hit index in the y direction (left edge).
  uint8_t theHitRowSpan=0; // Span hit index in the x direction (low edge).
  uint8_t theHitColSpan=0; // Span hit index in the y direction (left edge).
  
  float err_x=-99999.9f;
  float err_y=-99999.9f;
  float err_time=-99999.9f;

  uint8_t seed_;
};


// Comparison operators  (needed by DetSetVector & SortedCollection )
inline bool operator<( const FTLCluster& one, const FTLCluster& other) {
  if(one.detid() == other.detid()) { 
    if ( one.minHitRow() < other.minHitRow() ) {
      return true;
    } else if ( one.minHitRow() > other.minHitRow() ) {
      return false;
    } else if ( one.minHitCol() < other.minHitCol() ) {
      return true;
    } else {
      return false;
    }
  }
  return one.detid() < other.detid();
}

inline bool operator<( const FTLCluster& one, const uint32_t& detid) {
  return one.detid() < detid;}

inline bool operator<( const uint32_t& detid, const FTLCluster& other) { 
  return detid < other.detid();}

#endif 
