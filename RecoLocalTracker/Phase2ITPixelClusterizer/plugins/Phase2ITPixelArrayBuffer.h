#ifndef RecoLocalTracker_Phase2ITPixelClusterizer_Phase2ITPixelArrayBuffer_H
#define RecoLocalTracker_Phase2ITPixelClusterizer_Phase2ITPixelArrayBuffer_H

//----------------------------------------------------------------------------
//! \class Phase2ITPixelArrayBuffer
//! \brief Class to store ADC counts during clustering.
//!
//! This class defines the buffer where the pixel ADC are stored.
//! The size is the number of rows and cols into a
//! ROC and it is set in the PhasePixelThresholdClusterizer
//!
//! TO DO: the chip size should be obtained in some better way.
//!
//! History:
//!    Modify the indexing to col*nrows + row. 9/01 d.k.
//!    Add setSize method to adjust array size. 3/02 d.k.
//----------------------------------------------------------------------------

// We use PixelPos which is an inner class of Phase2ITPixelCluster:
#include "DataFormats/Phase2ITPixelCluster/interface/Phase2ITPixelCluster.h"

#include <vector>
#include <iostream>



class Phase2ITPixelArrayBuffer 
{
 public:
  inline Phase2ITPixelArrayBuffer( int rows, int cols);
  inline Phase2ITPixelArrayBuffer( ){}
  
  inline void setSize( int rows, int cols);
  inline int operator()( int row, int col) const;
  inline int operator()( const Phase2ITPixelCluster::PixelPos&) const;
  inline int rows() const { return nrows;}
  inline int columns() const { return ncols;}

  inline bool inside(int row, int col) const;
  inline void set_adc( int row, int col, int adc);
  inline void set_adc( const Phase2ITPixelCluster::PixelPos&, int adc);
  int size() const { return pixel_vec.size();}

  /// Definition of indexing within the buffer.
  int index( int row, int col) const {return col*nrows+row;}
  int index( const Phase2ITPixelCluster::PixelPos& pix) const { return index(pix.row(), pix.col()); }

 private:
  std::vector<int> pixel_vec;   // TO DO: any benefit in using shorts instead?
  int nrows;
  int ncols;
};



Phase2ITPixelArrayBuffer::Phase2ITPixelArrayBuffer( int rows, int cols) 
  : pixel_vec(rows*cols,0),  nrows(rows), ncols(cols) {}


void Phase2ITPixelArrayBuffer::setSize( int rows, int cols) {
  pixel_vec.resize(rows*cols,0);
  nrows = rows;
  ncols = cols;
}


bool Phase2ITPixelArrayBuffer::inside(int row, int col) const 
{
  return ( row >= 0 && row < nrows && col >= 0 && col < ncols);
}


int Phase2ITPixelArrayBuffer::operator()(int row, int col) const  { return pixel_vec[index(row,col)];}


int Phase2ITPixelArrayBuffer::operator()(const Phase2ITPixelCluster::PixelPos& pix) const {return pixel_vec[index(pix)];}

// unchecked!
void Phase2ITPixelArrayBuffer::set_adc( int row, int col, int adc) 
{
  pixel_vec[index(row,col)] = adc;
}


void Phase2ITPixelArrayBuffer::set_adc( const Phase2ITPixelCluster::PixelPos& pix, int adc)
{
  pixel_vec[index(pix)] = adc;
}


#endif
