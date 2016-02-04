#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelArrayBuffer_H
#define RecoLocalTracker_SiPixelClusterizer_SiPixelArrayBuffer_H

//----------------------------------------------------------------------------
//! \class SiPixelArrayBuffer
//! \brief Class to store ADC counts during clustering.
//!
//! This class defines the buffer where the pixel ADC are stored.
//! The size is the number of rows and cols into a
//! ROC and it is set in the PixelThresholdClusterizer
//!
//! TO DO: the chip size should be obtained in some better way.
//!
//! History:
//!    Modify the indexing to col*nrows + row. 9/01 d.k.
//!    Add setSize method to adjust array size. 3/02 d.k.
//----------------------------------------------------------------------------

// We use PixelPos which is an inner class of SiPixelCluster:
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include <vector>
#include <iostream>



class SiPixelArrayBuffer 
{
 public:
  inline SiPixelArrayBuffer( int rows, int cols);
  inline SiPixelArrayBuffer( ){}
  
  inline void setSize( int rows, int cols);
  inline int operator()( int row, int col) const;
  inline int operator()( const SiPixelCluster::PixelPos&) const;
  inline int rows() const { return nrows;}
  inline int columns() const { return ncols;}

  inline bool inside(int row, int col) const;
  inline void set_adc( int row, int col, int adc);
  inline void set_adc( const SiPixelCluster::PixelPos&, int adc);
  int size() const { return pixel_vec.size();}

  /// Definition of indexing within the buffer.
  int index( int row, int col) const {return col*nrows+row;}
  int index( const SiPixelCluster::PixelPos& pix) const { return index(pix.row(), pix.col()); }

 private:
  int nrows;
  int ncols;
  std::vector<int> pixel_vec;   // TO DO: any benefit in using shorts instead?
};



SiPixelArrayBuffer::SiPixelArrayBuffer( int rows, int cols) 
  :  nrows(rows), ncols(cols) 
{
  pixel_vec.resize(rows*cols);

  // TO DO: check this now:
  // Some STL implementations have problems with default values 
  // so a initialization loop is used instead
  std::vector<int>::iterator i=pixel_vec.begin(), iend=pixel_vec.end();
  for ( ; i!=iend; ++i) {
    *i = 0;
  }
}


void SiPixelArrayBuffer::setSize( int rows, int cols) 
{
  nrows = rows;
  ncols = cols;
  pixel_vec.resize(rows*cols);
  //std::cout << " Resize the clusterize pixel buffer " << (rows*cols) 
  //    << std::endl;

  // TO DO: check this now:
  // Some STL implementations have problems with default values 
  // so a initialization loop is used instead
  std::vector<int>::iterator i=pixel_vec.begin(), iend=pixel_vec.end();
  for ( ; i!=iend; ++i) {
    *i = 0;
  }
}


bool SiPixelArrayBuffer::inside(int row, int col) const 
{
  return ( row >= 0 && row < nrows && col >= 0 && col < ncols);
}


int SiPixelArrayBuffer::operator()(int row, int col) const 
{
  if (inside(row,col))  return pixel_vec[index(row,col)];
  else  return 0;
}


int SiPixelArrayBuffer::operator()(const SiPixelCluster::PixelPos& pix) const 
{
  if (inside( pix.row(), pix.col())) return pixel_vec[index(pix)];
  else return 0;
}


// unchecked!
void SiPixelArrayBuffer::set_adc( int row, int col, int adc) 
{
  pixel_vec[index(row,col)] = adc;
}


void SiPixelArrayBuffer::set_adc( const SiPixelCluster::PixelPos& pix, int adc)
{
  pixel_vec[index(pix)] = adc;
}


#endif
