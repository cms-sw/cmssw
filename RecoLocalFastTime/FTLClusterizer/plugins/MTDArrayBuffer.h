#ifndef RecoLocalTracker_MTDClusterizer_MTDArrayBuffer_H
#define RecoLocalTracker_MTDClusterizer_MTDArrayBuffer_H

//----------------------------------------------------------------------------
//! \class MTDArrayBuffer
//! \brief Class to store ADC counts and times during clustering.
//!
//----------------------------------------------------------------------------

// We use FTLHitPos which is an inner class of FTLCluster:
#include "DataFormats/FTLRecHit/interface/FTLCluster.h"

#include <vector>
#include <iostream>

class MTDArrayBuffer 
{
 public:
  inline MTDArrayBuffer( int rows, int cols);
  inline MTDArrayBuffer( ){}
  
  inline void setSize( int rows, int cols);

  inline float energy( int row, int col) const;
  inline float energy( const FTLCluster::FTLHitPos&) const;
  inline float time( int row, int col) const;
  inline float time( const FTLCluster::FTLHitPos&) const;
  inline float time_error( int row, int col) const;
  inline float time_error( const FTLCluster::FTLHitPos&) const;

  inline int rows() const { return nrows;}
  inline int columns() const { return ncols;}

  inline bool inside(int row, int col) const;

  inline void clear(int row, int col) 
  {
    set_energy( row, col, 0.);
    set_time( row, col, 0.);
    set_time_error( row, col, 0.);
  }
  inline void clear(const FTLCluster::FTLHitPos& pos) 
  {
    clear(pos.row(),pos.col());
  }

  inline void set( int row, int col, float energy, float time, float time_error);
  inline void set( const FTLCluster::FTLHitPos&, float energy, float time, float time_error);

  inline void set_energy( int row, int col, float energy);
  inline void set_energy( const FTLCluster::FTLHitPos&, float energy);
  inline void add_energy( int row, int col, float energy);

  inline void set_time( int row, int col, float time);
  inline void set_time( const FTLCluster::FTLHitPos&, float time);
  inline void add_time( int row, int col, float time);

  inline void set_time_error( int row, int col, float time_error);
  inline void set_time_error( const FTLCluster::FTLHitPos&, float time_error);
  inline void add_time_error( int row, int col, float time_error);

  int size() const { return hitEnergy_vec.size();}

  /// Definition of indexing within the buffer.
  int index( int row, int col) const {return col*nrows+row;}
  int index( const FTLCluster::FTLHitPos& pix) const { return index(pix.row(), pix.col()); }

 private:
  std::vector<float> hitEnergy_vec;   
  std::vector<float> hitTime_vec;  
  std::vector<float> hitTimeError_vec;   
  int nrows;
  int ncols;
};

MTDArrayBuffer::MTDArrayBuffer( int rows, int cols) 
  : hitEnergy_vec(rows*cols,0), hitTime_vec(rows*cols,0), hitTimeError_vec(rows*cols,0),  nrows(rows), ncols(cols) {}

void MTDArrayBuffer::setSize( int rows, int cols) {
  hitEnergy_vec.resize(rows*cols,0);
  hitTime_vec.resize(rows*cols,0);
  hitTimeError_vec.resize(rows*cols,0);
  nrows = rows;
  ncols = cols;
}

bool MTDArrayBuffer::inside(int row, int col) const 
{
  return ( row >= 0 && row < nrows && col >= 0 && col < ncols);
}

float MTDArrayBuffer::energy(int row, int col) const { return hitEnergy_vec[index(row,col)];}
float MTDArrayBuffer::energy(const FTLCluster::FTLHitPos& pix) const {return hitEnergy_vec[index(pix)];}

float MTDArrayBuffer::time(int row, int col) const  { return hitTime_vec[index(row,col)];}
float MTDArrayBuffer::time(const FTLCluster::FTLHitPos& pix) const {return hitTime_vec[index(pix)];}

float MTDArrayBuffer::time_error(int row, int col) const  { return hitTimeError_vec[index(row,col)];}
float MTDArrayBuffer::time_error(const FTLCluster::FTLHitPos& pix) const {return hitTimeError_vec[index(pix)];}

void MTDArrayBuffer::set( int row, int col, float energy, float time, float time_error) 
{
  hitEnergy_vec[index(row,col)] = energy;
  hitTime_vec[index(row,col)] = time;
  hitTimeError_vec[index(row,col)] = time_error;
}
void MTDArrayBuffer::set( const FTLCluster::FTLHitPos& pix, float energy, float time, float time_error) 
{
  set( pix.row(), pix.col(), energy, time, time_error);
}

void MTDArrayBuffer::set_energy( int row, int col, float energy) 
{
  hitEnergy_vec[index(row,col)] = energy;
}
void MTDArrayBuffer::set_energy( const FTLCluster::FTLHitPos& pix, float energy)
{
  hitEnergy_vec[index(pix)] = energy;
}
void MTDArrayBuffer::add_energy( int row, int col, float energy)
{
  hitEnergy_vec[index(row,col)] += energy;
}

void MTDArrayBuffer::set_time( int row, int col, float time) 
{
  hitTime_vec[index(row,col)] = time;
}
void MTDArrayBuffer::set_time( const FTLCluster::FTLHitPos& pix, float time)
{
  hitTime_vec[index(pix)] = time;
}
void MTDArrayBuffer::add_time( int row, int col, float time)
{
  hitTime_vec[index(row,col)] += time;
}

void MTDArrayBuffer::set_time_error( int row, int col, float time_error) 
{
  hitTimeError_vec[index(row,col)] = time_error;
}
void MTDArrayBuffer::set_time_error( const FTLCluster::FTLHitPos& pix, float time_error)
{
  hitTimeError_vec[index(pix)] = time_error;
}
void MTDArrayBuffer::add_time_error( int row, int col, float time_error)
{
  hitTimeError_vec[index(row,col)] += time_error;
}

#endif
