#ifndef RecoLocalTracker_FTLClusterizer_MTDArrayBuffer_H
#define RecoLocalTracker_FTLClusterizer_MTDArrayBuffer_H

//----------------------------------------------------------------------------
//! \class MTDArrayBuffer
//! \brief Class to store ADC counts and times during clustering.
//!
//----------------------------------------------------------------------------

// We use FTLHitPos which is an inner class of FTLCluster:
#include "DataFormats/FTLRecHit/interface/FTLCluster.h"

#include <vector>
#include <iostream>

class MTDArrayBuffer {
public:
  typedef unsigned int uint;

  inline MTDArrayBuffer(uint rows, uint cols);
  inline MTDArrayBuffer() {}

  inline void setSize(uint rows, uint cols);

  inline float energy(uint row, uint col) const;
  inline float energy(const FTLCluster::FTLHitPos&) const;
  inline float time(uint row, uint col) const;
  inline float time(const FTLCluster::FTLHitPos&) const;
  inline float time_error(uint row, uint col) const;
  inline float time_error(const FTLCluster::FTLHitPos&) const;

  inline uint rows() const { return nrows; }
  inline uint columns() const { return ncols; }

  inline bool inside(uint row, uint col) const;

  inline void clear(uint row, uint col) {
    set_energy(row, col, 0.);
    set_time(row, col, 0.);
    set_time_error(row, col, 0.);
  }
  inline void clear(const FTLCluster::FTLHitPos& pos) { clear(pos.row(), pos.col()); }

  inline void set(uint row, uint col, float energy, float time, float time_error);
  inline void set(const FTLCluster::FTLHitPos&, float energy, float time, float time_error);

  inline void set_energy(uint row, uint col, float energy);
  inline void set_energy(const FTLCluster::FTLHitPos&, float energy);
  inline void add_energy(uint row, uint col, float energy);

  inline void set_time(uint row, uint col, float time);
  inline void set_time(const FTLCluster::FTLHitPos&, float time);

  inline void set_time_error(uint row, uint col, float time_error);
  inline void set_time_error(const FTLCluster::FTLHitPos&, float time_error);

  uint size() const { return hitEnergy_vec.size(); }

  /// Definition of indexing within the buffer.
  uint index(uint row, uint col) const { return col * nrows + row; }
  uint index(const FTLCluster::FTLHitPos& pix) const { return index(pix.row(), pix.col()); }

private:
  std::vector<float> hitEnergy_vec;
  std::vector<float> hitTime_vec;
  std::vector<float> hitTimeError_vec;
  uint nrows;
  uint ncols;
};

MTDArrayBuffer::MTDArrayBuffer(uint rows, uint cols)
    : hitEnergy_vec(rows * cols, 0),
      hitTime_vec(rows * cols, 0),
      hitTimeError_vec(rows * cols, 0),
      nrows(rows),
      ncols(cols) {}

void MTDArrayBuffer::setSize(uint rows, uint cols) {
  hitEnergy_vec.resize(rows * cols, 0);
  hitTime_vec.resize(rows * cols, 0);
  hitTimeError_vec.resize(rows * cols, 0);
  nrows = rows;
  ncols = cols;
}

bool MTDArrayBuffer::inside(uint row, uint col) const { return (row < nrows && col < ncols); }

float MTDArrayBuffer::energy(uint row, uint col) const { return hitEnergy_vec[index(row, col)]; }
float MTDArrayBuffer::energy(const FTLCluster::FTLHitPos& pix) const { return hitEnergy_vec[index(pix)]; }

float MTDArrayBuffer::time(uint row, uint col) const { return hitTime_vec[index(row, col)]; }
float MTDArrayBuffer::time(const FTLCluster::FTLHitPos& pix) const { return hitTime_vec[index(pix)]; }

float MTDArrayBuffer::time_error(uint row, uint col) const { return hitTimeError_vec[index(row, col)]; }
float MTDArrayBuffer::time_error(const FTLCluster::FTLHitPos& pix) const { return hitTimeError_vec[index(pix)]; }

void MTDArrayBuffer::set(uint row, uint col, float energy, float time, float time_error) {
  hitEnergy_vec[index(row, col)] = energy;
  hitTime_vec[index(row, col)] = time;
  hitTimeError_vec[index(row, col)] = time_error;
}
void MTDArrayBuffer::set(const FTLCluster::FTLHitPos& pix, float energy, float time, float time_error) {
  set(pix.row(), pix.col(), energy, time, time_error);
}

void MTDArrayBuffer::set_energy(uint row, uint col, float energy) { hitEnergy_vec[index(row, col)] = energy; }
void MTDArrayBuffer::set_energy(const FTLCluster::FTLHitPos& pix, float energy) { hitEnergy_vec[index(pix)] = energy; }
void MTDArrayBuffer::add_energy(uint row, uint col, float energy) { hitEnergy_vec[index(row, col)] += energy; }

void MTDArrayBuffer::set_time(uint row, uint col, float time) { hitTime_vec[index(row, col)] = time; }
void MTDArrayBuffer::set_time(const FTLCluster::FTLHitPos& pix, float time) { hitTime_vec[index(pix)] = time; }

void MTDArrayBuffer::set_time_error(uint row, uint col, float time_error) {
  hitTimeError_vec[index(row, col)] = time_error;
}
void MTDArrayBuffer::set_time_error(const FTLCluster::FTLHitPos& pix, float time_error) {
  hitTimeError_vec[index(pix)] = time_error;
}

#endif
