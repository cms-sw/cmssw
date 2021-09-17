#ifndef RecoLocalTracker_FTLClusterizer_MTDArrayBuffer_H
#define RecoLocalTracker_FTLClusterizer_MTDArrayBuffer_H

//----------------------------------------------------------------------------
//! \class MTDArrayBuffer
//! \brief Class to store ADC counts and times during clustering.
//!
//----------------------------------------------------------------------------

// We use FTLHitPos which is an inner class of FTLCluster:
#include "DataFormats/FTLRecHit/interface/FTLCluster.h"

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

#include <vector>
#include <iostream>

class MTDArrayBuffer {
public:
  typedef unsigned int uint;

  inline MTDArrayBuffer(uint rows, uint cols);
  inline MTDArrayBuffer() {}

  inline void setSize(uint rows, uint cols);

  /// Use subDet to identify whether the Hit is in BTL or ETL
  inline GeomDetEnumerators::Location subDet(uint row, uint col) const;
  inline GeomDetEnumerators::Location subDet(const FTLCluster::FTLHitPos&) const;
  inline float energy(uint row, uint col) const;
  inline float energy(const FTLCluster::FTLHitPos&) const;
  inline float time(uint row, uint col) const;
  inline float time(const FTLCluster::FTLHitPos&) const;
  inline float time_error(uint row, uint col) const;
  inline float time_error(const FTLCluster::FTLHitPos&) const;

  inline LocalError local_error(uint row, uint col) const;
  inline LocalError local_error(const FTLCluster::FTLHitPos&) const;
  inline GlobalPoint global_point(uint row, uint col) const;
  inline GlobalPoint global_point(const FTLCluster::FTLHitPos&) const;

  inline uint rows() const { return nrows; }
  inline uint columns() const { return ncols; }

  inline bool inside(uint row, uint col) const;

  inline void clear(uint row, uint col) {
    LocalError le_n(0, 0, 0);
    GlobalPoint gp_n(0, 0, 0);
    set_subDet(row, col, GeomDetEnumerators::invalidLoc);
    set_energy(row, col, 0.);
    set_time(row, col, 0.);
    set_time_error(row, col, 0.);
    set_local_error(row, col, le_n);
    set_global_point(row, col, gp_n);
  }
  inline void clear(const FTLCluster::FTLHitPos& pos) { clear(pos.row(), pos.col()); }

  inline void set(uint row,
                  uint col,
                  GeomDetEnumerators::Location subDet,
                  float energy,
                  float time,
                  float time_error,
                  const LocalError& local_error,
                  const GlobalPoint& global_point);
  inline void set(const FTLCluster::FTLHitPos&,
                  GeomDetEnumerators::Location subDet,
                  float energy,
                  float time,
                  float time_error,
                  const LocalError& local_error,
                  const GlobalPoint& global_point);

  inline void set_subDet(uint row, uint col, GeomDetEnumerators::Location subDet);
  inline void set_subDet(const FTLCluster::FTLHitPos&, GeomDetEnumerators::Location subDet);

  inline void set_energy(uint row, uint col, float energy);
  inline void set_energy(const FTLCluster::FTLHitPos&, float energy);
  inline void add_energy(uint row, uint col, float energy);

  inline void set_time(uint row, uint col, float time);
  inline void set_time(const FTLCluster::FTLHitPos&, float time);

  inline void set_time_error(uint row, uint col, float time_error);
  inline void set_time_error(const FTLCluster::FTLHitPos&, float time_error);

  inline void set_global_point(uint row, uint col, const GlobalPoint& gp);
  inline void set_global_point(const FTLCluster::FTLHitPos&, const GlobalPoint& gp);

  inline void set_local_error(uint row, uint col, const LocalError& le);
  inline void set_local_error(const FTLCluster::FTLHitPos&, const LocalError& le);

  uint size() const { return hitEnergy_vec.size(); }

  /// Definition of indexing within the buffer.
  uint index(uint row, uint col) const { return col * nrows + row; }
  uint index(const FTLCluster::FTLHitPos& pix) const { return index(pix.row(), pix.col()); }

private:
  std::vector<GeomDetEnumerators::Location> hitSubDet_vec;
  std::vector<float> hitEnergy_vec;
  std::vector<float> hitTime_vec;
  std::vector<float> hitTimeError_vec;
  std::vector<GlobalPoint> hitGP_vec;
  std::vector<LocalError> hitLE_vec;
  uint nrows;
  uint ncols;
};

MTDArrayBuffer::MTDArrayBuffer(uint rows, uint cols)
    : hitSubDet_vec(rows * cols, GeomDetEnumerators::invalidLoc),
      hitEnergy_vec(rows * cols, 0),
      hitTime_vec(rows * cols, 0),
      hitTimeError_vec(rows * cols, 0),
      hitGP_vec(rows * cols),
      hitLE_vec(rows * cols),
      nrows(rows),
      ncols(cols) {}

void MTDArrayBuffer::setSize(uint rows, uint cols) {
  hitSubDet_vec.resize(rows * cols, GeomDetEnumerators::invalidLoc);
  hitEnergy_vec.resize(rows * cols, 0);
  hitTime_vec.resize(rows * cols, 0);
  hitTimeError_vec.resize(rows * cols, 0);
  hitGP_vec.resize(rows * cols);
  hitLE_vec.resize(rows * cols);
  nrows = rows;
  ncols = cols;
}

bool MTDArrayBuffer::inside(uint row, uint col) const { return (row < nrows && col < ncols); }

GeomDetEnumerators::Location MTDArrayBuffer::subDet(uint row, uint col) const { return hitSubDet_vec[index(row, col)]; }
GeomDetEnumerators::Location MTDArrayBuffer::subDet(const FTLCluster::FTLHitPos& pix) const {
  return hitSubDet_vec[index(pix)];
}

float MTDArrayBuffer::energy(uint row, uint col) const { return hitEnergy_vec[index(row, col)]; }
float MTDArrayBuffer::energy(const FTLCluster::FTLHitPos& pix) const { return hitEnergy_vec[index(pix)]; }

float MTDArrayBuffer::time(uint row, uint col) const { return hitTime_vec[index(row, col)]; }
float MTDArrayBuffer::time(const FTLCluster::FTLHitPos& pix) const { return hitTime_vec[index(pix)]; }

float MTDArrayBuffer::time_error(uint row, uint col) const { return hitTimeError_vec[index(row, col)]; }
float MTDArrayBuffer::time_error(const FTLCluster::FTLHitPos& pix) const { return hitTimeError_vec[index(pix)]; }

LocalError MTDArrayBuffer::local_error(uint row, uint col) const { return hitLE_vec[index(row, col)]; }
LocalError MTDArrayBuffer::local_error(const FTLCluster::FTLHitPos& pix) const { return hitLE_vec[index(pix)]; }

GlobalPoint MTDArrayBuffer::global_point(uint row, uint col) const { return hitGP_vec[index(row, col)]; }
GlobalPoint MTDArrayBuffer::global_point(const FTLCluster::FTLHitPos& pix) const { return hitGP_vec[index(pix)]; }

void MTDArrayBuffer::set(uint row,
                         uint col,
                         GeomDetEnumerators::Location subDet,
                         float energy,
                         float time,
                         float time_error,
                         const LocalError& local_error,
                         const GlobalPoint& global_point) {
  hitSubDet_vec[index(row, col)] = subDet;
  hitEnergy_vec[index(row, col)] = energy;
  hitTime_vec[index(row, col)] = time;
  hitTimeError_vec[index(row, col)] = time_error;
  hitGP_vec[index(row, col)] = global_point;
  hitLE_vec[index(row, col)] = local_error;
}
void MTDArrayBuffer::set(const FTLCluster::FTLHitPos& pix,
                         GeomDetEnumerators::Location subDet,
                         float energy,
                         float time,
                         float time_error,
                         const LocalError& local_error,
                         const GlobalPoint& global_point) {
  set(pix.row(), pix.col(), subDet, energy, time, time_error, local_error, global_point);
}

void MTDArrayBuffer::set_subDet(uint row, uint col, GeomDetEnumerators::Location subDet) {
  hitSubDet_vec[index(row, col)] = subDet;
}
void MTDArrayBuffer::set_subDet(const FTLCluster::FTLHitPos& pix, GeomDetEnumerators::Location subDet) {
  hitSubDet_vec[index(pix)] = subDet;
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

void MTDArrayBuffer::set_global_point(uint row, uint col, const GlobalPoint& gp) { hitGP_vec[index(row, col)] = gp; }
void MTDArrayBuffer::set_global_point(const FTLCluster::FTLHitPos& pix, const GlobalPoint& gp) {
  hitGP_vec[index(pix)] = gp;
}

void MTDArrayBuffer::set_local_error(uint row, uint col, const LocalError& le) { hitLE_vec[index(row, col)] = le; }
void MTDArrayBuffer::set_local_error(const FTLCluster::FTLHitPos& pix, const LocalError& le) {
  hitLE_vec[index(pix)] = le;
}

#endif
