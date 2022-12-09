#ifndef DataFormats_FTLRecHit_FTLCluster_h
#define DataFormats_FTLRecHit_FTLCluster_h

/** \class FTLCluster
 *  
 * based on SiPixelCluster
 *
 * \author Paolo Meridiani
 */

#include <cmath>
#include <vector>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <numeric>

#include "DataFormats/DetId/interface/DetId.h"

class FTLCluster {
public:
  typedef DetId key_type;

  class FTLHit {
  public:
    constexpr FTLHit() : x_(0), y_(0), energy_(0), time_(0), time_error_(0) {}
    constexpr FTLHit(uint16_t hit_x, uint16_t hit_y, float hit_energy, float hit_time, float hit_time_error)
        : x_(hit_x), y_(hit_y), energy_(hit_energy), time_(hit_time), time_error_(hit_time_error) {}
    constexpr uint16_t x() { return x_; }
    constexpr uint16_t y() { return y_; }
    constexpr uint16_t energy() { return energy_; }
    constexpr uint16_t time() { return time_; }
    constexpr uint16_t time_error() { return time_error_; }

  private:
    uint16_t x_;  //row
    uint16_t y_;  //col
    float energy_;
    float time_;
    float time_error_;
  };

  //--- Integer shift in x and y directions.
  class Shift {
  public:
    constexpr Shift(int dx, int dy) : dx_(dx), dy_(dy) {}
    constexpr Shift() : dx_(0), dy_(0) {}
    constexpr int dx() const { return dx_; }
    constexpr int dy() const { return dy_; }

  private:
    int dx_;
    int dy_;
  };

  //--- Position of a FTL Hit
  class FTLHitPos {
  public:
    constexpr FTLHitPos() : row_(0), col_(0) {}
    constexpr FTLHitPos(int row, int col) : row_(row), col_(col) {}
    constexpr int row() const { return row_; }
    constexpr int col() const { return col_; }
    constexpr FTLHitPos operator+(const Shift& shift) const {
      return FTLHitPos(row() + shift.dx(), col() + shift.dy());
    }

  private:
    int row_;
    int col_;
  };

  static constexpr unsigned int MAXSPAN = 255;
  static constexpr unsigned int MAXPOS = 2047;

  /** Construct from a range of digis that form a cluster and from 
   *  a DetID. The range is assumed to be non-empty.
   */
  FTLCluster() {}

  FTLCluster(DetId id,
             unsigned int isize,
             float const* energys,
             float const* times,
             float const* time_errors,
             uint16_t const* xpos,
             uint16_t const* ypos,
             uint16_t const xmin,
             uint16_t const ymin)
      : theid(id),
        theHitOffset(2 * isize),
        theHitENERGY(energys, energys + isize),
        theHitTIME(times, times + isize),
        theHitTIME_ERROR(time_errors, time_errors + isize) {
    uint16_t maxCol = 0;
    uint16_t maxRow = 0;
    int maxHit = -1;
    float maxEnergy = -99999;
    for (unsigned int i = 0; i != isize; ++i) {
      uint16_t xoffset = xpos[i] - xmin;
      uint16_t yoffset = ypos[i] - ymin;
      theHitOffset[i * 2] = std::min(uint16_t(MAXSPAN), xoffset);
      theHitOffset[i * 2 + 1] = std::min(uint16_t(MAXSPAN), yoffset);
      if (xoffset > maxRow)
        maxRow = xoffset;
      if (yoffset > maxCol)
        maxCol = yoffset;
      if (theHitENERGY[i] > maxEnergy) {
        maxHit = i;
        maxEnergy = theHitENERGY[i];
      }
    }
    packRow(xmin, maxRow);
    packCol(ymin, maxCol);

    if (maxHit >= 0)
      seed_ = std::min(uint8_t(MAXSPAN), uint8_t(maxHit));
  }

  // linear average position (barycenter)
  inline float x() const {
    auto x_pos = [this](unsigned int i) { return this->theHitOffset[i * 2] + minHitRow(); };
    return weighted_mean(this->theHitENERGY, x_pos);
  }

  inline float y() const {
    auto y_pos = [this](unsigned int i) { return this->theHitOffset[i * 2 + 1] + minHitCol(); };
    return weighted_mean(this->theHitENERGY, y_pos);
  }

  inline float positionError(const float sigmaPos) const {
    float sumW2(0.f), sumW(0.f);
    for (const auto& hitW : theHitENERGY) {
      sumW2 += hitW * hitW;
      sumW += hitW;
    }
    if (sumW > 0)
      return sigmaPos * std::sqrt(sumW2) / sumW;
    else
      return -999.f;
  }

  inline float time() const {
    auto t = [this](unsigned int i) { return this->theHitTIME[i]; };
    return weighted_mean(this->theHitENERGY, t);
  }

  inline float timeError() const {
    auto t_err = [this](unsigned int i) { return this->theHitTIME_ERROR[i]; };
    return weighted_mean_error(this->theHitENERGY, t_err);
  }

  // Return number of hits.
  inline int size() const { return theHitENERGY.size(); }

  // Return cluster dimension in the x direction.
  inline int sizeX() const { return rowSpan() + 1; }

  // Return cluster dimension in the y direction.
  inline int sizeY() const { return colSpan() + 1; }

  inline float energy() const {
    return std::accumulate(theHitENERGY.begin(), theHitENERGY.end(), 0.f);
  }  // Return total cluster energy.

  inline int minHitRow() const { return theMinHitRow; }             // The min x index.
  inline int maxHitRow() const { return minHitRow() + rowSpan(); }  // The max x index.
  inline int minHitCol() const { return theMinHitCol; }             // The min y index.
  inline int maxHitCol() const { return minHitCol() + colSpan(); }  // The max y index.

  const std::vector<uint8_t>& hitOffset() const { return theHitOffset; }
  const std::vector<float>& hitENERGY() const { return theHitENERGY; }
  const std::vector<float>& hitTIME() const { return theHitTIME; }
  const std::vector<float>& hitTIME_ERROR() const { return theHitTIME_ERROR; }

  // infinite faster than above...
  FTLHit hit(int i) const {
    return FTLHit(minHitRow() + theHitOffset[i * 2],
                  minHitCol() + theHitOffset[i * 2 + 1],
                  theHitENERGY[i],
                  theHitTIME[i],
                  theHitTIME_ERROR[i]);
  }

  FTLHit seed() const { return hit(seed_); }

  int colSpan() const { return theHitColSpan; }

  int rowSpan() const { return theHitRowSpan; }

  const DetId& id() const { return theid; }
  const DetId& detid() const { return id(); }

  bool overflowCol() const { return overflow_(theHitColSpan); }

  bool overflowRow() const { return overflow_(theHitRowSpan); }

  bool overflow() const { return overflowCol() || overflowRow(); }

  void packCol(uint16_t ymin, uint16_t yspan) {
    theMinHitCol = ymin;
    theHitColSpan = std::min(yspan, uint16_t(MAXSPAN));
  }
  void packRow(uint16_t xmin, uint16_t xspan) {
    theMinHitRow = xmin;
    theHitRowSpan = std::min(xspan, uint16_t(MAXSPAN));
  }

  void setClusterPosX(float posx) { pos_x = posx; }
  void setClusterErrorX(float errx) { err_x = errx; }
  void setClusterErrorTime(float errtime) { err_time = errtime; }
  float getClusterPosX() const { return pos_x; }
  float getClusterErrorX() const { return err_x; }
  float getClusterErrorTime() const { return err_time; }

private:
  DetId theid;

  std::vector<uint8_t> theHitOffset;
  std::vector<float> theHitENERGY;
  std::vector<float> theHitTIME;
  std::vector<float> theHitTIME_ERROR;

  uint16_t theMinHitRow = MAXPOS;  // Minimum hit index in the x direction (low edge).
  uint16_t theMinHitCol = MAXPOS;  // Minimum hit index in the y direction (left edge).
  uint8_t theHitRowSpan = 0;       // Span hit index in the x direction (low edge).
  uint8_t theHitColSpan = 0;       // Span hit index in the y direction (left edge).

  float pos_x = -99999.9f;  // For pixels with internal position information in one coordinate (i.e. BTL crystals)
  float err_x = -99999.9f;  // For pixels with internal position information in one coordinate (i.e. BTL crystals)
  float err_time = -99999.9f;

  uint8_t seed_;

  template <typename SumFunc, typename OutFunc>
  float weighted_sum(const std::vector<float>& weights, SumFunc&& sumFunc, OutFunc&& outFunc) const {
    float tot = 0;
    float sumW = 0;
    for (unsigned int i = 0; i < weights.size(); ++i) {
      tot += sumFunc(i);
      sumW += weights[i];
    }
    return outFunc(tot, sumW);
  }

  template <typename Value>
  float weighted_mean(const std::vector<float>& weights, Value&& value) const {
    auto sumFunc = [&weights, value](unsigned int i) { return weights[i] * value(i); };
    auto outFunc = [](float x, float y) {
      if (y > 0)
        return (float)x / y;
      else
        return -999.f;
    };
    return weighted_sum(weights, sumFunc, outFunc);
  }

  template <typename Err>
  float weighted_mean_error(const std::vector<float>& weights, Err&& err) const {
    auto sumFunc = [&weights, err](unsigned int i) { return weights[i] * weights[i] * err(i) * err(i); };
    auto outFunc = [](float x, float y) {
      if (y > 0)
        return (float)sqrt(x) / y;
      else
        return -999.f;
    };
    return weighted_sum(weights, sumFunc, outFunc);
  }

  static int overflow_(uint16_t span) { return span == uint16_t(MAXSPAN); }
};

// Comparison operators  (needed by DetSetVector & SortedCollection )
inline bool operator<(const FTLCluster& one, const FTLCluster& other) {
  if (one.detid() == other.detid()) {
    if (one.minHitRow() < other.minHitRow()) {
      return true;
    } else if (one.minHitRow() > other.minHitRow()) {
      return false;
    } else if (one.minHitCol() < other.minHitCol()) {
      return true;
    } else {
      return false;
    }
  }
  return one.detid() < other.detid();
}

inline bool operator<(const FTLCluster& one, const uint32_t& detid) { return one.detid() < detid; }

inline bool operator<(const uint32_t& detid, const FTLCluster& other) { return detid < other.detid(); }

#endif
