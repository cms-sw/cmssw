/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_CTPPSReco_TotemRPCluster
#define DataFormats_CTPPSReco_TotemRPCluster

#include <cstdint>

/**
 *\brief Cluster of TOTEM RP strip hits.
 **/
class TotemRPCluster {
public:
  TotemRPCluster(unsigned short str_beg = 0, unsigned short str_end = 0) : str_beg_(str_beg), str_end_(str_end) {}

  inline uint16_t stripBegin() const { return str_beg_; }
  inline void setStripBegin(unsigned short str_beg) { str_beg_ = str_beg; }

  inline uint16_t stripEnd() const { return str_end_; }
  inline void setStripEnd(unsigned short str_end) { str_end_ = str_end; }

  inline int numberOfStrips() const { return str_end_ - str_beg_ + 1; }

  inline double centerStripPosition() const { return (str_beg_ + str_end_) / 2.; }

private:
  uint16_t str_beg_;
  uint16_t str_end_;
};

//----------------------------------------------------------------------------------------------------

inline bool operator<(const TotemRPCluster& l, const TotemRPCluster& r) {
  if (l.stripBegin() < r.stripBegin())
    return true;
  if (l.stripBegin() > r.stripBegin())
    return false;

  if (l.stripEnd() < r.stripEnd())
    return true;

  return false;
}

#endif
