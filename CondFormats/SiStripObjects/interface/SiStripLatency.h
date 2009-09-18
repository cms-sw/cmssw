#ifndef SiStripLatency_h
#define SiStripLatency_h

#include <vector>

using namespace std;

/**
 * Holds the latency and the mode of the run. <br>
 * The latency is stored per apv and the information is compressed by reducing
 * sequences of apvs with the same latency to a single value plus information on the
 * start and end of the sequence. <br>
 * The mode is a single value, stored as a uint16_t defined as: <br>
 * - mode == 0 : deconvolution mode <br>
 * - mode == 1 : peak mode <br>
 * The put method requires the latency values for a given apv and detId. <br>
 * <br>
 * The internal Latency object stores the detId and apv value in a compressed
 * (bit shifted) uint32_t holding both the values. It stores the latency value
 * in a float. <br>
 * the Apv where the latency assumes a value different than the one before. <br>
 * To save space, since typically the latency is the same for all modules, the ranges
 * of consecutive detIds and apvs are collapsed in the last value, so that the lower_bound
 * binary search will return the correct latency.
 */

class SiStripLatency
{
 public:

  SiStripLatency() :
    mode_(-1)
  {}
  SiStripLatency(const int16_t mode) :
    mode_(mode)
  {}

  /** Saves the detIdAndApv and latency values in the vector of Latency objects.
   * At the end of the filling phase, the compress method should be called to
   * collapse all ranges in single values. Note that everything would work even
   * if the compress method is not called, only the space used would be more than
   * needed.
   */
  bool put( const uint32_t detId, const uint16_t apv, const float & latency );
  float get(const uint32_t detId, const uint16_t apv);
  /** Reduce ranges of consecutive detIdsAndApvs with the same latency to one value (the latest)
   * so that lower_bound will return the correct value for latency.
   */
  void compress();
  /// If all the latency values stored are equal return that value, otherwise return -1
  float getSingleLatency();

  inline void setMode( const int16_t mode ) {
    mode_ = mode;
  }
  inline bool mode() {
    return mode_;
  }
 private:
  struct Latency
  {
    Latency(const uint32_t inputDetIdAndApv, const float & inputLatency) :
      latency(inputLatency),
      detIdAndApv(inputDetIdAndApv)
    {}
    float latency;
    uint32_t detIdAndApv;
  };
  typedef vector<Latency>::iterator latIt;
  typedef vector<Latency>::const_iterator latConstIt;
  struct OrderByDetIdAndApv
  {
    bool operator()(const Latency & lat1, const uint32_t detIdAndApv) const {
      return lat1.detIdAndApv < detIdAndApv;
    }
  };
  struct EqualByLatency
  {
    bool operator()(const Latency & lat1, const Latency & lat2) {
      return( lat1.latency == lat2.latency );
    }
  };
  vector<Latency> latencies_;
  int16_t mode_;
};

#endif
