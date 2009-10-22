#ifndef SiStripLatency_h
#define SiStripLatency_h

#include <vector>
#include <algorithm>

using namespace std;

/**
 * Holds the latency and the mode of the run. <br>
 * The latency is stored per apv and the information is compressed by reducing
 * sequences of apvs with the same latency to a single value plus information on the
 * start and end of the sequence. <br>
 * The mode is a single value, stored as a char defined as: <br>
 * - mode == 37 : deconvolution mode <br>
 * - mode == 47 : peak mode <br>
 * See here http://www.id.clrc.ac.uk/INS/Electronic_Systems/Microelectronics_Design/Projects/High_Energy_Physics/CMS/APV25-S1/files/User_Guide_2.0.pdf
 * page 13 section 6.5 table 7 for the definition of possible modes. <br>
 * The put method requires the latency and mode values for a given apv and detId. <br>
 * <br>
 * The internal Latency object stores the detId and apv value in a compressed
 * (bit shifted) uint32_t holding both the values. It stores the latency value
 * in a uint8_t (unsigned char). The APV user guide reports a maximum value of 191
 * for the latency, so the uint8_t (0-255) is enough. If the value is not filled or
 * the detId is not found it will return latency = 255. <br>
 * The mode value is also stored in an unsigned char (possible values 0-255). The invalid
 * case returns mode = 0 in this case. <br>
 * To save space, since typically the latency and mode is the same for all apvs, the ranges
 * of consecutive detIds and apvs are collapsed in the last value, so that the lower_bound
 * binary search will return the correct latency and mode. <br>
 * <br>
 * Methods are provided to extract latency and mode (separately or together in a pair)
 * for each apv. <br>
 * If the value of latency (mode) is the same for the whole Tracker, the singleLatency()
 * (singleMode()) method will return it, otherwise it will return 255 (0). <br>
 * <br>
 * The method allLatencyAndModes() returns the internal vector<Latency> (by value).
 * <br>
 * ATTENTION: the biggest possible detId value that can be stored is 536870911 because
 * 29 bits out of 32 are reserved for the detId and the remaining 3 for the apv value (1-6). <br>
 * The biggest possible Tracker detId at this moment is 470178036 so this is not a problem.
 * In any case the code will throw an "InsertFailure" cms::Exception if trying to insert a detId
 * above the maximum and output a detailed LogError message about the problem.
 */

class SiStripLatency
{
 public:

  SiStripLatency() {}

//   SiStripLatency() :
//     mode_(-1)
//   {}
//   SiStripLatency(const int16_t mode) :
//     mode_(mode)
//   {}

  // Defined as public for genreflex
  struct Latency
  {
    Latency(const uint32_t inputDetIdAndApv, const uint16_t inputLatency, const uint16_t inputMode) :
      detIdAndApv(inputDetIdAndApv),
      latency(inputLatency),
      mode(inputMode)
    {}
    /// Default constructor needed by genreflex
    Latency() :
      detIdAndApv(0),
      latency(255),
      mode(0)
    {}
    uint32_t detIdAndApv;
    unsigned char latency;
    unsigned char mode;
  };
  typedef vector<Latency>::iterator latIt;
  typedef vector<Latency>::const_iterator latConstIt;

  /** Saves the detIdAndApv and latency values in the vector of Latency objects.
   * At the end of the filling phase, the compress method should be called to
   * collapse all ranges in single values. Note that everything would work even
   * if the compress method is not called, only the space used would be more than
   * needed.
   */
  bool put( const uint32_t detId, const uint16_t apv, const uint16_t latency, const uint16_t mode );
  uint16_t latency(const uint32_t detId, const uint16_t apv) const;
  uint16_t mode(const uint32_t detId, const uint16_t apv) const;
  pair<uint16_t, uint16_t> latencyAndMode(const uint32_t detId, const uint16_t apv) const;
  inline vector<Latency> allLatencyAndModes() const { return latencies_; }

  /// Fills the passed vector with all the possible latencies in the Tracker
  void allLatencies(vector<uint16_t> & allLatenciesVector) const;
  /// Fills the passed vector with all the possible modes in the Tracker
  void allModes(vector<uint16_t> & allModesVector) const;

  /** Reduce ranges of consecutive detIdsAndApvs with the same latency and mode to
   * one value (the latest) so that lower_bound will return the correct value for
   * latency and mode.
   */
  void compress();
  /// If all the latency values stored are equal return that value, otherwise return -1
  uint16_t singleLatency() const;
  uint16_t singleMode() const;
  //   pair<uint16_t, uint16_t> singleLatencyAndMode() const;

  /// Prints the number of ranges as well as the value of singleLatency and singleMode
  void printSummary(std::stringstream & ss) const;
  /// Prints the full list of all ranges and corresponding values of latency and mode
  void printDebug(std::stringstream & ss) const;

 private:

  struct OrderByDetIdAndApv
  {
    bool operator()(const Latency & lat1, const uint32_t detIdAndApv) const {
      return lat1.detIdAndApv < detIdAndApv;
    }
  };

  // struct EqualByLatency
  // {
  //   bool operator()(const Latency & lat1, const Latency & lat2) {
  //     return( lat1.latency == lat2.latency );
  //   }
  // };

  /// Used to compute the position with the lower_bound binary search
  // If put in the cc file it will not know about the typedefs and the Latency class
  const latConstIt position(const uint32_t detId, const uint16_t apv) const
  {
    if( latencies_.empty() ) {
      // cout << "SiStripLatency: Error, range is empty" << endl;
      return latencies_.end();
    }
    uint32_t detIdAndApv = (detId << 3) | apv;
    latConstIt pos = lower_bound(latencies_.begin(), latencies_.end(), detIdAndApv, OrderByDetIdAndApv());
    return pos;
  }
  vector<Latency> latencies_;
};

#endif
