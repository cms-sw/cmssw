#ifndef CSCRecHitD_CSCStripGain_h
#define CSCRecHitD_CSCStripGain_h

/** \class CSCCalibrateStrip
 *
 * This routine finds for a given DetId
 * the average global gain, the correction weight for the gains, 
 * in a given layer.  This is done such that it 
 * minimizes the number of calls to this class.
 *
 * \author Dominique Fortin - UCR
 */

#include <CondFormats/CSCObjects/interface/CSCDBGains.h>
#include <CondFormats/DataRecord/interface/CSCDBGainsRcd.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

class CSCIndexer;


class CSCStripGain
{
 public:

  typedef uint16_t IndexType;
  typedef uint32_t LongIndexType;

  /// configurable parameters
  explicit CSCStripGain(const edm::ParameterSet & ps);  
  ~CSCStripGain();

  // Member functions

  /// Load in the gains, X-talk and noise matrix and store in memory
  void setCalibration( float GlobalGainAvg, const CSCDBGains* gains ) { 
    globalGainAvg = GlobalGainAvg;
    Gains_ = const_cast<CSCDBGains*> (gains); 
  }
 
  /// Get the gains out of the database for each of the strips within a cluster.
  void getStripGain( const CSCDetId& id, float* weights );

 private:

  bool debug;
  //bool isData;

  // Store in memory Gains
  float globalGainAvg;
  CSCDBGains* Gains_;

  CSCIndexer* theIndexer;

};

#endif

