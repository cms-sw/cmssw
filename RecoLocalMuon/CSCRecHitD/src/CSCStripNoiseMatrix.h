#ifndef CSCRecHitD_CSCStripNoiseMatrix_h
#define CSCRecHitD_CSCStripNoiseMatrix_h

/** \class CSCStripNoiseMatrix
 *
 * This routine finds for a given DetId the autocorrelation noise in a given layer.
 * Note that the elements of the matrix have to be multiplied by the square of the
 * strip gain for that strip.
 *
 * \author Dominique Fortin - UCR
 */

#include <CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h>
#include <CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h>
#include <CondFormats/CSCObjects/interface/CSCDBGains.h>
#include <CondFormats/DataRecord/interface/CSCDBGainsRcd.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <vector>
#include <string>

class CSCIndexer;

class CSCStripNoiseMatrix
{
 public:

  typedef uint16_t IndexType;
  typedef uint32_t LongIndexType;

  /// configurable parameters
  explicit CSCStripNoiseMatrix(const edm::ParameterSet & ps);  
  ~CSCStripNoiseMatrix();

  // Member functions

  /// Load in the noise matrix and store in memory
  void setNoiseMatrix( float GlobalGainAvg, 
                       const CSCDBGains* gains, 
                       const CSCDBNoiseMatrix* noise ) { 

    globalGainAvg = GlobalGainAvg;
    Gains = const_cast<CSCDBGains*> (gains);
    Noise = const_cast<CSCDBNoiseMatrix*> (noise); 
  }
 
  /// Get the noise matrix out of the database for each of the strips within a cluster.
  void getNoiseMatrix( const CSCDetId& id, int centralStrip, std::vector<float>& nMatrix );

  /// Get the gains out of the database for each of the strips within a cluster.
  float getStripGain( LongIndexType thisStrip );

 private:

  bool debug;
  //bool isData;

  // Store in memory Gains and Noise matrix
  float globalGainAvg;
  CSCDBGains         * Gains;
  CSCDBNoiseMatrix   * Noise;

  CSCIndexer* theIndexer;

};

#endif

