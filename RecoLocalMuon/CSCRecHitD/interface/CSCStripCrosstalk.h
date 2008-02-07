#ifndef CSCRecHitD_CSCStripCrosstalk_h
#define CSCRecHitD_CSCStripCrosstalk_h

/** \class CSCStripCrosstalk
 *
 * This routine finds for a given DetId the X-talk.
 *
 * \author Dominique Fortin - UCR
 */

#include <CondFormats/CSCObjects/interface/CSCDBCrosstalk.h>
#include <CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h>
#include <DataFormats/MuonDetId/interface/CSCIndexer.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <vector>
#include <string>

class CSCIndexer;

class CSCStripCrosstalk
{
 public:

  typedef uint16_t IndexType;
  typedef uint32_t LongIndexType;

  /// configurable parameters
  explicit CSCStripCrosstalk( const edm::ParameterSet & ps );  
  ~CSCStripCrosstalk();

  // Member functions

  /// Load in the gains, X-talk and noise matrix and store in memory
  void setCrossTalk( const CSCDBCrosstalk* xtalk ) { 
    xTalk_ = const_cast<CSCDBCrosstalk*> (xtalk); 
  }
 
  /// Get the Xtalks out of the database for each of the strips within layer.
  void getCrossTalk( const CSCDetId& id, int centralStrip, std::vector<float>& xtalks);

 private:

  bool debug;
 // bool isData;

  // Store in memory xtalks
  CSCDBCrosstalk* xTalk_;

  CSCIndexer* theIndexer;
};

#endif

