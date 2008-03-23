// This is CSCPeakBinOfStripPulse

#include <RecoLocalMuon/CSCRecHitD/src/CSCPeakBinOfStripPulse.h>

#include <DataFormats/CSCDigi/interface/CSCStripDigi.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <vector>


CSCPeakBinOfStripPulse::CSCPeakBinOfStripPulse( const edm::ParameterSet& ps ) {


}


CSCPeakBinOfStripPulse::~CSCPeakBinOfStripPulse() {

}


/* peakAboveBaseline
 *
 * This finds the seed for the cluster, 
 *
 */
bool CSCPeakBinOfStripPulse::peakAboveBaseline( const CSCStripDigi& digi, float& hmax, int& tmax, float* height ) const {
  
  std::vector<int> sca = digi.getADCCounts();

  // Initialize parameters, just in case...  
  tmax = 0;
  hmax = 0;
  if ( sca.empty() ) return false;


  // First find maximum time bin
  for (int i = 0; i < 8; ++i ) {
     if (sca[i] > sca[tmax] ) tmax = i;
  }

  // Find pedestal
  float ped = baseline( digi );

  // Store ADC signal for time bins [2-7]
  int i = 0;  
  for ( int t = 2; t < 8; ++t ) {
    height[i] = sca[t] - ped;
    i++; 
  }

  // Maximum cannot occur in first 3 time bins or in last time bin.
  if ( tmax < 2 || tmax > 6) return false;  
  hmax = height[tmax-2];
  return true;
}



/* baseline
 *
 */
float CSCPeakBinOfStripPulse::baseline(const CSCStripDigi& digi) const {
    std::vector<int> sca = digi.getADCCounts();

  float ped = ( sca[0]+sca[1] ) / 2.;

  return ped;
}


