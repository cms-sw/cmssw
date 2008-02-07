// Read in strip digi collection and apply calibrations to ADC counts


#include <RecoLocalMuon/CSCRecHitD/interface/CSCStripGainAvg.h>

#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <CondFormats/CSCObjects/interface/CSCDBGains.h>
#include <CondFormats/DataRecord/interface/CSCDBGainsRcd.h>

#include <map>
#include <vector>
#include <iostream>

CSCStripGainAvg::CSCStripGainAvg( const edm::ParameterSet & ps ) {

}

CSCStripGainAvg::~CSCStripGainAvg() {

}


/* getStripGainAvg
 *
 */
float CSCStripGainAvg::getStripGainAvg() {
   
  long n_strip    = 0;
  float gain_avg = 1.;
  float gain_tot = 0.;
       
  // Build iterator which loops on all gain entries:
  std::vector<CSCDBGains::Item>::const_iterator it;
  
  for ( it=Gains->gains.begin(); it!=Gains->gains.end(); ++it ) {

    float the_gain = it->gain_slope;  

    if (the_gain < 10.0 && the_gain > 5.0 ) {
      gain_tot += the_gain;
      n_strip++;
    } 
  }

  // Average gain
  if ( n_strip > 0 ) { 
    gain_avg = gain_tot / n_strip;
  }

  // Avg Gain has been around ~7.5 so far in MTCC:  so can do consistency test
  if ( gain_avg < 6.0 || gain_avg > 9.0 ) {
    LogTrace("CSC") << "[CSCMakeStripDigiCollections] Check global CSC strip gain: "
		    << gain_avg << "  should be ~7.5 ";
    gain_avg = 7.5;
  }


  return gain_avg;
}

