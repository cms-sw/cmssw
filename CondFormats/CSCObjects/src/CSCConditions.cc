#include "CondFormats/CSCObjects/interface/CSCConditions.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"



CSCConditions::CSCConditions() 
 :theNoiseMatrix(0),
  theGains(0),
  thePedestals(0),
  theCrosstalk(0),
  theAverageGain( -1.0 )
{
}


CSCConditions::~CSCConditions()
{
}


void CSCConditions::initializeEvent(const edm::EventSetup & es)
{
  // Strip gains
  edm::ESHandle<CSCDBGains> hGains;
  es.get<CSCDBGainsRcd>().get( hGains );
  theGains = hGains.product();
  // Strip X-talk
  edm::ESHandle<CSCDBCrosstalk> hCrosstalk;
  es.get<CSCDBCrosstalkRcd>().get( hCrosstalk );
  theCrosstalk = hCrosstalk.product();
  // Strip pedestals
  edm::ESHandle<CSCDBPedestals> hPedestals;
  es.get<CSCDBPedestalsRcd>().get( hPedestals );
  thePedestals = hPedestals.product();

  // Strip autocorrelation noise matrix
  edm::ESHandle<CSCDBNoiseMatrix> hNoiseMatrix;
  es.get<CSCDBNoiseMatrixRcd>().get(hNoiseMatrix);
  theNoiseMatrix = hNoiseMatrix.product();

  // Has GainsRcd changed?
  if( gainsWatcher_.check( es ) ) { // Yes...
    theAverageGain = -1.0; // ...reset, so next access will recalculate it
  }
  
//  print();
}


void CSCConditions::print() const
{
/*
  std::cout << "SIZES: GAINS: " << theGains->gains.size()
            << "   PEDESTALS: " << thePedestals->pedestals.size()
            << "   NOISES "  << theNoiseMatrix->matrix.size() << std::endl;;

  std::map< int,std::vector<CSCDBGains::Item> >::const_iterator layerGainsItr = theGains->gains.begin(), 
      lastGain = theGains->gains.end();
  for( ; layerGainsItr != lastGain; ++layerGainsItr)
  {
    std::cout << "GAIN " << layerGainsItr->first 
              << " STRIPS " << layerGainsItr->second.size() << " "
              << layerGainsItr->second[0].gain_slope 
              << " " << layerGainsItr->second[0].gain_intercept << std::endl;
  }

  std::map< int,std::vector<CSCDBPedestals::Item> >::const_iterator pedestalItr = thePedestals->pedestals.begin(), 
                                                                  lastPedestal = thePedestals->pedestals.end();
  for( ; pedestalItr != lastPedestal; ++pedestalItr)
  {
    std::cout << "PEDS " << pedestalItr->first << " " 
              << " STRIPS " << pedestalItr->second.size() << " ";
    for(int i = 1; i < 80; ++i)
    {
       std::cout << pedestalItr->second[i-1].rms << " " ;
     }
     std::cout << std::endl;
  }

  std::map< int,std::vector<CSCDBCrosstalk::Item> >::const_iterator crosstalkItr = theCrosstalk->crosstalk.begin(),
                                                                  lastCrosstalk = theCrosstalk->crosstalk.end();
  for( ; crosstalkItr != lastCrosstalk; ++crosstalkItr)
  {
    std::cout << "XTALKS " << crosstalkItr->first 
      << " STRIPS " << crosstalkItr->second.size() << " "  
     << crosstalkItr->second[5].xtalk_slope_left << " " 
     << crosstalkItr->second[5].xtalk_slope_right << " " 
     << crosstalkItr->second[5].xtalk_intercept_left << " " 
     << crosstalkItr->second[5].xtalk_intercept_right << std::endl;
  }
*/
}

//@@ WARNING As of 21.02.2008 CODE SCALES USING THE DEFAULT FACTORS
//@@ RATHER THAN EXPLICIT VALUES FROM DB. TO BE CHANGED.

float CSCConditions::gain(const CSCDetId & detId, int channel) const
{
  assert(theGains != 0);
  return theGains->item(detId, channel).gain_slope/CSCDBGains::FGAIN;
}


float CSCConditions::pedestal(const CSCDetId & detId, int channel) const
{
  assert(thePedestals != 0);
  return thePedestals->item(detId, channel).ped/CSCDBPedestals::FPED;
}


float CSCConditions::pedestalSigma(const CSCDetId&detId, int channel) const
{
  assert(thePedestals != 0);
  return thePedestals->item(detId, channel).rms/CSCDBPedestals::FRMS;
}


float CSCConditions::crosstalkIntercept(const CSCDetId&detId, int channel, bool leftRight) const
{
  assert(theCrosstalk != 0);
  const CSCDBCrosstalk::Item & item = theCrosstalk->item(detId, channel);

  // resistive fraction is at the peak, where t=0
  return leftRight ? item.xtalk_intercept_right/CSCDBCrosstalk::FINTERCEPT 
                   : item.xtalk_intercept_left/CSCDBCrosstalk::FINTERCEPT ;
}


float CSCConditions::crosstalkSlope(const CSCDetId&detId, int channel, bool leftRight) const
{
  assert(theCrosstalk != 0);
  const CSCDBCrosstalk::Item & item = theCrosstalk->item(detId, channel);

  // resistive fraction is at the peak, where t=0
  return leftRight ? item.xtalk_slope_right/CSCDBCrosstalk::FSLOPE
                   : item.xtalk_slope_left/CSCDBCrosstalk::FSLOPE ;
}

const CSCDBNoiseMatrix::Item & CSCConditions::noiseMatrix(const CSCDetId& detId, int channel) const
{
  assert(theNoiseMatrix != 0);
  return theNoiseMatrix->item(detId, channel);
}

void CSCConditions::noiseMatrixElements( const CSCDetId& id, int channel, std::vector<float>& me ) const {
  assert(me.size()>11);
  const CSCDBNoiseMatrix::Item& item = noiseMatrix(id, channel);
  me[0] = item.elem33/CSCDBNoiseMatrix::FNOISE;
  me[1] = item.elem34/CSCDBNoiseMatrix::FNOISE;
  me[2] = item.elem35/CSCDBNoiseMatrix::FNOISE;
  me[3] = item.elem44/CSCDBNoiseMatrix::FNOISE;
  me[4] = item.elem45/CSCDBNoiseMatrix::FNOISE;
  me[5] = item.elem46/CSCDBNoiseMatrix::FNOISE;
  me[6] = item.elem55/CSCDBNoiseMatrix::FNOISE;
  me[7] = item.elem56/CSCDBNoiseMatrix::FNOISE;
  me[8] = item.elem57/CSCDBNoiseMatrix::FNOISE;
  me[9] = item.elem66/CSCDBNoiseMatrix::FNOISE;
  me[10] = item.elem67/CSCDBNoiseMatrix::FNOISE;
  me[11] = item.elem77/CSCDBNoiseMatrix::FNOISE;
}

void CSCConditions::crossTalk( const CSCDetId& id, int channel, std::vector<float>& ct ) const {
  assert(theCrosstalk != 0);
  const CSCDBCrosstalk::Item & item = theCrosstalk->item(id, channel);
  ct[0] = item.xtalk_slope_left/CSCDBCrosstalk::FSLOPE;
  ct[1] = item.xtalk_intercept_left/CSCDBCrosstalk::FINTERCEPT;
  ct[2] = item.xtalk_slope_right/CSCDBCrosstalk::FSLOPE;
  ct[3] = item.xtalk_intercept_right/CSCDBCrosstalk::FINTERCEPT;
}

/// Return average strip gain for full CSC system. Lazy evaluation.
/// Restrict averaging to gains between 5 and 10, and require average
/// is between 6 or 9 otherwise fix it to 7.5.
float CSCConditions::averageGain() const {

  const float loEdge = 5.0; // consider gains above this
  const float hiEdge = 10.0; // consider gains below this
  const float loLimit = 6.0; // lowest acceptable average gain
  const float hiLimit = 9.0; // highest acceptable average gain
  const float expectedAverage = 7.5; // default average gain

  if ( theAverageGain > 0. ) return theAverageGain; // only recalculate if necessary

  int  n_strip   = 0;
  float gain_tot = 0.;

  CSCDBGains::GainContainer::const_iterator it;
  for ( it=theGains->gains.begin(); it!=theGains->gains.end(); ++it ) {
    float the_gain = it->gain_slope/CSCDBGains::FGAIN;
    if (the_gain > loEdge && the_gain < hiEdge ) {
      gain_tot += the_gain;
      ++n_strip;
    }
  }

  // Average gain
  if ( n_strip > 0 ) {
    theAverageGain = gain_tot / n_strip;
  }

  // Average gain has been around 7.5 in real data
  if ( theAverageGain < loLimit || theAverageGain > hiLimit ) {
    //    LogTrace("CSC") << "Average CSC strip gain = "
    //                    << theAverageGain << "  is reset to expected value " << expectedAverage;
    theAverageGain = expectedAverage;
  }

  return theAverageGain;
}
