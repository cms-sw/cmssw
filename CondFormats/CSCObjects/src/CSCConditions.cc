#include "CondFormats/CSCObjects/interface/CSCConditions.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"



CSCConditions::CSCConditions() 
 :theNoiseMatrix(0),
  theGains(0),
  thePedestals(0),
  theCrosstalk(0)
{
}


CSCConditions::~CSCConditions()
{
  //delete theNoiseMatrix;
  //delete theGains;
  //delete thePedestals;
  //delete theCrosstalk;
}


void CSCConditions::initializeEvent(const edm::EventSetup & es)
{
  // Strip gains
  edm::ESHandle<CSCDBGains> hGains;
  es.get<CSCDBGainsRcd>().get( hGains );
  theGains = &*hGains.product();
  // Strip X-talk
  edm::ESHandle<CSCDBCrosstalk> hCrosstalk;
  es.get<CSCDBCrosstalkRcd>().get( hCrosstalk );
  theCrosstalk = &*hCrosstalk.product();
  // Strip pedestals
  edm::ESHandle<CSCDBPedestals> hPedestals;
  es.get<CSCDBPedestalsRcd>().get( hPedestals );
  thePedestals = &*hPedestals.product();

  // Strip autocorrelation noise matrix
  edm::ESHandle<CSCDBNoiseMatrix> hNoiseMatrix;
  es.get<CSCDBNoiseMatrixRcd>().get(hNoiseMatrix);
  theNoiseMatrix = &*hNoiseMatrix.product();

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


float CSCConditions::gain(const CSCDetId & detId, int channel) const
{
  assert(theGains != 0);
  return theGains->item(detId, channel).gain_slope;
}


float CSCConditions::pedestal(const CSCDetId & detId, int channel) const
{
  assert(thePedestals != 0);
  return thePedestals->item(detId, channel).ped;
}


float CSCConditions::pedestalSigma(const CSCDetId&detId, int channel) const
{
  assert(thePedestals != 0);
  return thePedestals->item(detId, channel).rms;
}


float CSCConditions::crosstalkIntercept(const CSCDetId&detId, int channel, bool leftRight) const
{
  assert(theCrosstalk != 0);
  const CSCDBCrosstalk::Item & item = theCrosstalk->item(detId, channel);

  // resistive fraction is at the peak, where t=0
  return leftRight ? item.xtalk_intercept_right 
                   : item.xtalk_intercept_left;
}


float CSCConditions::crosstalkSlope(const CSCDetId&detId, int channel, bool leftRight) const
{
  assert(theCrosstalk != 0);
  const CSCDBCrosstalk::Item & item = theCrosstalk->item(detId, channel);

  // resistive fraction is at the peak, where t=0
  return leftRight ? item.xtalk_slope_right
                   : item.xtalk_slope_left;
}

const CSCDBNoiseMatrix::Item &
CSCConditions::noiseMatrix(const CSCDetId&detId, int channel) const
{
  assert(theNoiseMatrix != 0);
  return theNoiseMatrix->item(detId, channel);
}

//void CSCConditions::fetchNoisifier(const CSCDetId & detId, int istrip)
//{
  //assert(theNoiseMatrix != 0);
  //const CSCDBNoiseMatrix::Item & item = theNoiseMatrix->item(detId, istrip);
//}

