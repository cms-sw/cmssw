#include "CondFormats/CSCObjects/interface/CSCConditions.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/CSCObjects/interface/CSCBadStrips.h"
#include "CondFormats/CSCObjects/interface/CSCBadWires.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

CSCConditions::CSCConditions( const edm::ParameterSet& ps ) 
 :theNoiseMatrix(0),
  theGains(0),
  thePedestals(0),
  theCrosstalk(0),
  theBadStrips(0),
  theBadWires(0),
  readBadChannels_(false),
  theAverageGain( -1.0 )
{
  readBadChannels_ = ps.getParameter<bool>("readBadChannels");
  // initialize #layers = 2808
  badStripWords.resize( 2808, 0 );
  badWireWords.resize( 2808, 0 );
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

  if ( readBadChannels() ) {
  // Bad strip channels
    edm::ESHandle<CSCBadStrips> hBadS;
    es.get<CSCBadStripsRcd>().get( hBadS );
    theBadStrips = hBadS.product();
  // Bad wiregroup channels
    edm::ESHandle<CSCBadWires> hBadW;
    es.get<CSCBadWiresRcd>().get( hBadW );
    theBadWires = hBadW.product();

    //@@    if( badStripsWatcher_.check( es ) ) { 
      fillBadStripWords();
    //@@    }
    //@@    if( badWiresWatcher_.check( es ) ) { 
      fillBadWireWords();
    //@    }

  }

  // Has GainsRcd changed?
  if( gainsWatcher_.check( es ) ) { // Yes...
    theAverageGain = -1.0; // ...reset, so next access will recalculate it
  }


//  print();
}

void CSCConditions::fillBadStripWords(){
  // reset existing values
  badStripWords.assign( 2808, 0 );
  if ( readBadChannels() ) {
    // unpack what we've read from theBadStrips

    // chambers is a vector<BadChamber>
    // channels is a vector<BadChannel>
    // Each BadChamber contains its index (1-468), the no. of bad channels, 
    // and the index within vector<BadChannel> where this chamber's bad channels start.

    CSCIndexer indexer;

    for ( size_t i=0; i<theBadStrips->chambers.size(); ++i ) { // loop over bad chambers
      int indexc = theBadStrips->chambers[i].chamber_index;
      int start =  theBadStrips->chambers[i].pointer;  // where this chamber's bad channels start in vector<BadChannel>
      int nbad  =  theBadStrips->chambers[i].bad_channels;

      CSCDetId id = indexer.detIdFromChamberIndex( indexc ); // We need this to build layer index (1-2808)

      for ( int j=start; j<start+nbad; ++j ) { // bad channels in this chamber
        short lay  = theBadStrips->channels[j].layer;    // value 1-6
        short chan = theBadStrips->channels[j].channel;  // value 1-80
    //    short f1 = theBadStrips->channels[j].flag1;
    //    short f2 = theBadStrips->channels[j].flag2;
    //    short f3 = theBadStrips->channels[j].flag3;
        int indexl = indexer.layerIndex( id.endcap(), id.station(), id.ring(), id.chamber(), lay );
        badStripWords[indexl-1].set( chan, 1 ); // set bit in 80-bit bitset representing this layer
      } // j
    } // i

  } 
}

void CSCConditions::fillBadWireWords(){
  // reset existing values
  badWireWords.assign( 2808, 0 );
  if ( readBadChannels() ) {
    // unpack what we've read from theBadWires
  } 
}

void CSCConditions::print() const
  //@@ NEEDS THOROUGH UPDATING
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
  return float( theGains->item(detId, channel).gain_slope )/theGains->factor_gain;
}


float CSCConditions::pedestal(const CSCDetId & detId, int channel) const
{
  assert(thePedestals != 0);
  return float ( thePedestals->item(detId, channel).ped )/thePedestals->factor_ped;
}


float CSCConditions::pedestalSigma(const CSCDetId&detId, int channel) const
{
  assert(thePedestals != 0);
  return float ( thePedestals->item(detId, channel).rms )/thePedestals->factor_rms;
}


float CSCConditions::crosstalkIntercept(const CSCDetId&detId, int channel, bool leftRight) const
{
  assert(theCrosstalk != 0);
  const CSCDBCrosstalk::Item & item = theCrosstalk->item(detId, channel);

  // resistive fraction is at the peak, where t=0
  return leftRight ? float ( item.xtalk_intercept_right )/theCrosstalk->factor_intercept 
                   : float ( item.xtalk_intercept_left )/theCrosstalk->factor_intercept ;
}


float CSCConditions::crosstalkSlope(const CSCDetId&detId, int channel, bool leftRight) const
{
  assert(theCrosstalk != 0);
  const CSCDBCrosstalk::Item & item = theCrosstalk->item(detId, channel);

  // resistive fraction is at the peak, where t=0
  return leftRight ? float ( item.xtalk_slope_right )/theCrosstalk->factor_slope
                   : float ( item.xtalk_slope_left )/theCrosstalk->factor_slope ;
}

const CSCDBNoiseMatrix::Item & CSCConditions::noiseMatrix(const CSCDetId& detId, int channel) const
{
  assert(theNoiseMatrix != 0);
  return theNoiseMatrix->item(detId, channel);
}

void CSCConditions::noiseMatrixElements( const CSCDetId& id, int channel, std::vector<float>& me ) const {
  assert(me.size()>11);
  const CSCDBNoiseMatrix::Item& item = noiseMatrix(id, channel);
  me[0] = float ( item.elem33 )/theNoiseMatrix->factor_noise;
  me[1] = float ( item.elem34 )/theNoiseMatrix->factor_noise;
  me[2] = float ( item.elem35 )/theNoiseMatrix->factor_noise;
  me[3] = float ( item.elem44 )/theNoiseMatrix->factor_noise;
  me[4] = float ( item.elem45 )/theNoiseMatrix->factor_noise;
  me[5] = float ( item.elem46 )/theNoiseMatrix->factor_noise;
  me[6] = float ( item.elem55 )/theNoiseMatrix->factor_noise;
  me[7] = float ( item.elem56 )/theNoiseMatrix->factor_noise;
  me[8] = float ( item.elem57 )/theNoiseMatrix->factor_noise;
  me[9] = float ( item.elem66 )/theNoiseMatrix->factor_noise;
  me[10] = float ( item.elem67 )/theNoiseMatrix->factor_noise;
  me[11] = float ( item.elem77 )/theNoiseMatrix->factor_noise;
}

void CSCConditions::crossTalk( const CSCDetId& id, int channel, std::vector<float>& ct ) const {
  assert(theCrosstalk != 0);
  const CSCDBCrosstalk::Item & item = theCrosstalk->item(id, channel);
  ct[0] = float ( item.xtalk_slope_left )/theCrosstalk->factor_slope;
  ct[1] = float ( item.xtalk_intercept_left )/theCrosstalk->factor_intercept;
  ct[2] = float ( item.xtalk_slope_right )/theCrosstalk->factor_slope;
  ct[3] = float ( item.xtalk_intercept_right )/theCrosstalk->factor_intercept;
}

const std::bitset<80>& CSCConditions::badStripWord( const CSCDetId& id ) const {
  CSCIndexer indexer;
  return badStripWords[indexer.layerIndex(id)];
}

const std::bitset<112>& CSCConditions::badWireWord( const CSCDetId& id ) const {
  CSCIndexer indexer;
  return badWireWords[indexer.layerIndex(id)];
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
    float the_gain = float( it->gain_slope )/theGains->factor_gain;
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
