#include "CalibMuon/CSCCalibration/interface/CSCConditions.h"

#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperRecord.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerRecord.h"

#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperBase.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerBase.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/DataRecord/interface/CSCDBChipSpeedCorrectionRcd.h"
#include "CondFormats/DataRecord/interface/CSCChamberTimeCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBGasGainCorrectionRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"

#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/CSCObjects/interface/CSCDBChipSpeedCorrection.h"
#include "CondFormats/CSCObjects/interface/CSCChamberTimeCorrections.h"
#include "CondFormats/CSCObjects/interface/CSCDBGasGainCorrection.h"

#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"

#include "CondFormats/CSCObjects/interface/CSCBadStrips.h"
#include "CondFormats/CSCObjects/interface/CSCBadWires.h"
#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"

CSCConditions::CSCConditions( const edm::ParameterSet& ps )
: theGains(), theCrosstalk(), thePedestals(), theNoiseMatrix(),
  theBadStrips(), theBadWires(), theBadChambers(),
  theChipCorrections(), theChamberTimingCorrections(), theGasGainCorrections(),
  indexer_(0), mapper_(0),
  readBadChannels_(false), readBadChambers_(false),
  useTimingCorrections_(false), useGasGainCorrections_(false), theAverageGain( -1.0 )
{
  readBadChannels_ = ps.getParameter<bool>("readBadChannels");
  readBadChambers_ = ps.getParameter<bool>("readBadChambers");
  useTimingCorrections_ = ps.getParameter<bool>("CSCUseTimingCorrections");
  useGasGainCorrections_ = ps.getParameter<bool>("CSCUseGasGainCorrections");

  // set size to hold all layers, using enum defined in .h
  badStripWords.resize( MAX_LAYERS, 0 );
  badWireWords.resize( MAX_LAYERS, 0 );
}


CSCConditions::~CSCConditions(){}

void CSCConditions::initializeEvent(const edm::EventSetup & es)
{
  // Algorithms
  es.get<CSCIndexerRecord>().get( indexer_ );
  es.get<CSCChannelMapperRecord>().get( mapper_ );

  // Strip gains
  es.get<CSCDBGainsRcd>().get( theGains );
  // Strip X-talk
  es.get<CSCDBCrosstalkRcd>().get( theCrosstalk );
  // Strip pedestals
  es.get<CSCDBPedestalsRcd>().get( thePedestals );
  // Strip autocorrelation noise matrix
  es.get<CSCDBNoiseMatrixRcd>().get( theNoiseMatrix );

  if ( useTimingCorrections()){
    // Buckeye chip speeds
    es.get<CSCDBChipSpeedCorrectionRcd>().get( theChipCorrections );
    // Cable lengths from chambers to peripheral crate and additional chamber level timing correction
    es.get<CSCChamberTimeCorrectionsRcd>().get( theChamberTimingCorrections );
  }

  if ( readBadChannels() ) {
  // Bad strip channels
    es.get<CSCBadStripsRcd>().get( theBadStrips );
  // Bad wiregroup channels
    es.get<CSCBadWiresRcd>().get( theBadWires );

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

  if ( readBadChambers() ) {
  // Entire bad chambers
    es.get<CSCBadChambersRcd>().get( theBadChambers );
  }

  if ( useGasGainCorrections()){
    es.get<CSCDBGasGainCorrectionRcd>().get( theGasGainCorrections );
  }

//  print();
}

void CSCConditions::fillBadStripWords(){
  //@@ NOT YET THOUGHT THROUGH FOR UNGANGED ME11A

  // reset existing values
  badStripWords.assign( MAX_LAYERS, 0 );
  if ( readBadChannels() ) {
    // unpack what we've read from theBadStrips

    // chambers is a vector<BadChamber>
    // channels is a vector<BadChannel>
    // Each BadChamber contains its index (1-468 or 540 w. ME42), the no. of bad channels,
    // and the index within vector<BadChannel> where this chamber's bad channels start.

    for ( size_t i=0; i<theBadStrips->chambers.size(); ++i ) { // loop over bad chambers
      int indexc = theBadStrips->chambers[i].chamber_index;
      int start =  theBadStrips->chambers[i].pointer;  // where this chamber's bad channels start in vector<BadChannel>
      int nbad  =  theBadStrips->chambers[i].bad_channels;

      CSCDetId id = indexer_->detIdFromChamberIndex( indexc ); // We need this to build layer index (1-2808)

      for ( int j=start-1; j<start-1+nbad; ++j ) { // bad channels in this chamber
        short lay  = theBadStrips->channels[j].layer;    // value 1-6
        short chan = theBadStrips->channels[j].channel;  // value 1-80
    //    short f1 = theBadStrips->channels[j].flag1;
    //    short f2 = theBadStrips->channels[j].flag2;
    //    short f3 = theBadStrips->channels[j].flag3;
        int indexl = indexer_->layerIndex( id.endcap(), id.station(), id.ring(), id.chamber(), lay );
        badStripWords[indexl-1].set( chan-1, 1 ); // set bit 0-79 in 80-bit bitset representing this layer
      } // j
    } // i

  }
}

void CSCConditions::fillBadWireWords(){
  // reset existing values
  badWireWords.assign( MAX_LAYERS, 0 );
  if ( readBadChannels() ) {
    // unpack what we've read from theBadWires

    for ( size_t i=0; i<theBadWires->chambers.size(); ++i ) { // loop over bad chambers
      int indexc = theBadWires->chambers[i].chamber_index;
      int start =  theBadWires->chambers[i].pointer;  // where this chamber's bad channels start in vector<BadChannel>
      int nbad  =  theBadWires->chambers[i].bad_channels;

      CSCDetId id = indexer_->detIdFromChamberIndex( indexc ); // We need this to build layer index (1-2808)

      for ( int j=start-1; j<start-1+nbad; ++j ) { // bad channels in this chamber
        short lay  = theBadWires->channels[j].layer;    // value 1-6
        short chan = theBadWires->channels[j].channel;  // value 1-80
    //    short f1 = theBadWires->channels[j].flag1;
    //    short f2 = theBadWires->channels[j].flag2;
    //    short f3 = theBadWires->channels[j].flag3;
        int indexl = indexer_->layerIndex( id.endcap(), id.station(), id.ring(), id.chamber(), lay );
        badWireWords[indexl-1].set( chan-1, 1 ); // set bit 0-111 in 112-bit bitset representing this layer
      } // j
    } // i

  }
}

bool CSCConditions::isInBadChamber( const CSCDetId& id ) const {
  //@@ NOT YET THOUGHT THROUGH FOR UNGANGED ME11A

  if ( readBadChambers() )  {
    CSCDetId idraw  = mapper_->rawCSCDetId( id );
    int index = indexer_->chamberIndex( idraw );
    return theBadChambers->isInBadChamber( index );
  }
  else return false;
}

float CSCConditions::gain(const CSCDetId & id, int geomChannel) const
{
  assert(theGains.isValid());
  CSCDetId idraw  = mapper_->rawCSCDetId( id );
  int iraw        = mapper_->rawStripChannel( id, geomChannel );
  int index       = indexer_->stripChannelIndex( idraw, iraw ) - 1; // NOTE THE MINUS ONE!
  return float( theGains->gain(index) ) /theGains->scale();
}

float CSCConditions::pedestal(const CSCDetId & id, int geomChannel) const
{
  assert(thePedestals.isValid());
  CSCDetId idraw  = mapper_->rawCSCDetId( id );
  int iraw        = mapper_->rawStripChannel( id, geomChannel );
  int index       = indexer_->stripChannelIndex( idraw, iraw ) - 1; // NOTE THE MINUS ONE!
  return float( thePedestals->pedestal(index) )/thePedestals->scale_ped();
}


float CSCConditions::pedestalSigma(const CSCDetId& id, int geomChannel) const
{
  assert(thePedestals.isValid());
  CSCDetId idraw  = mapper_->rawCSCDetId( id );
  int iraw        = mapper_->rawStripChannel( id, geomChannel );
  int index       = indexer_->stripChannelIndex( idraw, iraw ) - 1; // NOTE THE MINUS ONE!
  return float( thePedestals->pedestal_rms(index) )/thePedestals->scale_rms();
}


float CSCConditions::crosstalkIntercept(const CSCDetId& id, int geomChannel, bool leftRight) const
{
  assert(theCrosstalk.isValid());
  CSCDetId idraw  = mapper_->rawCSCDetId( id );
  int iraw        = mapper_->rawStripChannel( id, geomChannel );
  int index       = indexer_->stripChannelIndex( idraw, iraw ) - 1; // NOTE THE MINUS ONE!
  // resistive fraction is at the peak, where t=0
  return leftRight ? float ( theCrosstalk->rinter(index) )/theCrosstalk->iscale()
                   : float ( theCrosstalk->linter(index) )/theCrosstalk->iscale() ;
}


float CSCConditions::crosstalkSlope(const CSCDetId& id, int geomChannel, bool leftRight) const
{
  assert(theCrosstalk.isValid());
  CSCDetId idraw  = mapper_->rawCSCDetId( id );
  int iraw        = mapper_->rawStripChannel( id, geomChannel );
  int index       = indexer_->stripChannelIndex( idraw, iraw ) - 1; // NOTE THE MINUS ONE!
  // resistive fraction is at the peak, where t=0
  return leftRight ? float ( theCrosstalk->rslope(index) )/theCrosstalk->sscale()
                   : float ( theCrosstalk->lslope(index) )/theCrosstalk->sscale() ;
}

const CSCDBNoiseMatrix::Item & CSCConditions::noiseMatrix(const CSCDetId& id, int geomChannel) const
{
  //@@ BEWARE - THIS FUNCTION DOES NOT APPLy SCALE FACTOR USED IN PACKING VALUES IN CONDITIONS DATA
  //@@ MAY BE AN ERROR? WHO WOULD WANT ACCESS WITHOUT IT?

  assert(theNoiseMatrix.isValid());
  CSCDetId idraw  = mapper_->rawCSCDetId( id );
  int iraw        = mapper_->rawStripChannel( id, geomChannel );
  int index       = indexer_->stripChannelIndex( idraw, iraw ) - 1; // NOTE THE MINUS ONE!
  return theNoiseMatrix->item(index);
}

void CSCConditions::noiseMatrixElements( const CSCDetId& id, int geomChannel, std::vector<float>& me ) const {
  assert(me.size() > 11 );
  const CSCDBNoiseMatrix::Item& item = noiseMatrix(id, geomChannel); // i.e. the function above
  me[0] = float ( item.elem33 )/theNoiseMatrix->scale();
  me[1] = float ( item.elem34 )/theNoiseMatrix->scale();
  me[2] = float ( item.elem35 )/theNoiseMatrix->scale();
  me[3] = float ( item.elem44 )/theNoiseMatrix->scale();
  me[4] = float ( item.elem45 )/theNoiseMatrix->scale();
  me[5] = float ( item.elem46 )/theNoiseMatrix->scale();
  me[6] = float ( item.elem55 )/theNoiseMatrix->scale();
  me[7] = float ( item.elem56 )/theNoiseMatrix->scale();
  me[8] = float ( item.elem57 )/theNoiseMatrix->scale();
  me[9] = float ( item.elem66 )/theNoiseMatrix->scale();
  me[10] = float ( item.elem67 )/theNoiseMatrix->scale();
  me[11] = float ( item.elem77 )/theNoiseMatrix->scale();
}

void CSCConditions::crossTalk( const CSCDetId& id, int geomChannel, std::vector<float>& ct ) const {
  assert(theCrosstalk.isValid());
  CSCDetId idraw  = mapper_->rawCSCDetId( id );
  int iraw        = mapper_->rawStripChannel( id, geomChannel );
  int index       = indexer_->stripChannelIndex( idraw, iraw ) - 1; // NOTE THE MINUS ONE!

  ct[0] = float ( theCrosstalk->lslope(index) )/theCrosstalk->sscale();
  ct[1] = float ( theCrosstalk->linter(index) )/theCrosstalk->iscale();
  ct[2] = float ( theCrosstalk->rslope(index) )/theCrosstalk->sscale();
  ct[3] = float ( theCrosstalk->rinter(index) )/theCrosstalk->iscale();
}

float CSCConditions::chipCorrection(const CSCDetId & id, int geomChannel) const
{
  if ( useTimingCorrections() ){
    assert(theChipCorrections.isValid());
    CSCDetId idraw  = mapper_->rawCSCDetId( id );
    int iraw        = mapper_->rawStripChannel( id, geomChannel);
    int ichip       = indexer_->chipIndex(iraw); // converts 1-80 to 1-5 (chip#, CFEB#)
    int index       = indexer_->chipIndex(idraw, ichip) - 1; // NOTE THE MINUS ONE!
    return float ( theChipCorrections->value(index) )/theChipCorrections->scale();
  }
  else
    return 0;
}
float CSCConditions::chamberTimingCorrection(const CSCDetId & id) const
{
  if ( useTimingCorrections() ){
    assert(theChamberTimingCorrections.isValid());
    CSCDetId idraw  = mapper_->rawCSCDetId( id );
    int index = indexer_->chamberIndex(idraw) - 1; // NOTE THE MINUS ONE!
    return float (
      theChamberTimingCorrections->item(index).cfeb_tmb_skew_delay*1./theChamberTimingCorrections->precision()
		   + theChamberTimingCorrections->item(index).cfeb_timing_corr*1./theChamberTimingCorrections->precision()
		   + (theChamberTimingCorrections->item(index).cfeb_cable_delay*25. )
);
  }
  else
    return 0;
}
float CSCConditions::anodeBXoffset(const CSCDetId & id) const
{
  if ( useTimingCorrections() ){
    assert(theChamberTimingCorrections.isValid());
    CSCDetId idraw  = mapper_->rawCSCDetId( id );
    int index = indexer_->chamberIndex(idraw) - 1; // NOTE THE MINUS ONE!
    return float ( theChamberTimingCorrections->item(index).anode_bx_offset*1./theChamberTimingCorrections->precision() );
  }
  else
    return 0;
}

const std::bitset<80>& CSCConditions::badStripWord( const CSCDetId& id ) const {
  //@@ NOT YET THOUGHT THROUGH FOR UNGANGED ME11A

  CSCDetId idraw  = mapper_->rawCSCDetId( id );
  return badStripWords[indexer_->layerIndex(idraw) - 1];
}

const std::bitset<112>& CSCConditions::badWireWord( const CSCDetId& id ) const {
  //@@ NEED TO THINK ABOUT THIS SINCE ME11A & ME11B SHARE A COMMON WIRE PLANE
  //@@ SHOULD WE JUST USE ME11B?

  CSCDetId idraw  = mapper_->rawCSCDetId( id );
  return badWireWords[indexer_->layerIndex(idraw) - 1];
}

/// Return average strip gain for full CSC system. Lazy evaluation.
/// Restrict averaging to gains between 5 and 10, and require average
/// is between 6 or 9 otherwise fix it to 7.5.
/// These values came from Dominique and Stan,
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
    float the_gain = float( it->gain_slope )/theGains->scale();
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
//
float CSCConditions::gasGainCorrection( const CSCDetId & id, int geomChannel, int iwiregroup ) const
{
  if ( useGasGainCorrections() ){
    assert(theGasGainCorrections.isValid());
    CSCDetId idraw  = mapper_->rawCSCDetId( id );
    int iraw        = mapper_->rawStripChannel( id, geomChannel );
    int index       = indexer_->gasGainIndex(idraw, iraw, iwiregroup) - 1; // NOTE THE MINUS ONE!
    return float ( theGasGainCorrections->value(index) );
  } else {
    return 1.;
  }
}

int CSCConditions::channelFromStrip( const CSCDetId& id, int geomStrip) const
{ return mapper_->channelFromStrip(id, geomStrip); }

int CSCConditions::rawStripChannel( const CSCDetId& id, int geomChannel) const
{ return mapper_->rawStripChannel( id, geomChannel); }


void CSCConditions::print() const
  //@@ HAS NOT BEEN UPDATED THROUGH SEVERAL VERSIONS OF THE CONDITIONS DATA
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
