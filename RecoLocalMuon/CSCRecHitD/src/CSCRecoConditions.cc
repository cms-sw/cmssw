#include <RecoLocalMuon/CSCRecHitD/src/CSCRecoConditions.h>
#include <CondFormats/CSCObjects/interface/CSCChannelTranslator.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <iostream>

CSCRecoConditions::CSCRecoConditions( const edm::ParameterSet & ps ) : theConditions( ps ) {
}

CSCRecoConditions::~CSCRecoConditions() {
}

void CSCRecoConditions::initializeEvent( const edm::EventSetup& es ) {
  theConditions.initializeEvent( es );
}

/// gains & pedestals are requested by geometric channel (as in CSCStripDigi-
/// e.g. 1-16 for ganged ME1a, and with any readout flips already removed)

float CSCRecoConditions::pedestal(const CSCDetId& id, int geomChannel) const {
  LogTrace("CSCRecoConditions") << id << " geomChannel " << geomChannel << " pedestal " << theConditions.pedestal(id, geomChannel);
  return theConditions.pedestal(id, geomChannel);
}

float CSCRecoConditions::pedestalSigma(const CSCDetId& id, int geomChannel) const {
  return theConditions.pedestalSigma(id, geomChannel);
}

float CSCRecoConditions::gain(const CSCDetId& id, int geomChannel) const {
  LogTrace("CSCRecoConditions") << id << " geomChannel " <<  geomChannel << " gain " << theConditions.gain(id, geomChannel);
  return theConditions.gain(id, geomChannel);
}

/// All other functions are accessed by geometrical strip label (i.e. strip number according to local coordinates)

float CSCRecoConditions::chipCorrection(const CSCDetId & id, int geomStrip) const {
  // geometric strip to geometric channel (e.g. ME1a, 1-48->1-16 ganged or 1-48 unganged)
  int geomChannel = theConditions.channelFromStrip( id, geomStrip );
  return theConditions.chipCorrection(id, geomChannel);
}

//  stripWeights is required in CSCHitFromStripOnly.
// - Has nstrips in arg list because caller already has this value from CSCChamberSpecs.
// - We only have gains per geometric channel of course, and we only apply them by channel too
// (in CSCHitFromStripOnly), but we may as well fill values for each strip.

void CSCRecoConditions::stripWeights( const CSCDetId& id, short int nstrips, float* weights ) const {

  for ( short int i = 1; i < nstrips+1; ++i) {
      weights[i-1] = stripWeight(id, i) ;
  }
}

//  Calculate weight as 1/(gain/average gain)
//  Input is offline CSCDetId (e.g. ir=4 for ME1A), and geom strip # (e.g. 1-48 for ME1A)

float CSCRecoConditions::stripWeight( const CSCDetId& id, int geomStrip ) const {
   int geomChannel = theConditions.channelFromStrip( id, geomStrip );
   float w = averageGain() / gain(id, geomChannel); // averageGain() from CSCConditions
   // Weights are forced to lie within 0.5 and 1.5
   if (w > 1.5) w = 1.5;
   if (w < 0.5) w = 0.5;
   LogTrace("CSCRecoConditions") << id << " geomStrip " << geomStrip << " stripWeight " << w;
   return w;
}

void CSCRecoConditions::noiseMatrix( const CSCDetId& id, int geomStrip, std::vector<float>& nMatrix ) const {

  // nMatrix will be filled with expanded noise matrix elements for strip 'geomStrip' and its immediate neighbours

  nMatrix.clear();

  // These are ME1/2 constants as fall-back
  const float fakeme12[15] = {8.64, 3.47, 2.45, 8.60, 3.28, 1.88, 8.61, 3.18, 1.99, 7.67, 2.64, 0., 7.71, 0., 0.};

  float elem[15];

  for ( short int i = geomStrip-1; i < geomStrip+2; ++i) {

    std::vector<float> me(12);

    float w = stripWeight(id, i);
    w = w*w;
    int geomChannel = theConditions.channelFromStrip( id, i );
    theConditions.noiseMatrixElements(id, geomChannel, me);
    for ( short int j=0; j<11; ++j ) {
      elem[j] = me[j] * w;
    }
    elem[11]= 0.;
    elem[12]= me[11] * w;
    elem[13]= 0.;
    elem[14]= 0.;

    // Test that elements make sense:
    bool isFlawed = false;
    for ( short int k = 0; k < 15; ++k) {
      if (elem[k] < 0.001) elem[k] = 0.001; // fix if too small...
      if (elem[k] > 50.) isFlawed = true; // fail if too big...
    }

    if ( isFlawed ) {
      // These are fake ME1/2:
      for ( short int m = 0; m < 15; ++m ) { elem[m] = fakeme12[m]; }
    }

    for (int k = 0; k < 15; ++k) { nMatrix.push_back( elem[k] ); }
  }
}

void CSCRecoConditions::crossTalk( const CSCDetId& id, int geomStrip, std::vector<float>& xtalks) const {

  // xtalks will be filled with crosstalk for geomStrip and its immediate neighbours

  xtalks.clear();

  for ( short int i = geomStrip-1; i < geomStrip+2; ++i) {
    int geomChannel = theConditions.channelFromStrip( id, i );
    std::vector<float> ct(4);
    theConditions.crossTalk(id, geomChannel, ct);
    xtalks.push_back(ct[0]);
    xtalks.push_back(ct[1]);
    xtalks.push_back(ct[2]);
    xtalks.push_back(ct[3]);
  }
}

///  Is an immediate neighbour a bad strip?
bool CSCRecoConditions::nearBadStrip( const CSCDetId& id, int geomStrip ) const {
  bool nearBad = (badStrip(id,geomStrip-1) || badStrip(id,geomStrip+1));
  return nearBad;
}

/// Is strip itself a bad strip?
bool CSCRecoConditions::badStrip( const CSCDetId& id, int geomStrip ) const {
  //@@ NOT YET UPDATED FOR UNGANGED ME11A

  bool aBadS = false;
  if(geomStrip>0 && geomStrip<81){
    int geomChan = theConditions.channelFromStrip( id, geomStrip );
    const std::bitset<80>& badStrips = theConditions.badStripWord(id);

    int rawChan = theConditions.rawStripChannel( id, geomChan );
    if( rawChan>0 && rawChan<81 ){
      aBadS = badStrips.test(rawChan-1); // 80 bits max, labelled 0-79.

    }
  }
  return aBadS;
}

/// Get bad wiregroup word
const std::bitset<112>& CSCRecoConditions::badWireWord( const CSCDetId& id ) const {
    return theConditions.badWireWord( id );
}

float CSCRecoConditions::chamberTimingCorrection(const CSCDetId & id) const {
  return theConditions.chamberTimingCorrection(id);
}

float CSCRecoConditions::anodeBXoffset(const CSCDetId & id) const {
  return theConditions.anodeBXoffset(id);
}

float CSCRecoConditions::gasGainCorrection(const CSCDetId & id, int geomStrip, int wiregroup ) const {
  int geomChannel = theConditions.channelFromStrip( id, geomStrip);
  return theConditions.gasGainCorrection(id, geomChannel, wiregroup);
}
